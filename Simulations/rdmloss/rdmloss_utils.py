import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform,pdist
from scipy.stats import multivariate_normal
from numpy.linalg import norm
from sklearn import linear_model
from sklearn.manifold import MDS
from scipy.stats import zscore

def compute_relchange(w0,wt):    
    return (norm(wt.flatten())-norm(w0.flatten()))/norm(w0.flatten())


def tf_zscore(x):
    mean,var = tf.nn.moments(x,axes=[0])
    std = tf.math.sqrt(mean)
    return tf.divide(x-mean,std)

def tf_compute_rdm(X,nconds=50):
    
    ones = tf.ones([nconds,1],dtype=tf.dtypes.float32)
    G = tf.matmul(tf.transpose(X),X)
    DG = tf.expand_dims(tf.diag_part(G),1)
    rdm = tf.matmul(ones,tf.transpose(DG)) -tf.multiply(2.0,G) + tf.matmul(DG,tf.transpose(ones))
    
    return rdm

def tf_compute_lowertriang_flattened(rdm):
    ones = tf.ones_like(rdm)
    mask_lt = tf.cast(tf.linalg.band_part(ones,-1,0)-tf.linalg.band_part(ones,0,0),dtype=tf.dtypes.bool)   
    lt_vect = tf.boolean_mask(rdm,mask_lt)
    return lt_vect

def var_weights(shape,stdval=0.1):
    return tf.Variable(tf.truncated_normal(shape,stddev=stdval))

def var_bias(shape):
    return tf.Variable(tf.constant(.1,shape=shape))
    

def gen2Dgauss(x_mu=.0, y_mu=.0, xy_sigma=.1, n=20):
    xx,yy = np.meshgrid(np.linspace(0, 1, n),np.linspace(0, 1, n))
    gausspdf = multivariate_normal([x_mu,y_mu],[[xy_sigma,0],[0,xy_sigma]])
    x_in = np.empty(xx.shape + (2,))
    x_in[:, :, 0] = xx; x_in[:, :, 1] = yy
    return gausspdf.pdf(x_in)

def mk_block(garden, do_shuffle):
    """
    generates block of experiment

    Input:
      - garden  : 'north' or 'south'
      - do_shuffle: True or False, shuffles  values
    """
    resolution = 5
    n_units = resolution**2
    l, b = np.meshgrid(np.linspace(0.2, .8, 5),np.linspace(0.2, .8, 5))
    b = b.flatten()
    l = l.flatten()
    r_n, r_s = np.meshgrid(np.linspace(-2, 2, 5),np.linspace(-2, 2, 5))
    r_s = r_s.flatten()
    r_n = r_n.flatten()
    val_l, val_b = np.meshgrid(np.linspace(1, 5, 5),np.linspace(1, 5, 5))
    val_b = val_b.flatten()
    val_l = val_l.flatten()

    # plt.figure()
    ii_sub = 1
    blobs = np.empty((25,n_units))
    for ii in range(0,25):
        blob = gen2Dgauss(x_mu=b[ii], y_mu=l[ii],xy_sigma=0.08,n=resolution)
        blob = blob/ np.max(blob)
        ii_sub += 1
        blobs[ii,:] = blob.flatten()
    x1 = blobs
    if garden == 'north':        
        reward = r_n
    elif garden == 'south':        
        reward = r_s

    feature_vals = np.vstack((val_b,val_l)).T
    if do_shuffle:
        ii_shuff = np.random.permutation(25)
        x1 = x1[ii_shuff,:]
        feature_vals = feature_vals[ii_shuff,:]
        reward = reward[ii_shuff]
    return x1, reward, feature_vals



# misc ---------------------------------------------

def compute_sparsity_stats(yout):
    # yout is n_units x n_trials 
    # 1. average within contexts
    x = np.vstack((np.mean(yout[:,0:25],1).T,np.mean(yout[:,25:-1],1).T))
    # should yield a 2xn_hidden vector
    # now count n dead units (i.e. return 0 in both tasks)
    n_dead = np.sum(~np.any(x,axis=0))
    # now count number of local units in total (only active in one task)
    n_local = np.sum(~(np.all(x,axis=0)) & np.any(x,axis=0))
    # count units only active in task a 
    n_only_A = np.sum(np.all(np.vstack((x[0,:]>0,x[1,:]==0)),axis=0))
    # count units only active in task b 
    n_only_B = np.sum(np.all(np.vstack((x[0,:]==0,x[1,:]>0)),axis=0))
    # compute dot product of hiden layer activations 
    h_dotprod = np.dot(x[0,:],x[1,:].T)
    # return all
    return n_dead, n_local, n_only_A, n_only_B, h_dotprod
    


def plot_rdms_mds(results,n_dims=3,runlabel='scale_id',runvalue=np.arange(1,11)):
    
    fig=plt.figure(figsize=(15, 25), dpi= 80, facecolor='w', edgecolor='k')
    n_rows = results['all_x_hidden'].shape[0]
    col_idx = 1
    for ii in range(n_rows):
        plt.subplot(n_rows,5,col_idx+0)
        rdm = squareform(pdist(np.mean(results['all_x_hidden'][ii,:,:,:],0).T,'euclidean'))
        plt.imshow(rdm)
        plt.title(runlabel + ' =' + str(runvalue[ii]) + ' | hidden layer (linearity)')

        plt.subplot(n_rows,5,col_idx+1)
        rdm = squareform(pdist(np.mean(results['all_y_hidden'][ii,:,:,:],0).T,'euclidean'))
        plt.imshow(rdm)
        plt.title(runlabel + ' =' + str(runvalue[ii]) + ' | hidden layer (relu)')

        plt.subplot(n_rows,5,col_idx+2)
        rdm = squareform(pdist(np.mean(results['all_y_out'][ii,:,:,:],0).T))
        plt.imshow(rdm)
        plt.title(runlabel + ' =' + str(runvalue[ii]) + ' | output')

        if n_dims == 3:
            embedding = MDS(n_components=3)
            xyz = embedding.fit_transform(np.mean(results['all_x_hidden'][ii,:,:,:],0).T)
            ax = fig.add_subplot(n_rows,5,col_idx+3, projection='3d')
            ax.scatter(xyz[0:25,0],xyz[0:25,1],xyz[0:25,2],marker='o')
            ax.scatter(xyz[25:,0],xyz[25:,1],xyz[25:,2],marker='^')    
            plt.title('hidden layer (linearity)')

            embedding = MDS(n_components=3)
            xyz = embedding.fit_transform(np.mean(results['all_y_hidden'][ii,:,:,:],0).T)
            ax = fig.add_subplot(n_rows,5,col_idx+4, projection='3d')
            ax.scatter(xyz[0:25,0],xyz[0:25,1],xyz[0:25,2],marker='o')
            ax.scatter(xyz[25:,0],xyz[25:,1],xyz[25:,2],marker='^')    
            plt.title('hidden layer (relu)')
        elif n_dims == 2:
            embedding = MDS(n_components=2)
            xy = embedding.fit_transform(squareform(pdist(np.mean(results['all_x_hidden'][ii,:,:,:],0).T,'euclidean')))
            plt.subplot(n_rows,5,col_idx+3)
            plt.scatter(xy[0:25,0],xy[0:25,1],marker='o')
            plt.scatter(xy[25:,0],xy[25:,1],marker='^')    
            plt.title('hidden layer (linearity)')
            embedding = MDS(n_components=2)
            xy = embedding.fit_transform(squareform(pdist(np.mean(results['all_y_hidden'][ii,:,:,:],0).T,'euclidean')))
            plt.subplot(n_rows,5,col_idx+4)
            plt.scatter(xy[0:25,0],xy[0:25,1],marker='o')
            plt.scatter(xy[25:,0],xy[25:,1],marker='^')    
            plt.title('hidden layer (relu)')

        plt.tight_layout()
        col_idx +=5



def stats_fit_rdms_precomp(dmat,mlp_rdms,runlabel='scale_id',runvalue=np.arange(1,11)):
    # stats 

    regr = linear_model.LinearRegression()
    n_runs = mlp_rdms.shape[1]
    n_factors  =mlp_rdms.shape[0]

    coeffs = np.empty((n_factors,n_runs,3))
    # loop through scaling factors 
    for ii in range(n_factors):
        for jj in range(n_runs):
            rdm = mlp_rdms[ii,jj,:,:]
            y  = zscore(rdm[np.tril_indices(50,k=-1)])
            regr.fit(dmat,y)
            coeffs[ii,jj,:] = np.asarray(regr.coef_)

    coeffs_mu = np.mean(coeffs,1)
    coeffs_err = np.std(coeffs,1)/np.sqrt(n_runs)
    fig=plt.figure(figsize=(10, 5), dpi= 80, facecolor='w', edgecolor='k')
    plt.errorbar(np.arange(1,n_factors+1),coeffs_mu[:,0],yerr=coeffs_err[:,0],color='r')
    plt.errorbar(np.arange(1,n_factors+1),coeffs_mu[:,1],yerr=coeffs_err[:,1],color='g')
    plt.errorbar(np.arange(1,n_factors+1),coeffs_mu[:,2],yerr=coeffs_err[:,2],color='b')
    plt.legend(['grid','orthogonal','parallel'])
    plt.xlabel(runlabel)
    plt.ylabel('beta coefficient')
    plt.title('Model RSA')


def stats_fit_rdms(dmat,mlp_outputs,runlabel='scale_id',runvalue=np.arange(1,11)):
    # stats 

    regr = linear_model.LinearRegression()
    n_runs = mlp_outputs.shape[1]
    n_factors  =mlp_outputs.shape[0]

    coeffs = np.empty((n_factors,n_runs,3))
    # loop through scaling factors 
    for ii in range(n_factors):
        for jj in range(n_runs):
            rdm = squareform(pdist(mlp_outputs[ii,jj,:,:].T))
            y  = zscore(rdm[np.tril_indices(50,k=-1)])
            regr.fit(dmat,y)
            coeffs[ii,jj,:] = np.asarray(regr.coef_)

    coeffs_mu = np.mean(coeffs,1)
    coeffs_err = np.std(coeffs,1)/np.sqrt(n_runs)
    fig=plt.figure(figsize=(10, 5), dpi= 80, facecolor='w', edgecolor='k')
    plt.errorbar(np.arange(1,n_factors+1),coeffs_mu[:,0],yerr=coeffs_err[:,0],color='r')
    plt.errorbar(np.arange(1,n_factors+1),coeffs_mu[:,1],yerr=coeffs_err[:,1],color='g')
    plt.errorbar(np.arange(1,n_factors+1),coeffs_mu[:,2],yerr=coeffs_err[:,2],color='b')
    plt.legend(['grid','orthogonal','parallel'])
    plt.xlabel(runlabel)
    plt.ylabel('beta coefficient')
    plt.title('Model RSA')

def plot_losses(results,y,runlabel='scale_id',runvalue=np.arange(1,11)):
    n_factors = results['all_y_out'].shape[0]
    n_runs = results['all_y_out'].shape[1]
    # plot learning curves (loss)
    l_mu = np.mean(results['all_losses'][:,:,-1],1)
    l_err = np.std(results['all_losses'][:,:,-1],1)/np.sqrt(n_runs)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.errorbar(np.arange(1,n_factors+1),l_mu,yerr=l_err)
    plt.title('endpoint training loss')
    plt.xlabel(runlabel)
    plt.ylabel('loss')
    # plot endpoint accuracy 
    plt.subplot(1,2,2)
    y_ = (results['all_y_out']>0).astype('int')
    y = y>0
    acc = np.empty((n_factors,n_runs))
    for ii in range(n_factors):
        for jj in range(n_runs):
            acc[ii,jj] = np.mean(y_[ii,jj,0,:]==y)
    acc_mu = np.mean(acc,1)
    acc_err = np.std(acc,1)/np.sqrt(n_runs)
    plt.errorbar(np.arange(1,n_factors+1),acc_mu,yerr=acc_err)
    plt.title('endpoint accuracy')
    plt.xlabel(runlabel)
    plt.ylabel('accuracy')
    plt.ylim((.5,1))


def plot_ctxcorrelation(results,runlabel='scale_id',runvalue=np.arange(1,11),coeffs=[]):
       
    if not len(coeffs):
        n_factors = results['all_w_hxc'].shape[0]
        n_runs = results['all_w_hxc'].shape[1]
        coeffs = np.empty((n_factors,n_runs))
        for ii in range(n_factors):
            for jj in range(n_runs):
                c = np.corrcoef(results['all_w_hxc'][ii,jj,:,0],results['all_w_hxc'][ii,jj,:,1])
                coeffs[ii,jj] = c[0,1]

    coeffs_mu = np.mean(coeffs,1)
    coeffs_err = np.std(coeffs,1)/np.sqrt(coeffs.shape[1])
    print(coeffs.shape)
    print(coeffs.shape[1])
    print(coeffs.shape[0])
    print(coeffs_mu.shape)
    print(coeffs_err.shape)
    plt.figure()
    plt.errorbar(np.arange(1,coeffs.shape[0]+1),coeffs_mu,yerr=coeffs_err)
    plt.title('correlation between context weights')
    plt.xlabel(runlabel)
    plt.ylabel('pearson correlation')
    plt.ylim((-1,1))




def gen_modelrdms(monitor=0):
    ## model rdms:
    a,b = np.meshgrid(np.linspace(-2,2,5),np.linspace(-2,2,5))
    # grid model
    gridm = np.concatenate((a.flatten()[np.newaxis,:],b.flatten()[np.newaxis,:]),axis=0).T
    ctx = np.concatenate((np.ones((25,1)),0*np.ones((25,1))),axis=0).reshape(50,1)
    grid_rdm = squareform(pdist(np.concatenate((np.tile(gridm,(2,1)),ctx),axis=1)))


    # orthogonal model
    orthm = np.concatenate((np.concatenate((a.flatten()[np.newaxis,:],np.zeros((1,25))),axis=0).T,
                            np.concatenate((np.zeros((1,25)),b.flatten()[np.newaxis,:]),axis=0).T),axis=0)
    orth_rdm = squareform(pdist(np.concatenate((orthm,ctx),axis=1)))


    # parallel model 
    a = a.flatten()
    b = b.flatten()

    ta = np.stack((a,np.zeros((25))),axis=1)
    tb = np.stack((np.zeros(25),b),axis=1)
    theta = np.radians(-90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    parm = np.concatenate((ta.dot(R),tb),axis=0)
    parm = np.concatenate((parm,ctx),axis=1)
    par_rdm = squareform(pdist(parm,metric='euclidean'))
    
    dmat = np.asarray([zscore(grid_rdm[np.tril_indices(50,k=-1)].flatten()),zscore(orth_rdm[np.tril_indices(50,k=-1)].flatten()),zscore(par_rdm[np.tril_indices(50,k=-1)].flatten())]).T
    rdms = np.empty((3,50,50))
    rdms[0] = grid_rdm
    rdms[1] = orth_rdm
    rdms[2] = par_rdm

    if monitor:
        fig=plt.figure(figsize=(10, 5), dpi= 80, facecolor='w', edgecolor='k')
        labels = ['grid model', 'orthogonal model', 'parallel model']
        for ii in range(3):
            plt.subplot(1,3,ii+1)
            plt.imshow(rdms[ii])
            plt.title(labels[ii])

    return rdms,dmat
