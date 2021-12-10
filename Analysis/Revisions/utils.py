import pickle
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import expm 
from numpy.linalg import norm
from datetime import datetime
from scipy.spatial.distance import squareform,pdist
from sklearn.utils import shuffle
from sklearn.manifold import MDS
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D 
from scipy.stats import multivariate_normal
from scipy.stats import zscore
from models import MLP, MLP_L2, MLP_rew, MLP_L2_rew
from plotting import plot_grid2, plot_grid3, scatter_mds_2, scatter_mds_3
from copy import deepcopy

# data helper functions ---------------------------------------------------
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
    reward = r_n if garden =='north' else r_s
    # if garden == 'north':        

    # elif garden == 'south':
    #     reward = r_s

    feature_vals = np.vstack((val_b,val_l)).T
    if do_shuffle:
        ii_shuff = np.random.permutation(25)
        x1 = x1[ii_shuff,:]
        feature_vals = feature_vals[ii_shuff,:]
        reward = reward[ii_shuff]
    return x1, reward, feature_vals


def mk_experiment(whichtask='both'):
    # ------------------- Dataset----------------------
    if whichtask=='both':
        x_north,y_north,f_north = mk_block('north',0)
        y_north = y_north[:,np.newaxis]
        #l_north = (y_north>0).astype('int')
        c_north = np.repeat(np.array([[1,0]]),25,axis=0)

        x_south,y_south, f_south = mk_block('south',0)
        y_south = y_south[:,np.newaxis]
        #l_south = (y_south>0).astype('int')
        c_south = np.repeat(np.array([[0,1]]),25,axis=0)

        x_in = np.concatenate((x_north,x_south),axis=0)
        y_rew = np.concatenate((y_north,y_south), axis=0)
        f_in = np.concatenate((f_north,f_south),axis=0)
        #y_true = np.concatenate((l_north,l_south), axis=0)
        x_ctx = np.concatenate((c_north,c_south), axis=0)
    
    elif whichtask == 'north':
        x_in,y_rew,f_in = mk_block('north',0)
        y_rew = y_rew[:,np.newaxis]        
        x_ctx = np.repeat(np.array([[0,1]]),25,axis=0)

    elif whichtask == 'south':
        x_in,y_rew,f_in = mk_block('south',0)
        y_rew = y_rew[:,np.newaxis]        
        x_ctx = np.repeat(np.array([[0,1]]),25,axis=0)

    # normalise all inputs:
    x_in = x_in/norm(x_in)
    x_ctx = x_ctx/norm(x_ctx)

    # rename inputs     
    return x_in.T, x_ctx.T, (y_rew).T*(-1),f_in
    

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
    

def compute_relchange(w0,wt):    
    return (norm(wt.flatten())-norm(w0.flatten()))/norm(w0.flatten())



def gen_modelrdms(monitor=0):
    ## model rdms:
    a,b = np.meshgrid(np.linspace(-2,2,5),np.linspace(-2,2,5))
    # grid model
    gridm = np.concatenate((a.flatten()[np.newaxis,:],b.flatten()[np.newaxis,:]),axis=0).T
    ctx = np.concatenate((np.ones((25,1)),0*np.ones((25,1))),axis=0).reshape(50,1)
    gridm = np.concatenate((np.tile(gridm,(2,1)),ctx),axis=1)
    grid_rdm = squareform(pdist(gridm,metric='euclidean'))


    # orthogonal model
    orthm = np.concatenate((np.concatenate((a.flatten()[np.newaxis,:],np.zeros((1,25))),axis=0).T,
                            np.concatenate((np.zeros((1,25)),b.flatten()[np.newaxis,:]),axis=0).T),axis=0)
    orthm = np.concatenate((orthm,ctx),axis=1)
    orth_rdm = squareform(pdist(orthm,metric='euclidean'))


    # parallel model 
    theta = np.radians(90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    # parm = np.concatenate((R.dot(np.concatenate((a.flatten()[np.newaxis,:],np.zeros((1,25))),axis=0)).T,
                            # np.concatenate((np.zeros((1,25)),b.flatten()[np.newaxis,:]),axis=0).T),axis=0)
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
    data_orig = np.empty((3,50,3))
    rdms[0] = grid_rdm
    rdms[1] = orth_rdm
    rdms[2] = par_rdm
    data_orig[0,:,:] = gridm 
    data_orig[1,:,:] = orthm 
    data_orig[2,:,:] = parm

    if monitor:
        fig=plt.figure(1,figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
        
        labels = ['grid model', 'orthogonal model', 'parallel model']
        for ii in range(3):
            plt.subplot(2,3,ii+1)
            plt.imshow(rdms[ii])
            plt.title(labels[ii])
        for ii in range(3,6):
            ax = fig.add_subplot(2,3,ii+1,projection='3d')
            embedding = MDS(n_components=3,n_init=10,max_iter=1000)
            xyz = data_orig[ii-3]#embedding.fit_transform(data_orig[ii-3])
                  
            plot_grid3(xyz[0:25,:],line_colour=(0, 0, .5),fig_id=1)
            plot_grid3(xyz[25:,:],line_colour='orange',fig_id=1)
            scatter_mds_3(xyz,fig_id=1,task_id='both')
            
            plt.title(labels[ii-3])

    return rdms,dmat,data_orig


def stats_fit_rdms(dmat,mlp_outputs):
    # stats 

    regr = linear_model.LinearRegression()
    n_runs = mlp_outputs.shape[1]
    n_factors  =mlp_outputs.shape[0]

    coeffs = np.empty((n_factors,n_runs,3))
    # loop through scaling factors 
    for ii in range(n_factors):
        for jj in range(n_runs):
            rdm = squareform(pdist(mlp_outputs[ii,jj,:,:].T,metric='euclidean'))
            y  = zscore(rdm[np.tril_indices(50,k=-1)])
            regr.fit(dmat,y)
            coeffs[ii,jj,:] = np.asarray(regr.coef_)


    coeffs_mu = np.mean(coeffs,1)
    coeffs_err = np.std(coeffs,1)/np.sqrt(n_runs)
    return coeffs, coeffs_mu, coeffs_err

    
def run_simulation(N_STIM,N_CTX,N_HIDDEN,N_OUT,N_RUNS,N_ITER,LRATE,SCALE_WHXS,SCALE_WHXC,SCALE_WYH,N_FACTORS):
    x_stim,x_ctx,y = mk_experiment('both')
  

    # ------------------ Simulation -------------------
    

    # init results dict

    results = {
            'all_losses' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_hxc' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_hxs' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_yh' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'all_x_hidden' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,x_stim.shape[1])),
            'all_y_hidden' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,x_stim.shape[1])),
            'all_y_out' : np.empty((N_FACTORS,N_RUNS,1,x_stim.shape[1])),
            'all_w_hxs' :np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_STIM)),
            'all_w_hxc' :np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_CTX)),
            'all_w_yh' : np.empty((N_FACTORS,N_RUNS,N_OUT,N_HIDDEN)),
            'w_hxc_0' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_CTX)),
            'w_hxs_0' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_STIM)),
            'w_yh_0' : np.empty((N_FACTORS,N_RUNS,N_OUT,N_HIDDEN)),
            'n_dead': np.empty((N_FACTORS,N_RUNS,2)),
            'n_local': np.empty((N_FACTORS,N_RUNS,2)),
            'n_only_a': np.empty((N_FACTORS,N_RUNS,2)),
            'n_only_b': np.empty((N_FACTORS,N_RUNS,2)),
            'hidden_dotprod' : np.empty((N_FACTORS,N_RUNS,2)),
            'scale_whxs': SCALE_WHXS,
            'scale_whxc': SCALE_WHXC,
            'scale_wyh': SCALE_WYH,
            'n_hidden': N_HIDDEN,
            'lrate': LRATE

            }

    # loop through variance scaling factors 
    for ii in range(N_FACTORS):
        print((datetime.now().strftime("%d/%m/%Y %H:%M:%S") +  ' - scale factor {} / {}').format(ii+1,N_FACTORS))
        for jj in range(N_RUNS):
            print('... run {} / {}'.format(jj+1,N_RUNS))
            
            # initialise network and get starting values 
            mlp = MLP(n_in=N_STIM,n_ctx=N_CTX,n_hidden=N_HIDDEN,n_out=N_OUT,lrate=LRATE,scale_whxc=SCALE_WHXC[ii],scale_whxs=SCALE_WHXS[ii],scale_wyh=SCALE_WYH[ii])
            results['w_hxs_0'][ii,jj,:,:] = mlp.w_hxs
            results['w_hxc_0'][ii,jj,:,:] = mlp.w_hxc
            results['w_yh_0'][ii,jj,:,:] = mlp.w_yh
            # calculate sparsity and dead units at initialisation   
            mlp.fprop(x_stim,x_ctx,y)
            n_dead,n_local,n_only_a,n_only_b,h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,0] = n_dead
            results['n_local'][ii,jj,0] = n_local 
            results['n_only_a'][ii,jj,0] = n_only_a 
            results['n_only_b'][ii,jj,0] = n_only_b 
            results['hidden_dotprod'][ii,jj,0] = h_dotprod
            
            # train network 
            for kk in range(N_ITER):
                mlp.train(x_stim,x_ctx,y)
                
                # log data 
                results['all_losses'][ii,jj,kk] = mlp.l
                results['w_relchange_hxs'][ii,jj,kk] = compute_relchange(results['w_hxs_0'][ii,jj,:,:],mlp.w_hxs)
                results['w_relchange_hxc'][ii,jj,kk] = compute_relchange(results['w_hxc_0'][ii,jj,:,:],mlp.w_hxc)
                results['w_relchange_yh'][ii,jj,kk] = compute_relchange(results['w_yh_0'][ii,jj,:,:],mlp.w_yh)

            results['all_x_hidden'][ii,jj,:,:] = mlp.h_in
            results['all_y_hidden'][ii,jj,:,:] =  mlp.h_out
            results['all_y_out'][ii,jj,:,:] =  mlp.y_
            results['all_w_hxs'][ii,jj,:,:] =  mlp.w_hxs
            results['all_w_hxc'][ii,jj,:,:] =  mlp.w_hxc
            results['all_w_yh'][ii,jj,:,:] =   mlp.w_yh
            # calculate endpoint sparsity and n_dead 
            n_dead,n_local,n_only_a,n_only_b, h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,1] = n_dead
            results['n_local'][ii,jj,1] = n_local 
            results['n_only_a'][ii,jj,1] = n_only_a 
            results['n_only_b'][ii,jj,1] = n_only_b 
            results['hidden_dotprod'][ii,jj,1] = h_dotprod

    return results


def run_simulation_width(N_STIM,N_CTX,N_HIDDEN,N_OUT,N_RUNS,N_ITER,LRATE,SCALE_WHXS,SCALE_WHXC,SCALE_WYH,N_FACTORS):
    x_stim,x_ctx,y = mk_experiment('both')
  
    # ------------------ Simulation -------------------
    

    # init results dict

    results = {
            'all_losses' : np.empty((len(N_HIDDEN),N_RUNS,N_ITER)),
            'w_relchange_hxc' : np.empty((len(N_HIDDEN),N_RUNS,N_ITER)),
            'w_relchange_hxs' : np.empty((len(N_HIDDEN),N_RUNS,N_ITER)),
            'w_relchange_yh' : np.empty((len(N_HIDDEN),N_RUNS,N_ITER)),
            'rdms_x_hidden' : np.empty((len(N_HIDDEN),N_RUNS,x_stim.shape[1],x_stim.shape[1])),
            'rdms_y_hidden' : np.empty((len(N_HIDDEN),N_RUNS,x_stim.shape[1],x_stim.shape[1])),
            'rdms_y_out' : np.empty((len(N_HIDDEN),N_RUNS,x_stim.shape[1],x_stim.shape[1])),  
            'all_y_out' : np.empty((len(N_HIDDEN),N_RUNS,1,x_stim.shape[1])), 
            'n_dead': np.empty((len(N_HIDDEN),N_RUNS,2)),
            'n_local': np.empty((len(N_HIDDEN),N_RUNS,2)),
            'n_only_a': np.empty((len(N_HIDDEN),N_RUNS,2)),
            'n_only_b': np.empty((len(N_HIDDEN),N_RUNS,2)),
            'hidden_dotprod' : np.empty((len(N_HIDDEN),N_RUNS,2)),
            
            'scale_whxs': SCALE_WHXS,
            'scale_whxc': SCALE_WHXC,
            'scale_wyh': SCALE_WYH,
            'n_hidden': N_HIDDEN,
            'lrate': LRATE,
            'coeffs' : np.empty((len(N_HIDDEN),N_RUNS))

            }

    # loop through variance scaling factors 
    for ii in range(len(N_HIDDEN)):
        print((datetime.now().strftime("%d/%m/%Y %H:%M:%S") +  ' - nnet  {} / {}').format(ii+1,len(N_HIDDEN)))
        for jj in range(N_RUNS):
            print('... run {} / {}'.format(jj+1,N_RUNS))
            
            # initialise network and get starting values 
            mlp = MLP(n_in=N_STIM,n_ctx=N_CTX,n_hidden=N_HIDDEN[ii],n_out=N_OUT,lrate=LRATE,scale_whxc=SCALE_WHXC[0],scale_whxs=SCALE_WHXS[0],scale_wyh=SCALE_WYH[ii])
            w_hxs_0 = mlp.w_hxs
            w_hxc_0 = mlp.w_hxc
            w_yh_0 = mlp.w_yh
            # calculate sparsity and dead units at initialisation   
            mlp.fprop(x_stim,x_ctx,y)
            n_dead,n_local,n_only_a,n_only_b,h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,0] = n_dead
            results['n_local'][ii,jj,0] = n_local 
            results['n_only_a'][ii,jj,0] = n_only_a 
            results['n_only_b'][ii,jj,0] = n_only_b 
            results['hidden_dotprod'][ii,jj,0] = h_dotprod
            # train network 
            for kk in range(N_ITER):
                mlp.train(x_stim,x_ctx,y)
                
                # log data 
                results['all_losses'][ii,jj,kk] = mlp.l
                results['w_relchange_hxs'][ii,jj,kk] = compute_relchange(w_hxs_0,mlp.w_hxs)
                results['w_relchange_hxc'][ii,jj,kk] = compute_relchange(w_hxc_0,mlp.w_hxc)
                results['w_relchange_yh'][ii,jj,kk] = compute_relchange(w_yh_0,mlp.w_yh)

            results['rdms_x_hidden'][ii,jj,:,:] = squareform(pdist(mlp.h_in.T,'euclidean'))
            results['rdms_y_hidden'][ii,jj,:,:] =  squareform(pdist(mlp.h_out.T,'euclidean'))
            results['rdms_y_out'][ii,jj,:,:] =  squareform(pdist(mlp.y_.T,'euclidean'))
            results['all_y_out'][ii,jj,:,:] =  mlp.y_
            c = np.corrcoef(mlp.w_hxc[:,0],mlp.w_hxc[:,1])            
            results['coeffs'][ii,jj] = c[0,1]

            # calculate endpoint sparsity and n_dead 
            n_dead,n_local,n_only_a,n_only_b, h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,1] = n_dead
            results['n_local'][ii,jj,1] = n_local 
            results['n_only_a'][ii,jj,1] = n_only_a 
            results['n_only_b'][ii,jj,1] = n_only_b 
            results['hidden_dotprod'][ii,jj,1] = h_dotprod
            
    return results

def compute_svd_EVs(x):
    u,s,v = np.linalg.svd(x,full_matrices=False)    
    # turn singular values into eigenvalues and normalise 
    evs = s**2/(len(s)-1)
    evs_n = np.asarray([ii/np.sum(evs) for ii in evs])
    return evs, evs_n

def reduce_dimensionality(x,ndims=3):
    u,s,v = np.linalg.svd(x,full_matrices=False)    
    s[ndims:] = 0
    # s[:ndims-1] = 0
    
    return u.dot(np.diag(s)).dot(v)

def compute_svd_betas(h):
    _,dmat,_ = gen_modelrdms()
    coeffs = np.empty((h.shape[1],dmat.shape[1]))
    regr = linear_model.LinearRegression()
    for ii in range(1,h.shape[1]+1):
        h_reduced = reduce_dimensionality(h,ndims=ii)
        rdm = squareform(pdist(h_reduced.T))
        y  = zscore(rdm[np.tril_indices(50,k=-1)])    
        regr.fit(dmat,y)
        coeffs[ii-1,:] = np.asarray(regr.coef_)
    return coeffs 

def compute_svd_acc(nnet,y):
    h_out_0 = nnet.h_out
    ys = np.empty((h_out_0.shape[1],y.shape[1]))
    ls = np.empty((h_out_0.shape[1]))
    accs = np.empty((h_out_0.shape[1]))
    for ii in range(1,h_out_0.shape[1]+1):
        h_reduced = reduce_dimensionality(h_out_0,ndims=ii)
        y_ = nnet.sigmoid(nnet.w_yh.dot(h_reduced)+nnet.b_yh)
        ls[ii-1] = nnet.loss(y,y_)
        ys[ii-1,:] = y_
        # accs[ii-1] = np.mean((np.round(y_,0).astype('int')) == (y))
        # accs[ii-1] = np.mean(((np.round(y_,0).astype('int')) >0) == (y>0))
        idcs = y!=0
        accs[ii-1] = np.mean((y_[idcs] >0.5) == (y[idcs]>0))
    return ls,ys,accs


def compute_accuracy(y_,y):
    y = np.ravel(y)
    y_ = np.ravel(y_)
    idcs = y!=0
    y = y[idcs]
    y_ = y_[idcs]    
    return np.mean((y_>0.5)==(y>0))
    

def run_simulation_diffdims(N_STIM,N_CTX,N_HIDDEN,N_OUT,N_RUNS,N_ITER,LRATE,SCALE_WHXS,SCALE_WHXC,SCALE_WYH,N_FACTORS):
    x_stim,x_ctx,y = mk_experiment('both')
    # y[y<0] = -1
    # y[y>0] = 1

    # ------------------ Simulation -------------------
    

    # init results dict

    results = {
            'all_losses' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'all_accuracies' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_hxc' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_hxs' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_yh' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'all_x_hidden' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,x_stim.shape[1])),
            'all_y_hidden' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,x_stim.shape[1])),
            'all_y_out' : np.empty((N_FACTORS,N_RUNS,1,x_stim.shape[1])),
            'all_w_hxs' :np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_STIM)),
            'all_w_hxc' :np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_CTX)),
            'all_w_yh' : np.empty((N_FACTORS,N_RUNS,N_OUT,N_HIDDEN)),
            'w_hxc_0' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_CTX)),
            'w_hxs_0' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_STIM)),
            'w_yh_0' : np.empty((N_FACTORS,N_RUNS,N_OUT,N_HIDDEN)),
            'n_dead': np.empty((N_FACTORS,N_RUNS,2)),
            'n_local': np.empty((N_FACTORS,N_RUNS,2)),
            'n_only_a': np.empty((N_FACTORS,N_RUNS,2)),
            'n_only_b': np.empty((N_FACTORS,N_RUNS,2)),
            'hidden_dotprod' : np.empty((N_FACTORS,N_RUNS,2)),
            'svd_scree': np.empty((N_FACTORS,N_RUNS,y.shape[1])),
            'svd_betas': np.empty((N_FACTORS,N_RUNS,y.shape[1],3)),
            'svd_loss': np.empty((N_FACTORS,N_RUNS,y.shape[1])),
            'svd_y': np.empty((N_FACTORS,N_RUNS,y.shape[1],y.shape[1])),
            'svd_acc': np.empty((N_FACTORS,N_RUNS,y.shape[1])),
            'scale_whxs': SCALE_WHXS,
            'scale_whxc': SCALE_WHXC,
            'scale_wyh': SCALE_WYH,
            'n_hidden': N_HIDDEN,
            'lrate': LRATE,
            'corrs': np.empty((3,2,N_FACTORS,N_RUNS)),
            'acc_y_ref': np.empty((N_FACTORS,N_RUNS,1)),
            'acc_y_mixed': np.empty((N_FACTORS,N_RUNS,1)),
            'acc_y_local': np.empty((N_FACTORS,N_RUNS,1))
            

            }

    # loop through variance scaling factors 
    for ii in range(N_FACTORS):
        print((datetime.now().strftime("%d/%m/%Y %H:%M:%S") +  ' - scale factor {} / {}').format(ii+1,N_FACTORS))
        for jj in range(N_RUNS):
            print('... run {} / {}'.format(jj+1,N_RUNS))
            
            # initialise network and get starting values 
            mlp = MLP_rew(n_in=N_STIM,n_ctx=N_CTX,n_hidden=N_HIDDEN,n_out=N_OUT,lrate=LRATE,scale_whxc=SCALE_WHXC[ii],scale_whxs=SCALE_WHXS[ii],scale_wyh=SCALE_WYH[ii])
            results['w_hxs_0'][ii,jj,:,:] = mlp.w_hxs
            results['w_hxc_0'][ii,jj,:,:] = mlp.w_hxc
            results['w_yh_0'][ii,jj,:,:] = mlp.w_yh
            # calculate sparsity and dead units at initialisation   
            mlp.fprop(x_stim,x_ctx,y)
            n_dead,n_local,n_only_a,n_only_b,h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,0] = n_dead
            results['n_local'][ii,jj,0] = n_local 
            results['n_only_a'][ii,jj,0] = n_only_a 
            results['n_only_b'][ii,jj,0] = n_only_b 
            results['hidden_dotprod'][ii,jj,0] = h_dotprod
            
            # train network 
            for kk in range(N_ITER):
                mlp.train(x_stim,x_ctx,y)
                
                # log data 
                results['all_losses'][ii,jj,kk] = mlp.l
                results['all_accuracies'][ii,jj,kk] = compute_accuracy(mlp.y_,y)
                results['w_relchange_hxs'][ii,jj,kk] = compute_relchange(results['w_hxs_0'][ii,jj,:,:],mlp.w_hxs)
                results['w_relchange_hxc'][ii,jj,kk] = compute_relchange(results['w_hxc_0'][ii,jj,:,:],mlp.w_hxc)
                results['w_relchange_yh'][ii,jj,kk] = compute_relchange(results['w_yh_0'][ii,jj,:,:],mlp.w_yh)

            results['all_x_hidden'][ii,jj,:,:] = mlp.h_in
            results['all_y_hidden'][ii,jj,:,:] =  mlp.h_out
            results['all_y_out'][ii,jj,:,:] =  mlp.y_
            results['all_w_hxs'][ii,jj,:,:] =  mlp.w_hxs
            results['all_w_hxc'][ii,jj,:,:] =  mlp.w_hxc
            results['all_w_yh'][ii,jj,:,:] =   mlp.w_yh
            # calculate endpoint sparsity and n_dead 
            n_dead,n_local,n_only_a,n_only_b, h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,1] = n_dead
            results['n_local'][ii,jj,1] = n_local 
            results['n_only_a'][ii,jj,1] = n_only_a 
            results['n_only_b'][ii,jj,1] = n_only_b 
            results['hidden_dotprod'][ii,jj,1] = h_dotprod
            # --- dimensionality ---- 
            # calculate endpoint Scree plot 
            _,results['svd_scree'][ii,jj,:] = compute_svd_EVs(mlp.h_out)
            # calculate endpoint model correlations
            results['svd_betas'][ii,jj,:,:] = compute_svd_betas(mlp.h_out)
            # calculate endpoint model accuracy
            results['svd_loss'][ii,jj,:],results['svd_y'][ii,jj,:,:],results['svd_acc'][ii,jj,:] = compute_svd_acc(mlp,y)
            # ----- ctx weight correlations ----------
            yout = results['all_y_hidden'][ii,jj,:,:]
            x = np.vstack((np.nanmean(yout[:,0:25],1).T,np.nanmean(yout[:,25:],1).T))
            # local units
            mask_local = ~(np.all(x,axis=0)) & np.any(x,axis=0)
            mask_a = np.all(np.vstack((x[0,:]>0,x[1,:]==0)),axis=0)
            mask_b = np.all(np.vstack((x[0,:]==0,x[1,:]>0)),axis=0) 
            results['corrs'][0,0,ii,jj] = np.corrcoef(results['w_hxc_0'][ii,jj,mask_local==1,:].T)[0,1]
            results['corrs'][0,1,ii,jj] = np.corrcoef(results['all_w_hxc'][ii,jj,mask_local==1,:].T)[0,1]
            results['corrs'][1,0,ii,jj] = np.corrcoef(results['w_hxc_0'][ii,jj,~mask_local==1,:].T)[0,1]
            results['corrs'][1,1,ii,jj] = np.corrcoef(results['all_w_hxc'][ii,jj,~mask_local==1,:].T)[0,1]
            results['corrs'][2,0,ii,jj] = np.corrcoef(results['w_hxc_0'][ii,jj,:,:].T)[0,1]
            results['corrs'][2,1,ii,jj] = np.corrcoef(results['all_w_hxc'][ii,jj,:,:].T)[0,1]

            # ----- ablation study --------
            yout = results['all_y_hidden'][0,jj,:,:]
            x = np.vstack((np.nanmean(yout[:,0:25],1).T,np.nanmean(yout[:,25:],1).T))
            mask_local = (~(np.all(x,axis=0)) & np.any(x,axis=0))
            mask_mixed = np.all(x,axis=0)
            
            results['acc_y_ref'][ii,jj,:]   = compute_accuracy(mlp.sigmoid(mlp.w_yh.dot(mlp.h_out)+mlp.b_yh) ,y)
            results['acc_y_mixed'][ii,jj,:] = compute_accuracy(mlp.sigmoid(mlp.w_yh.dot(mlp.h_out*mask_mixed[:,np.newaxis])+mlp.b_yh),y)
            results['acc_y_local'][ii,jj,:] = compute_accuracy(mlp.sigmoid(mlp.w_yh.dot(mlp.h_out*mask_local[:,np.newaxis])+mlp.b_yh),y)
    
    # for debugging: save entire network instance 
    results['nnet'] = mlp
    return results





def run_simulation_norm(N_STIM,N_CTX,N_HIDDEN,N_OUT,N_RUNS,N_ITER,LRATE,SCALE_WHXS,SCALE_WHXC,SCALE_WYH,N_FACTORS,LAMBDA):
    x_stim,x_ctx,y = mk_experiment('both')
    # y[y<0] = -1
    # y[y>0] = 1

    # ------------------ Simulation -------------------
    

    # init results dict

    results = {
            'all_losses' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_hxc' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_hxs' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_yh' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'all_x_hidden' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,x_stim.shape[1])),
            'all_y_hidden' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,x_stim.shape[1])),
            'all_y_out' : np.empty((N_FACTORS,N_RUNS,1,x_stim.shape[1])),
            'all_w_hxs' :np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_STIM)),
            'all_w_hxc' :np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_CTX)),
            'all_w_yh' : np.empty((N_FACTORS,N_RUNS,N_OUT,N_HIDDEN)),
            'w_hxc_0' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_CTX)),
            'w_hxs_0' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_STIM)),
            'w_yh_0' : np.empty((N_FACTORS,N_RUNS,N_OUT,N_HIDDEN)),
            'n_dead': np.empty((N_FACTORS,N_RUNS,2)),
            'n_local': np.empty((N_FACTORS,N_RUNS,2)),
            'n_only_a': np.empty((N_FACTORS,N_RUNS,2)),
            'n_only_b': np.empty((N_FACTORS,N_RUNS,2)),
            'hidden_dotprod' : np.empty((N_FACTORS,N_RUNS,2)),
            'scale_whxs': SCALE_WHXS,
            'scale_whxc': SCALE_WHXC,
            'scale_wyh': SCALE_WYH,
            'n_hidden': N_HIDDEN,
            'lrate': LRATE

            }

    # loop through variance scaling factors 
    for ii in range(N_FACTORS):
        print((datetime.now().strftime("%d/%m/%Y %H:%M:%S") +  ' - scale factor {} / {}').format(ii+1,N_FACTORS))
        for jj in range(N_RUNS):
            print('... run {} / {}'.format(jj+1,N_RUNS))
            
            # initialise network and get starting values 
            mlp = MLP_L2_rew(n_in=N_STIM,n_ctx=N_CTX,n_hidden=N_HIDDEN,n_out=N_OUT,lrate=LRATE,scale_whxc=SCALE_WHXC[ii],scale_whxs=SCALE_WHXS[ii],scale_wyh=SCALE_WYH[ii],lmbd=LAMBDA[ii])
            results['w_hxs_0'][ii,jj,:,:] = mlp.w_hxs
            results['w_hxc_0'][ii,jj,:,:] = mlp.w_hxc
            results['w_yh_0'][ii,jj,:,:] = mlp.w_yh
            # calculate sparsity and dead units at initialisation   
            mlp.fprop(x_stim,x_ctx,y)
            n_dead,n_local,n_only_a,n_only_b,h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,0] = n_dead
            results['n_local'][ii,jj,0] = n_local 
            results['n_only_a'][ii,jj,0] = n_only_a 
            results['n_only_b'][ii,jj,0] = n_only_b 
            results['hidden_dotprod'][ii,jj,0] = h_dotprod
            
            # train network 
            for kk in range(N_ITER):
                mlp.train(x_stim,x_ctx,y)
                
                # log data 
                results['all_losses'][ii,jj,kk] = mlp.l
                results['w_relchange_hxs'][ii,jj,kk] = compute_relchange(results['w_hxs_0'][ii,jj,:,:],mlp.w_hxs)
                results['w_relchange_hxc'][ii,jj,kk] = compute_relchange(results['w_hxc_0'][ii,jj,:,:],mlp.w_hxc)
                results['w_relchange_yh'][ii,jj,kk] = compute_relchange(results['w_yh_0'][ii,jj,:,:],mlp.w_yh)

            results['all_x_hidden'][ii,jj,:,:] = mlp.h_in
            results['all_y_hidden'][ii,jj,:,:] =  mlp.h_out
            results['all_y_out'][ii,jj,:,:] =  mlp.y_
            results['all_w_hxs'][ii,jj,:,:] =  mlp.w_hxs
            results['all_w_hxc'][ii,jj,:,:] =  mlp.w_hxc
            results['all_w_yh'][ii,jj,:,:] =   mlp.w_yh
            # calculate endpoint sparsity and n_dead 
            n_dead,n_local,n_only_a,n_only_b, h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,1] = n_dead
            results['n_local'][ii,jj,1] = n_local 
            results['n_only_a'][ii,jj,1] = n_only_a 
            results['n_only_b'][ii,jj,1] = n_only_b 
            results['hidden_dotprod'][ii,jj,1] = h_dotprod

    return results


 
def run_simulation_noiselevel(N_STIM,N_CTX,N_HIDDEN,N_OUT,N_RUNS,N_ITER,LRATE,SCALE_WHXS,SCALE_WHXC,SCALE_WYH,SCALE_NOISE,N_FACTORS):
    x_stim,x_ctx,y = mk_experiment('both')
 

    # ------------------ Simulation -------------------
    # init results dict

    results = {
            'all_losses' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_hxc' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_hxs' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_yh' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'all_x_hidden' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,x_stim.shape[1])),
            'all_y_hidden' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,x_stim.shape[1])),
            'all_y_out' : np.empty((N_FACTORS,N_RUNS,1,x_stim.shape[1])),
            'all_w_hxs' :np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_STIM)),
            'all_w_hxc' :np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_CTX)),
            'all_w_yh' : np.empty((N_FACTORS,N_RUNS,N_OUT,N_HIDDEN)),
            'w_hxc_0' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_CTX)),
            'w_hxs_0' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_STIM)),
            'w_yh_0' : np.empty((N_FACTORS,N_RUNS,N_OUT,N_HIDDEN)),
            'n_dead': np.empty((N_FACTORS,N_RUNS,2)),
            'n_local': np.empty((N_FACTORS,N_RUNS,2)),
            'n_only_a': np.empty((N_FACTORS,N_RUNS,2)),
            'n_only_b': np.empty((N_FACTORS,N_RUNS,2)),
            'hidden_dotprod' : np.empty((N_FACTORS,N_RUNS,2)),
            'scale_whxs': SCALE_WHXS,
            'scale_whxc': SCALE_WHXC,
            'scale_wyh': SCALE_WYH,
            'n_hidden': N_HIDDEN,
            'lrate': LRATE,
            'acc_noise':np.empty((N_FACTORS,N_RUNS,len(SCALE_NOISE),1)),
            'loss_noise':np.empty((N_FACTORS,N_RUNS,len(SCALE_NOISE),1))

            }
   
    # loop through variance scaling factors 
    for ii in range(N_FACTORS):
        print((datetime.now().strftime("%d/%m/%Y %H:%M:%S") +  ' - scale factor {} / {}').format(ii+1,N_FACTORS))
        for jj in range(N_RUNS):
            print('... run {} / {}'.format(jj+1,N_RUNS))
            
            # initialise network and get starting values 
            mlp = MLP_rew(n_in=N_STIM,n_ctx=N_CTX,n_hidden=N_HIDDEN,n_out=N_OUT,lrate=LRATE,scale_whxc=SCALE_WHXC[ii],scale_whxs=SCALE_WHXS[ii],scale_wyh=SCALE_WYH[ii])
            results['w_hxs_0'][ii,jj,:,:] = mlp.w_hxs
            results['w_hxc_0'][ii,jj,:,:] = mlp.w_hxc
            results['w_yh_0'][ii,jj,:,:] = mlp.w_yh
            # calculate sparsity and dead units at initialisation   
            mlp.fprop(x_stim,x_ctx,y)
            n_dead,n_local,n_only_a,n_only_b,h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,0] = n_dead
            results['n_local'][ii,jj,0] = n_local 
            results['n_only_a'][ii,jj,0] = n_only_a 
            results['n_only_b'][ii,jj,0] = n_only_b 
            results['hidden_dotprod'][ii,jj,0] = h_dotprod
            
            # train network 
            for kk in range(N_ITER):
                mlp.train(x_stim,x_ctx,y)
                
                # log data 
                results['all_losses'][ii,jj,kk] = mlp.l
                results['w_relchange_hxs'][ii,jj,kk] = compute_relchange(results['w_hxs_0'][ii,jj,:,:],mlp.w_hxs)
                results['w_relchange_hxc'][ii,jj,kk] = compute_relchange(results['w_hxc_0'][ii,jj,:,:],mlp.w_hxc)
                results['w_relchange_yh'][ii,jj,kk] = compute_relchange(results['w_yh_0'][ii,jj,:,:],mlp.w_yh)

            results['all_x_hidden'][ii,jj,:,:] = mlp.h_in
            results['all_y_hidden'][ii,jj,:,:] =  mlp.h_out
            results['all_y_out'][ii,jj,:,:] =  mlp.y_
            results['all_w_hxs'][ii,jj,:,:] =  mlp.w_hxs
            results['all_w_hxc'][ii,jj,:,:] =  mlp.w_hxc
            results['all_w_yh'][ii,jj,:,:] =   mlp.w_yh
            # calculate endpoint sparsity and n_dead 
            n_dead,n_local,n_only_a,n_only_b, h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,1] = n_dead
            results['n_local'][ii,jj,1] = n_local 
            results['n_only_a'][ii,jj,1] = n_only_a 
            results['n_only_b'][ii,jj,1] = n_only_b 
            results['hidden_dotprod'][ii,jj,1] = h_dotprod
            # test with different noise levels
            for kk,nlvl in enumerate(SCALE_NOISE): 
                noisemat_stim = nlvl*np.random.randn(x_stim.shape[0],x_stim.shape[1])
                noisemat_ctx = nlvl*np.random.randn(x_ctx.shape[0],x_ctx.shape[1])
                mlp.fprop(noisemat_stim+x_stim,noisemat_ctx+x_ctx,y)                
                y_ = mlp.y_                
                idcs = y!=0
                results['acc_noise'][ii,jj,kk,:] = np.mean((y_[idcs]>0.5) == (y[idcs]>0))
                
                results['loss_noise'][ii,jj,kk,:] = mlp.loss(y_,y)

    return results


def run_simulation_noiselevel_hidden(N_STIM,N_CTX,N_HIDDEN,N_OUT,N_RUNS,N_ITER,LRATE,SCALE_WHXS,SCALE_WHXC,SCALE_WYH,SCALE_NOISE,N_FACTORS):
    x_stim,x_ctx,y = mk_experiment('both')
 

    # ------------------ Simulation -------------------
    # init results dict

    results = {
            'all_losses' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_hxc' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_hxs' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_yh' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'all_x_hidden' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,x_stim.shape[1])),
            'all_y_hidden' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,x_stim.shape[1])),
            'all_y_out' : np.empty((N_FACTORS,N_RUNS,1,x_stim.shape[1])),
            'all_w_hxs' :np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_STIM)),
            'all_w_hxc' :np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_CTX)),
            'all_w_yh' : np.empty((N_FACTORS,N_RUNS,N_OUT,N_HIDDEN)),
            'w_hxc_0' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_CTX)),
            'w_hxs_0' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_STIM)),
            'w_yh_0' : np.empty((N_FACTORS,N_RUNS,N_OUT,N_HIDDEN)),
            'n_dead': np.empty((N_FACTORS,N_RUNS,2)),
            'n_local': np.empty((N_FACTORS,N_RUNS,2)),
            'n_only_a': np.empty((N_FACTORS,N_RUNS,2)),
            'n_only_b': np.empty((N_FACTORS,N_RUNS,2)),
            'hidden_dotprod' : np.empty((N_FACTORS,N_RUNS,2)),
            'scale_whxs': SCALE_WHXS,
            'scale_whxc': SCALE_WHXC,
            'scale_wyh': SCALE_WYH,
            'n_hidden': N_HIDDEN,
            'lrate': LRATE,
            'acc_noise':np.empty((N_FACTORS,N_RUNS,len(SCALE_NOISE),1)),
            'loss_noise':np.empty((N_FACTORS,N_RUNS,len(SCALE_NOISE),1)),
            'svd_scree': np.empty((N_FACTORS,N_RUNS,len(SCALE_NOISE),y.shape[1])),
            'svd_betas': np.empty((N_FACTORS,N_RUNS,len(SCALE_NOISE),y.shape[1],3)),
            'svd_loss': np.empty((N_FACTORS,N_RUNS,len(SCALE_NOISE),y.shape[1])),
            'svd_y': np.empty((N_FACTORS,N_RUNS,len(SCALE_NOISE),y.shape[1],y.shape[1])),
            'svd_acc': np.empty((N_FACTORS,N_RUNS,len(SCALE_NOISE),y.shape[1])),
            }

    # loop through variance scaling factors 
    for ii in range(N_FACTORS):
        print((datetime.now().strftime("%d/%m/%Y %H:%M:%S") +  ' - scale factor {} / {}').format(ii+1,N_FACTORS))
        for jj in range(N_RUNS):
            print('... run {} / {}'.format(jj+1,N_RUNS))
            
            # initialise network and get starting values 
            mlp = MLP(n_in=N_STIM,n_ctx=N_CTX,n_hidden=N_HIDDEN,n_out=N_OUT,lrate=LRATE,scale_whxc=SCALE_WHXC[ii],scale_whxs=SCALE_WHXS[ii],scale_wyh=SCALE_WYH[ii])
            results['w_hxs_0'][ii,jj,:,:] = mlp.w_hxs
            results['w_hxc_0'][ii,jj,:,:] = mlp.w_hxc
            results['w_yh_0'][ii,jj,:,:] = mlp.w_yh
            # calculate sparsity and dead units at initialisation   
            mlp.fprop(x_stim,x_ctx,y)
            n_dead,n_local,n_only_a,n_only_b,h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,0] = n_dead
            results['n_local'][ii,jj,0] = n_local 
            results['n_only_a'][ii,jj,0] = n_only_a 
            results['n_only_b'][ii,jj,0] = n_only_b 
            results['hidden_dotprod'][ii,jj,0] = h_dotprod
            
            # train network 
            for kk in range(N_ITER):
                mlp.train(x_stim,x_ctx,y)                
                # log data 
                results['all_losses'][ii,jj,kk] = mlp.l
                results['w_relchange_hxs'][ii,jj,kk] = compute_relchange(results['w_hxs_0'][ii,jj,:,:],mlp.w_hxs)
                results['w_relchange_hxc'][ii,jj,kk] = compute_relchange(results['w_hxc_0'][ii,jj,:,:],mlp.w_hxc)
                results['w_relchange_yh'][ii,jj,kk] = compute_relchange(results['w_yh_0'][ii,jj,:,:],mlp.w_yh)

            results['all_x_hidden'][ii,jj,:,:] = mlp.h_in
            results['all_y_hidden'][ii,jj,:,:] =  mlp.h_out
            results['all_y_out'][ii,jj,:,:] =  mlp.y_
            results['all_w_hxs'][ii,jj,:,:] =  mlp.w_hxs
            results['all_w_hxc'][ii,jj,:,:] =  mlp.w_hxc
            results['all_w_yh'][ii,jj,:,:] =   mlp.w_yh
            # calculate endpoint sparsity and n_dead 
            n_dead,n_local,n_only_a,n_only_b, h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,1] = n_dead
            results['n_local'][ii,jj,1] = n_local 
            results['n_only_a'][ii,jj,1] = n_only_a 
            results['n_only_b'][ii,jj,1] = n_only_b 
            results['hidden_dotprod'][ii,jj,1] = h_dotprod
            # test with different noise levels
            h_out_0 = np.copy(mlp.h_out)
            for kk,nlvl in enumerate(SCALE_NOISE): 
                # --- dimensionality ---- 
                mlp.h_out += nlvl*np.random.randn(mlp.h_out.shape[0],mlp.h_out.shape[1])
                # calculate endpoint Scree plot 
                _,results['svd_scree'][ii,jj,kk,:] = compute_svd_EVs(mlp.h_out)
                # calculate endpoint model correlations
                results['svd_betas'][ii,jj,kk,:,:] = compute_svd_betas(mlp.h_out)
                # calculate endpoint model accuracy                
                results['svd_loss'][ii,jj,kk,:],results['svd_y'][ii,jj,kk,:,:],results['svd_acc'][ii,jj,kk,:] = compute_svd_acc(mlp,y)
                
                noisemat_stim = nlvl*np.random.randn(x_stim.shape[0],x_stim.shape[1])
                noisemat_ctx = nlvl*np.random.randn(x_ctx.shape[0],x_ctx.shape[1])
                mlp.fprop(noisemat_stim+x_stim,noisemat_ctx+x_ctx,y)                
                y_ = mlp.y_                
                idcs = y!=0
                results['acc_noise'][ii,jj,kk,:] = np.mean((y_[idcs]>0) == (y[idcs]>0))                
                results['loss_noise'][ii,jj,kk,:] = mlp.loss(y_,y)
                # reset before I add again more noise
                mlp.h_out = np.copy(h_out_0)

    return results


def mk_testdata_switchcost(n_reps=4,blend_stim=0,blend_ctx=0.1):
    x_stim,x_ctx,y = mk_experiment('both')
    # repeat stimsets to have sufficient trials
    c_test = np.tile(x_ctx,(1,n_reps))
    x_test = np.tile(x_stim,(1,n_reps))
    y_test = np.tile(y,n_reps)
    # on average, about 50% switch trials
    ii_shuff = np.random.permutation(len(y_test.T))
    x_test = x_test[:,ii_shuff]
    c_test = c_test[:,ii_shuff]
    y_test = y_test[:,ii_shuff]
    switch_c_test = np.asarray(np.roll(c_test[0,:]>0,1) != (c_test[0,:]>0))
    switch_x_test = np.asarray(np.roll(x_test[0,:]>0,1) != (x_test[0,:]>0))
    # blend nth inputh with (weighted) n-1th input (unless they were the same)
    for ii in range(1,len(y_test.T)):       
        # c_test[:,ii] = np.mean(c_test[:,ii-1:ii+1],1)
        # x_test[:,ii] = np.mean(x_test[:,ii-1:ii+1],1)
        # c_test[:,ii] += c_test[:,ii-1]*blend_ctx
        # x_test[:,ii] += x_test[:,ii-1]*blend_stim
        if switch_c_test[ii] and switch_x_test[ii]:            
            c_test[:,ii] += c_test[:,ii-1]*blend_ctx
            x_test[:,ii] += x_test[:,ii-1]*blend_stim
            # c_test[:,ii] = np.mean(c_test[:,ii-1:ii+1],1)
            # x_test[:,ii] = np.mean(x_test[:,ii-1:ii+1],1)
        elif switch_c_test[ii] and not switch_x_test[ii]:
            c_test[:,ii] += c_test[:,ii-1]*blend_ctx
            # c_test[:,ii] = np.mean(c_test[:,ii-1:ii+1],1)
        elif switch_x_test[ii]:
            # x_test[:,ii] += x_test[:,ii-1]*blend_stim
            x_test[:,ii] = np.mean(x_test[:,ii-1:ii+1],1)
    return x_test,c_test,y_test,switch_c_test


def mk_intAndDup(x):
    x1 = x[:,:25]
    x2 = x[:,25:]
    xi = np.empty(x.shape)
    xi[:,0::2] = x1
    xi[:,1::2] = x2
    xi = np.repeat(xi,2,axis=1)
    return xi


def mk_switchtestdata(blend_ctx=0.1,blend_stim=0):
    x_north,y_north,f_north = mk_block('north',1)
    y_north = y_north[:,np.newaxis]
    c_north = np.repeat(np.array([[1,0]]),25,axis=0)

    x_south,y_south,f_south = mk_block('south',1)
    y_south = y_south[:,np.newaxis]
    c_south = np.repeat(np.array([[0,1]]),25,axis=0)

    xs = np.concatenate((x_north,x_south),axis=0)
    xc = np.concatenate((c_north,c_south), axis=0)
    fs = np.concatenate((f_north,f_south),axis=0)
    y = np.concatenate((y_north,y_south), axis=0)
    xs = (xs/norm(xs)).T
    xc = (xc/norm(xc)).T
    fs = fs.T
    y = (-1)*y.T
    # duplicate entries 
    xs = mk_intAndDup(xs)
    fs = mk_intAndDup(fs)
    xc = mk_intAndDup(xc)
    y = mk_intAndDup(y)
    xc0 = deepcopy(xc)
    # add sluggishness to ctx signal
    switch_xc = np.asarray(np.roll(xc[0,:]>0,1) != (xc[0,:]>0))
    switch_xs = np.asarray(np.roll(xs[0,:]>0,1) != (xs[0,:]>0))
    # blend nth inputh with (weighted) n-1th input (unless they were the same)
    for ii in range(1,len(y.T)):               
        if switch_xc[ii] and switch_xs[ii]:            
            xc[:,ii] += xc[:,ii-1]*blend_ctx
            xs[:,ii] += xs[:,ii-1]*blend_stim            
        elif switch_xc[ii] and not switch_xs[ii]:
            xc[:,ii] += xc[:,ii-1]*blend_ctx            
        elif switch_xs[ii]:            
            xs[:,ii] = np.mean(xs[:,ii-1:ii+1],1)

    return xs,fs,xc,y,switch_xc,xc0


def compute_acc_switchstay(switch_vect,y_,y):
    y = np.ravel(y)
    y_ = np.ravel(y_)
    idcs = y!=0
    y = y[idcs]
    y_ = y_[idcs]
    switch_vect = switch_vect[idcs]
    acc_switch = np.mean((y_[switch_vect==1]>0)==(y[switch_vect==1]>0))
    acc_stay = np.mean((y_[switch_vect==0]>0)==(y[switch_vect==0]>0))
    return acc_switch,acc_stay

def compute_loss_switchstay(switch_vect,y_,y):
    y = np.ravel(y)
    y_ = np.ravel(y_)
    idcs = y!=0
    y = y[idcs]
    y_ = y_[idcs]
    switch_vect = switch_vect[idcs]
    loss_switch = .5*norm(y_[switch_vect==1]-y[switch_vect==1],2)**2
    loss_stay = .5*norm(y_[switch_vect==0]-y[switch_vect==0],2)**2
    return loss_switch,loss_stay
    
def run_simulation_switch(N_STIM,N_CTX,N_HIDDEN,N_OUT,N_RUNS,N_ITER,LRATE,SCALE_WHXS,SCALE_WHXC,SCALE_WYH,N_FACTORS,BLEND_CTX,BLEND_STIM):
    x_stim,x_ctx,y = mk_experiment('both')
 

    # ------------------ Simulation -------------------
    

    # init results dict

    results = {
            'all_losses' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_hxc' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_hxs' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_yh' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'all_x_hidden' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,x_stim.shape[1])),
            'all_y_hidden' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,x_stim.shape[1])),
            'all_y_out' : np.empty((N_FACTORS,N_RUNS,1,x_stim.shape[1])),
            'all_w_hxs' :np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_STIM)),
            'all_w_hxc' :np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_CTX)),
            'all_w_yh' : np.empty((N_FACTORS,N_RUNS,N_OUT,N_HIDDEN)),
            'w_hxc_0' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_CTX)),
            'w_hxs_0' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_STIM)),
            'w_yh_0' : np.empty((N_FACTORS,N_RUNS,N_OUT,N_HIDDEN)),
            'n_dead': np.empty((N_FACTORS,N_RUNS,2)),
            'n_local': np.empty((N_FACTORS,N_RUNS,2)),
            'n_only_a': np.empty((N_FACTORS,N_RUNS,2)),
            'n_only_b': np.empty((N_FACTORS,N_RUNS,2)),
            'hidden_dotprod' : np.empty((N_FACTORS,N_RUNS,2)),
            'scale_whxs': SCALE_WHXS,
            'scale_whxc': SCALE_WHXC,
            'scale_wyh': SCALE_WYH,
            'n_hidden': N_HIDDEN,
            'lrate': LRATE,
            'acc_switch':np.empty((N_FACTORS,N_RUNS,len(BLEND_STIM),len(BLEND_CTX))),
            'acc_stay':np.empty((N_FACTORS,N_RUNS,len(BLEND_STIM),len(BLEND_CTX))),
            'loss_switch':np.empty((N_FACTORS,N_RUNS,len(BLEND_STIM),len(BLEND_CTX))),
            'loss_stay':np.empty((N_FACTORS,N_RUNS,len(BLEND_STIM),len(BLEND_CTX)))            
            }

    # loop through variance scaling factors 
    for ii in range(N_FACTORS):
        print((datetime.now().strftime("%d/%m/%Y %H:%M:%S") +  ' - scale factor {} / {}').format(ii+1,N_FACTORS))
        for jj in range(N_RUNS):
            print('... run {} / {}'.format(jj+1,N_RUNS))
            
            # initialise network and get starting values 
            mlp = MLP(n_in=N_STIM,n_ctx=N_CTX,n_hidden=N_HIDDEN,n_out=N_OUT,lrate=LRATE,scale_whxc=SCALE_WHXC[ii],scale_whxs=SCALE_WHXS[ii],scale_wyh=SCALE_WYH[ii])
            results['w_hxs_0'][ii,jj,:,:] = mlp.w_hxs
            results['w_hxc_0'][ii,jj,:,:] = mlp.w_hxc
            results['w_yh_0'][ii,jj,:,:] = mlp.w_yh
            # calculate sparsity and dead units at initialisation   
            mlp.fprop(x_stim,x_ctx,y)
            n_dead,n_local,n_only_a,n_only_b,h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,0] = n_dead
            results['n_local'][ii,jj,0] = n_local 
            results['n_only_a'][ii,jj,0] = n_only_a 
            results['n_only_b'][ii,jj,0] = n_only_b 
            results['hidden_dotprod'][ii,jj,0] = h_dotprod
            
            # train network 
            for kk in range(N_ITER):
                mlp.train(x_stim,x_ctx,y)
                
                # log data 
                results['all_losses'][ii,jj,kk] = mlp.l
                results['w_relchange_hxs'][ii,jj,kk] = compute_relchange(results['w_hxs_0'][ii,jj,:,:],mlp.w_hxs)
                results['w_relchange_hxc'][ii,jj,kk] = compute_relchange(results['w_hxc_0'][ii,jj,:,:],mlp.w_hxc)
                results['w_relchange_yh'][ii,jj,kk] = compute_relchange(results['w_yh_0'][ii,jj,:,:],mlp.w_yh)

            results['all_x_hidden'][ii,jj,:,:] = mlp.h_in
            results['all_y_hidden'][ii,jj,:,:] =  mlp.h_out
            results['all_y_out'][ii,jj,:,:] =  mlp.y_
            results['all_w_hxs'][ii,jj,:,:] =  mlp.w_hxs
            results['all_w_hxc'][ii,jj,:,:] =  mlp.w_hxc
            results['all_w_yh'][ii,jj,:,:] =   mlp.w_yh
            # calculate endpoint sparsity and n_dead 
            n_dead,n_local,n_only_a,n_only_b, h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,1] = n_dead
            results['n_local'][ii,jj,1] = n_local 
            results['n_only_a'][ii,jj,1] = n_only_a 
            results['n_only_b'][ii,jj,1] = n_only_b 
            results['hidden_dotprod'][ii,jj,1] = h_dotprod
            # test switch/stay performance
            for kk,bs in enumerate(BLEND_STIM):
                for ll,bc in enumerate(BLEND_CTX):
                    xt,ct,yt,st = mk_testdata_switchcost(blend_stim=bs,blend_ctx=bc)       
                    mlp.fprop(xt,ct,yt)                    
                    y_ = mlp.y_                    
                    results['acc_switch'][ii,jj,kk,ll],results['acc_stay'][ii,jj,kk,ll] = compute_acc_switchstay(st,y_,yt)
                    results['loss_switch'][ii,jj,kk,ll],results['loss_stay'][ii,jj,kk,ll] = compute_loss_switchstay(st,y_,yt)
          
            
            
            # xt,ct,yt,st = mk_testdata_switchcost()       
            # mlp.fprop(xt,ct,yt,monitor=0)
            # y_ = mlp.y_
            # results['acc_switch'][ii,jj],results['acc_stay'][ii,jj] = compute_acc_switchstay(st,y_,yt)
          
    return results



def run_simulation_switch_RSA(N_STIM,N_CTX,N_HIDDEN,N_OUT,N_RUNS,N_ITER,LRATE,SCALE_WHXS,SCALE_WHXC,SCALE_WYH,N_FACTORS,BLEND_CTX,BLEND_STIM):
    x_stim,x_ctx,y = mk_experiment('both')
 

    # ------------------ Simulation -------------------
    

    # init results dict

    results = {
            'all_losses' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_hxc' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_hxs' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'w_relchange_yh' : np.empty((N_FACTORS,N_RUNS,N_ITER)),
            'all_x_hidden' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,x_stim.shape[1])),
            'all_y_hidden' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,x_stim.shape[1])),
            'all_y_out' : np.empty((N_FACTORS,N_RUNS,1,x_stim.shape[1])),
            'all_w_hxs' :np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_STIM)),
            'all_w_hxc' :np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_CTX)),
            'all_w_yh' : np.empty((N_FACTORS,N_RUNS,N_OUT,N_HIDDEN)),
            'w_hxc_0' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_CTX)),
            'w_hxs_0' : np.empty((N_FACTORS,N_RUNS,N_HIDDEN,N_STIM)),
            'w_yh_0' : np.empty((N_FACTORS,N_RUNS,N_OUT,N_HIDDEN)),
            'n_dead': np.empty((N_FACTORS,N_RUNS,2)),
            'n_local': np.empty((N_FACTORS,N_RUNS,2)),
            'n_only_a': np.empty((N_FACTORS,N_RUNS,2)),
            'n_only_b': np.empty((N_FACTORS,N_RUNS,2)),
            'hidden_dotprod' : np.empty((N_FACTORS,N_RUNS,2)),
            'scale_whxs': SCALE_WHXS,
            'scale_whxc': SCALE_WHXC,
            'scale_wyh': SCALE_WYH,
            'n_hidden': N_HIDDEN,
            'lrate': LRATE,
            'acc_switch':np.empty((N_FACTORS,N_RUNS,len(BLEND_STIM),len(BLEND_CTX))),
            'acc_stay':np.empty((N_FACTORS,N_RUNS,len(BLEND_STIM),len(BLEND_CTX))),
            'loss_switch':np.empty((N_FACTORS,N_RUNS,len(BLEND_STIM),len(BLEND_CTX))),
            'loss_stay':np.empty((N_FACTORS,N_RUNS,len(BLEND_STIM),len(BLEND_CTX))),            
            'rdms_hidden_switch':np.empty((N_FACTORS,N_RUNS,len(BLEND_STIM),len(BLEND_CTX),50,50)),
            'rdms_hidden_stay':np.empty((N_FACTORS,N_RUNS,len(BLEND_STIM),len(BLEND_CTX),50,50)),
            'rdms_out_switch':np.empty((N_FACTORS,N_RUNS,len(BLEND_STIM),len(BLEND_CTX),50,50)),
            'rdms_out_stay':np.empty((N_FACTORS,N_RUNS,len(BLEND_STIM),len(BLEND_CTX),50,50)),
            'responses_stay':np.empty((N_FACTORS,N_RUNS,len(BLEND_STIM),len(BLEND_CTX),2,5,5)),
            'responses_switch':np.empty((N_FACTORS,N_RUNS,len(BLEND_STIM),len(BLEND_CTX),2,5,5))
            }

    # loop through variance scaling factors 
    for ii in range(N_FACTORS):
        print((datetime.now().strftime("%d/%m/%Y %H:%M:%S") +  ' - scale factor {} / {}').format(ii+1,N_FACTORS))
        for jj in range(N_RUNS):
            print('... run {} / {}'.format(jj+1,N_RUNS))
            
            # initialise network and get starting values 
            mlp = MLP(n_in=N_STIM,n_ctx=N_CTX,n_hidden=N_HIDDEN,n_out=N_OUT,lrate=LRATE,scale_whxc=SCALE_WHXC[ii],scale_whxs=SCALE_WHXS[ii],scale_wyh=SCALE_WYH[ii])
            results['w_hxs_0'][ii,jj,:,:] = mlp.w_hxs
            results['w_hxc_0'][ii,jj,:,:] = mlp.w_hxc
            results['w_yh_0'][ii,jj,:,:] = mlp.w_yh
            # calculate sparsity and dead units at initialisation   
            mlp.fprop(x_stim,x_ctx,y)
            n_dead,n_local,n_only_a,n_only_b,h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,0] = n_dead
            results['n_local'][ii,jj,0] = n_local 
            results['n_only_a'][ii,jj,0] = n_only_a 
            results['n_only_b'][ii,jj,0] = n_only_b 
            results['hidden_dotprod'][ii,jj,0] = h_dotprod
            
            # train network 
            for kk in range(N_ITER):
                mlp.train(x_stim,x_ctx,y)
                
                # log data 
                results['all_losses'][ii,jj,kk] = mlp.l
                results['w_relchange_hxs'][ii,jj,kk] = compute_relchange(results['w_hxs_0'][ii,jj,:,:],mlp.w_hxs)
                results['w_relchange_hxc'][ii,jj,kk] = compute_relchange(results['w_hxc_0'][ii,jj,:,:],mlp.w_hxc)
                results['w_relchange_yh'][ii,jj,kk] = compute_relchange(results['w_yh_0'][ii,jj,:,:],mlp.w_yh)

            results['all_x_hidden'][ii,jj,:,:] = mlp.h_in
            results['all_y_hidden'][ii,jj,:,:] =  mlp.h_out
            results['all_y_out'][ii,jj,:,:] =  mlp.y_
            results['all_w_hxs'][ii,jj,:,:] =  mlp.w_hxs
            results['all_w_hxc'][ii,jj,:,:] =  mlp.w_hxc
            results['all_w_yh'][ii,jj,:,:] =   mlp.w_yh
            # calculate endpoint sparsity and n_dead 
            n_dead,n_local,n_only_a,n_only_b, h_dotprod = compute_sparsity_stats(mlp.h_out)
            results['n_dead'][ii,jj,1] = n_dead
            results['n_local'][ii,jj,1] = n_local 
            results['n_only_a'][ii,jj,1] = n_only_a 
            results['n_only_b'][ii,jj,1] = n_only_b 
            results['hidden_dotprod'][ii,jj,1] = h_dotprod
            
            # test switch/stay performance and RSA
            # for kk,bs in enumerate(BLEND_STIM):
            #     for ll,bc in enumerate(BLEND_CTX):
            #         xt,ct,yt,st = mk_testdata_switchcost(blend_stim=bs,blend_ctx=bc)       
            #         mlp.fprop(xt,ct,yt)                    
            #         y_ = mlp.y_                    
            #         results['acc_switch'][ii,jj,kk,ll],results['acc_stay'][ii,jj,kk,ll] = compute_acc_switchstay(st,y_,yt)
            #         results['loss_switch'][ii,jj,kk,ll],results['loss_stay'][ii,jj,kk,ll] = compute_loss_switchstay(st,y_,yt)
          
            
            
            for kk,bs in enumerate(BLEND_STIM):
                for ll,bc in enumerate(BLEND_CTX):                    
                    xs,fs,xc,yt,st,xc0 = mk_switchtestdata(blend_ctx=bc,blend_stim=bs)
                    t = np.asarray([xc0[0,:]== np.max(xc0)])
                    mlp.fprop(xs,xc,yt)                    
                    y_ = mlp.y_  
                    y_hidden = mlp.h_out  
                    
                    # log accuracy/loss 
                    results['acc_switch'][ii,jj,kk,ll],results['acc_stay'][ii,jj,kk,ll] = compute_acc_switchstay(st,y_,yt)
                    results['loss_switch'][ii,jj,kk,ll],results['loss_stay'][ii,jj,kk,ll] = compute_loss_switchstay(st,y_,yt)

                    # perform RSA switch/stay
                    # 1. hidden layer
                    respmat = np.empty((2,50,N_HIDDEN))                    
                    for si in range(2):
                        idx = 0
                        for ti in range(2):
                            for bi in range(5):
                                for li in range(5):
                                    try:
                                        respmat[si,idx,:] =y_hidden[:,np.ravel((fs[0,:]==bi+1) & (fs[1,:]==li+1) & (st==si) & (t==ti))].flatten()                                        
                                    except ValueError as e:
                                        
                                        print(y_hidden[0,np.ravel((fs[0,:]==bi+1) & (fs[1,:]==li+1) & (st==si) & (t==ti))].shape)
                                    idx +=1

                    results['rdms_hidden_stay'][ii,jj,kk,ll,:,:] = squareform(pdist(respmat[0,:,:]))
                    results['rdms_hidden_switch'][ii,jj,kk,ll,:,:] = squareform(pdist(respmat[1,:,:]))
                    # 2. output 
                    respmat = np.empty((2,2,5,5,1))
                    ri = 0
                    for si in range(2):
                        for ti in range(2):
                            for bi in range(5):
                                for li in range(5):
                                    respmat[si,ti,bi,li,:] =y_[:,np.ravel((fs[0,:]==bi+1) & (fs[1,:]==li+1) & (st==si) & (t==ti))].flatten()
                    results['rdms_out_stay'][ii,jj,kk,ll,:,:] = squareform(pdist(respmat[0,:,:,:].reshape(50,respmat.shape[-1])))
                    results['rdms_out_switch'][ii,jj,kk,ll,:,:] = squareform(pdist(respmat[1,:,:,:].reshape(50,respmat.shape[-1])))
                    results['responses_stay'][ii,jj,kk,ll,:,:,:] = respmat[0,:,:,:,0]
                    results['responses_switch'][ii,jj,kk,ll,:,:,:] = respmat[1,:,:,:,0]
    return results

