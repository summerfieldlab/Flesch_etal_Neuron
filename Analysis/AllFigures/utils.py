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
from scipy.optimize import minimize
from joblib import Parallel, delayed
# from models import MLP, MLP_L2, MLP_rew, MLP_L2_rew
from plotting import plot_grid2, plot_grid3, scatter_mds_2, scatter_mds_3
from copy import deepcopy
import math
from datetime import time
import seaborn as sns


def objective_function(theta, y_true):
    return np.sum((y_true-param_rdm_model(theta))**2) 



def param_rdm_model(theta):
    '''
    generates model rdm from free parameters for 
    compression (rel & irrel dimensions), offset & rotation
    '''
    # unpack parameters 
    c_rel_north,c_rel_south,c_irrel_north,c_irrel_south,a2,ctx = theta
    a1 = 0
    # note: north=90 and south =0 optim
    l,b = np.meshgrid(np.linspace(-2,2,5),np.linspace(-2,2,5))
    b = b.ravel()
    l = l.ravel()
    response_vect = np.concatenate((np.array([(1-c_irrel_north)*b,(1-c_rel_north)*l]),np.array([(1-c_rel_south)*b,(1-c_irrel_south)*l])),axis=1).T

    R1 = np.array([[np.cos(np.deg2rad(a1)),-np.sin(np.deg2rad(a1))],[np.sin(np.deg2rad(a1)),np.cos(np.deg2rad(a1))]])
    R2 = np.array([[np.cos(np.deg2rad(a2)),-np.sin(np.deg2rad(a2))],[np.sin(np.deg2rad(a2)),np.cos(np.deg2rad(a2))]])

    response_vect[:25,:] = response_vect[:25,:] @ R1 
    response_vect[25:,:] = response_vect[25:,:] @ R2 
    ctx_vect = np.zeros((50,1))
    ctx_vect[25:] += ctx
    response_vect = np.concatenate((response_vect,ctx_vect),axis=1)
    rdm = squareform(pdist(response_vect))

    
    # vectorise and scale rdm 
    rvect = rdm[np.triu_indices(50,k=1)].flatten()
    rvect /= np.max(rvect)
    return rvect


def fit_param_rdm_model(y_rdm,theta_init=[0.0,0.0,0.0,0.0,-20.0,0.0],ctx_bounds=(0,2),comp_bounds=(0.0,1.0),phi_bounds=(-90,90)):
    '''
    fits choice model to data, using  L-BFGS-B algorithm
    '''
    y_true = y_rdm[np.triu_indices(50,k=1)].flatten()
    y_true /= np.max(y_true)

    theta_bounds = (comp_bounds,comp_bounds,comp_bounds,comp_bounds,phi_bounds,ctx_bounds)    
    
    results = minimize(fun=objective_function,args=y_true,x0=theta_init,bounds=theta_bounds,method='L-BFGS-B')

    return results.x


def fit_model_randinit(y_true):
        # set starting values:
        theta_init = [
            np.random.uniform(0,1),
            np.random.uniform(0,1),
            np.random.uniform(0,1), 
            np.random.uniform(0,1), 
            np.random.choice(np.arange(-90,91,1)),
            np.random.uniform(0,2)
        ]
        
        # fit model:
        thetas = fit_param_rdm_model(y_true,theta_init=theta_init)
        return thetas 

def wrapper_fit_param_model(y_true,n_iters=100,para_iters=False):    
    if para_iters:
        thetas = Parallel(n_jobs=6,backend='loky',verbose=0)(delayed(fit_model_randinit)(y_true) for i in range(n_iters))
    else:
        thetas = [fit_model_randinit(y_true) for i in range(n_iters)]
    return np.array(thetas).mean(0)


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))







# helper functions
def rotate_axes(x,y,theta):
    # theta is in degrees
    theta_rad = theta * (math.pi/180)  # convert to radians
    x_new = x * math.cos(theta_rad) + y * math.sin(theta_rad)
    y_new =  -x * math.sin(theta_rad) + y * math.cos(theta_rad)
    return x_new, y_new

def rotate(X, theta, axis='x'):
    '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
    theta = theta * (math.pi/180)  # convert to radians
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x': return np.dot(X, np.array([
        [1.,  0,  0],
        [0 ,  c, -s],
        [0 ,  s,  c]
        ]))
    elif axis == 'y': return np.dot(X, np.array([
        [c,  0,  -s],
        [0,  1,   0],
        [s,  0,   c]
        ]))
    elif axis == 'z': return np.dot(X, np.array([
        [c, -s,  0 ],
        [s,  c,  0 ],
        [0,  0,  1.],
        ]))


def helper_jointplot(data,titlestr):

    g = sns.JointGrid(height=4)
    sns.set_style('ticks')
    # joint
    sns.kdeplot(data=data,x=data[:,0],y=data[:,1],fill=True,levels=10,thresh=0,cmap='viridis',ax=g.ax_joint)
    sns.scatterplot(data=data,x=data[:,0],y=data[:,1],ax=g.ax_joint)

    # joint, best fit
    ax = g.ax_joint
    ax.errorbar(np.mean(data[:,0]),np.mean(data[:,1]),color='r',xerr=np.std(data[:,0])/np.sqrt(len(data[:,0])),yerr=np.std(data[:,1])/np.sqrt(len(data[:,1])))
    ax.scatter(np.mean(data[:,0]),np.mean(data[:,1]),color='r')
    ax.plot(np.array([0,1]),np.array([0,1]),'r--')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    # ax.set_xticks(ticks=np.arange(1,182,45))
    # ax.set_xticklabels(labels=np.arange(-90,91,45))
    # ax.set_yticks(ticks=np.arange(0,101,25))
    ax.set_xlabel('Compression, rel. dim.')
    ax.set_ylabel('Compression, irrel. dim.')


    # marginals
    # sns.kdeplot(y=data[:,1],fill=True,ax=g.ax_marg_y,clip=[0,1],color=(0,0,.2))
    # sns.kdeplot(x=data[:,0],fill=True,ax=g.ax_marg_x,clip=[0,1],color=(0,0,.2))
    sns.histplot(y=data[:,1],ax=g.ax_marg_y,color=(0, 0, .2),kde=False)
    sns.histplot(x=data[:,0],ax=g.ax_marg_x,color=(0, 0, .2),kde=False)
    ax = g.ax_marg_x
    ax.set_title(titlestr)
    plt.tight_layout()
    f = plt.gcf()
    f.set_size_inches(4.09, 3.08)

def spiceUp_figure(fh,xlabels):
    plt.figure(fh.number)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    plt.xticks(ticks=np.arange(len(xlabels)),labels=xlabels,rotation=90)
    # plt.xlabel('Cross-Validated ROI',fontsize=6)
    

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
        x_north,y_north,_ = mk_block('north',0)
        y_north = y_north[:,np.newaxis]
        #l_north = (y_north>0).astype('int')
        c_north = np.repeat(np.array([[1,0]]),25,axis=0)

        x_south,y_south, _ = mk_block('south',0)
        y_south = y_south[:,np.newaxis]
        #l_south = (y_south>0).astype('int')
        c_south = np.repeat(np.array([[0,1]]),25,axis=0)

        x_in = np.concatenate((x_north,x_south),axis=0)
        y_rew = np.concatenate((y_north,y_south), axis=0)
        #y_true = np.concatenate((l_north,l_south), axis=0)
        x_ctx = np.concatenate((c_north,c_south), axis=0)
    
    elif whichtask == 'north':
        x_in,y_rew,_ = mk_block('north',0)
        y_rew = y_rew[:,np.newaxis]        
        x_ctx = np.repeat(np.array([[0,1]]),25,axis=0)

    elif whichtask == 'south':
        x_in,y_rew,_ = mk_block('south',0)
        y_rew = y_rew[:,np.newaxis]        
        x_ctx = np.repeat(np.array([[0,1]]),25,axis=0)

    # normalise all inputs:
    x_in = x_in/norm(x_in)
    x_ctx = x_ctx/norm(x_ctx)

    # rename inputs     
    return x_in.T, x_ctx.T, (y_rew).T
    

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

