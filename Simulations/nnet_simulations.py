
import pickle
import numpy as np 
from utils.trainer import *


def  main_expt():
    """
        runs main experiment with rich/lazy dynamics of MLP trained on gaussian blobs
    """
    N_STIM = 25
    N_CTX = 2
    N_HIDDEN = 100 
    N_OUT = 1
    N_RUNS = 30
    N_ITER = 10000
    LRATE = 5e-3 
    SCALE_WHXS = np.asarray([1e-2,1e-1,2e-1,3e-1,4e-1,1,2,3])
    SCALE_WHXC = np.asarray([.5,.5,.5,.5,.5,.8,1.35,1.5]) 
    SCALE_WYH  =  np.repeat(1/(N_HIDDEN),len(SCALE_WHXC))
    N_FACTORS = len(SCALE_WHXS)

    results = run_simulation_diffdims(N_STIM,N_CTX,N_HIDDEN,N_OUT,N_RUNS,N_ITER,LRATE,SCALE_WHXS,SCALE_WHXC,SCALE_WYH,N_FACTORS)


    with open('results_mlp_main_correctacccomp.pkl','wb') as f:
        pickle.dump(results,f)


def l2_expt():
    """
    same as main, but with L2 to push network from lazy into rich regime
    """
    # network shape 
    N_STIM = 25
    N_CTX = 2
    N_HIDDEN = 100
    N_OUT = 1
    N_RUNS = 30
    N_ITER = 10000
    LRATE = 5e-3

    SCALE_WHXS = np.asarray([1]*8)*3
    LAMBDA = np.flip(np.linspace(0,0.1,8))
    SCALE_WHXC = np.repeat(3/2,len(SCALE_WHXS))
    SCALE_WYH  =  np.repeat(1/(N_HIDDEN),len(SCALE_WHXC))
    N_FACTORS = len(SCALE_WHXS)

    results = run_simulation_norm(N_STIM,N_CTX,N_HIDDEN,N_OUT,N_RUNS,N_ITER,LRATE,SCALE_WHXS,SCALE_WHXC,SCALE_WYH,N_FACTORS,LAMBDA)

    with open('results_mlp_l2.pkl','wb') as f:
        pickle.dump(results,f)

def noise_expt():
    """
    impact of input noise on performance of network
    """
    
    N_STIM = 25
    N_CTX = 2
    N_HIDDEN = 100
    N_OUT = 1
    N_RUNS = 30
    N_ITER = 10000
    LRATE = 5e-3
    SCALE_NOISE = np.asarray([float(ii)*1e-2 for ii in range(0,11,1)])
    SCALE_WHXS = np.asarray([1e-2,1e-1,2e-1,3e-1,4e-1,1,2,3])
    SCALE_WHXC = np.asarray([.5,.5,.5,.5,.5,.8,1.35,1.5])
    SCALE_WYH  =  np.repeat(1/(N_HIDDEN),len(SCALE_WHXC))
    N_FACTORS = len(SCALE_WHXS)

    results = run_simulation_noiselevel(N_STIM,N_CTX,N_HIDDEN,N_OUT,N_RUNS,N_ITER,LRATE,SCALE_WHXS,SCALE_WHXC,SCALE_WYH,SCALE_NOISE,N_FACTORS)
    
    results['xlabel_noise'] = SCALE_NOISE
    results['ylabel_wscale'] = SCALE_WHXS
    with open('results_mlp_noise.pkl','wb') as f:
            pickle.dump(results,f)


def lrate_rich_expt():
    """
    impact of learning rate on rich network
    """
    N_STIM = 25
    N_CTX = 2
    N_HIDDEN = 100 
    N_OUT = 1
    N_RUNS = 20
    N_ITER = 10000
    LRATE = [1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,1e-2]
    SCALE_WHXS = 1e-2
    SCALE_WHXC = .5
    SCALE_WYH  =  1/N_HIDDEN
    N_FACTORS = len(LRATE)

    results = run_simulation_difflrs(N_STIM,N_CTX,N_HIDDEN,N_OUT,N_RUNS,N_ITER,LRATE,SCALE_WHXS,SCALE_WHXC,SCALE_WYH,N_FACTORS)

    with open('results_mlp_lr_rich.pkl','wb') as f:
        pickle.dump(results,f)

def lrate_lazy_expt():
    """
    impact of learning rate on lazy network
    """ 

    N_STIM = 25
    N_CTX = 2
    N_HIDDEN = 100 
    N_OUT = 1
    N_RUNS = 20
    N_ITER = 10000
    LRATE = [1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,1e-2]

    SCALE_WHXS = 3
    SCALE_WHXC = 1.5
    SCALE_WYH  =  1/N_HIDDEN
    N_FACTORS = len(LRATE)

    results = run_simulation_difflrs(N_STIM,N_CTX,N_HIDDEN,N_OUT,N_RUNS,N_ITER,LRATE,SCALE_WHXS,SCALE_WHXC,SCALE_WYH,N_FACTORS)

    with open('results_mlp_lr_lazy.pkl','wb') as f:
            pickle.dump(results,f)



if __name__ == "__main__":
    main_expt()
    l2_expt()
    noise_expt()
    lrate_lazy_expt()
    lrate_rich_expt()

