import time
import numpy as np
from scipy.optimize import minimize
from optimparallel import minimize_parallel
from scipy.spatial.distance import squareform, pdist
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")


def objective_function(theta, y_true):
    return np.sum((y_true-param_rdm_model(theta))**2)


def param_rdm_model(theta):
    '''
    generates model rdm from free parameters for 
    compression (rel & irrel dimensions), offset & rotation
    '''
    # unpack parameters
    c_rel_north, c_rel_south, c_irrel_north, c_irrel_south, a2, ctx = theta
    a1 = 0
    # note: north=90 and south =0 optim
    l, b = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    b = b.ravel()
    l = l.ravel()
    response_vect = np.concatenate((np.array(
        [(1-c_irrel_north)*b, (1-c_rel_north)*l]), np.array([(1-c_rel_south)*b, (1-c_irrel_south)*l])), axis=1).T

    R1 = np.array([[np.cos(np.deg2rad(a1)), -np.sin(np.deg2rad(a1))],
                  [np.sin(np.deg2rad(a1)), np.cos(np.deg2rad(a1))]])
    R2 = np.array([[np.cos(np.deg2rad(a2)), -np.sin(np.deg2rad(a2))],
                  [np.sin(np.deg2rad(a2)), np.cos(np.deg2rad(a2))]])

    response_vect[:25, :] = response_vect[:25, :] @ R1
    response_vect[25:, :] = response_vect[25:, :] @ R2
    ctx_vect = np.zeros((50, 1))
    ctx_vect[25:] += ctx
    response_vect = np.concatenate((response_vect, ctx_vect), axis=1)
    rdm = squareform(pdist(response_vect))

    # vectorise and scale rdm
    rvect = rdm[np.triu_indices(50, k=1)].flatten()
    rvect /= np.max(rvect)
    return rvect


def fit_param_rdm_model(y_rdm, theta_init=[0.0, 0.0, 0.0, 0.0, -20.0, 0.0], ctx_bounds=(0, 2), comp_bounds=(0.0, 1.0), phi_bounds=(-90, 90)):
    '''
    fits choice model to data, using Nelder-Mead or L-BFGS-B algorithm
    '''
    y_true = y_rdm[np.triu_indices(50, k=1)].flatten()
    y_true /= np.max(y_true)

    theta_bounds = (comp_bounds, comp_bounds, comp_bounds,
                    comp_bounds, phi_bounds, ctx_bounds)

    results = minimize(objective_function, args=y_true,
                       x0=theta_init, bounds=theta_bounds)

    return results.x


def fit_model_randinit(y_true):
    # set starting values:
    theta_init = [
        np.random.uniform(0, 1),
        np.random.uniform(0, 1),
        np.random.uniform(0, 1),
        np.random.uniform(0, 1),
        np.random.choice(np.arange(-90, 91, 1)),
        np.random.uniform(0, 2)
    ]

    # fit model:
    thetas = fit_param_rdm_model(y_true, theta_init=theta_init)
    return thetas


def wrapper_fit_param_model(y_true, n_iters=100, para_iters=False):
    if para_iters:
        thetas = Parallel(n_jobs=6, backend='loky', verbose=0)(
            delayed(fit_param_rdm_model)(y_true) for i in range(n_iters))
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


if __name__ == "__main__":

    paral_rdm = squareform(param_rdm_model([0, 0, 1, 1, -45, 1]))

    with Timer('parallel minimiser'):
        thetas = wrapper_fit_param_model(
            paral_rdm, n_iters=5000, para_iters=True)
