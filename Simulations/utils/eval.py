
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
from sklearn import linear_model
from scipy.stats import zscore
import torch
import pickle

from utils.nnet import from_gpu


def perform_rsa_cnn(args, model, filepath='../../Data/Simulations/stimuli/'):
    """performs RSA on layers of CNN

    Args:
        args (argparse): parameters
        model (nn.Module): a pytorch feedforward neural network
        filepath (str, optional): directory with train/test datasets. Defaults to '../../Data/Simulations/stimuli/'.

    Returns:
        [type]: [description]
    """
    with open(filepath + 'test_data_north_withgarden.pkl', 'rb') as f:
        d_north = pickle.load(f)

    with open(filepath + 'test_data_south_withgarden.pkl', 'rb') as f:
        d_south = pickle.load(f)

    datasets = [d_north, d_south]
    rdms = []
    outputs = []

    model.to(args.device)
    with torch.no_grad():
        layers = model.get_layers()
        for i in range(len(layers)):
            output = []
            rdm = []
            for d in datasets:
                for b in range(1, 6):
                    for l in range(1, 6):
                        # find patterns for this task
                        X = d['images'][(d['branchiness'] == b) & (
                            d['leafiness'] == l), :].reshape(-1, 96, 96, 3).astype(np.float32)/255.0
                        X = np.transpose(X, [0, 3, 1, 2])

                        X = torch.tensor(X).to(args.device)
                        # pass through network
                        activations = model.get_activations(X)

                        # record average layer-wise outputs
                        y = activations[i].cpu().detach().numpy()
                       # average over exemplars
                        output.append(y.mean(0))

            output = np.array(output)
            outputs.append(output)
            # flatten the filter dimensions
            output =  output.reshape(output.shape[0], np.prod(output.shape[1:]))
            # compute and append rdm
            rdms.append(squareform(pdist(output, metric='euclidean')))
    return outputs, rdms


def compute_accuracy_cnn(y, y_):
    """computes accuracy between predicted and gt class labels, cnn version

    Args:
        y_ (np array): predicted outputs
        y (np array): ground truth outputs

    Returns:
        float: test accuracy
    """
    valid_targets = y != 0
    outputs = y_ > 0
    targets = y > 0
    return from_gpu(torch.mean((outputs[valid_targets] == targets[valid_targets]).float())).ravel()[0]


def compute_accuracy(y_, y):
    """computes accuracy between predicted and gt class labels

    Args:
        y_ (np array): predicted outputs
        y (np array): ground truth outputs

    Returns:
        float: test accuracy
    """
    y = np.ravel(y)
    y_ = np.ravel(y_)
    idcs = y != 0
    y = y[idcs]
    y_ = y_[idcs]
    return np.mean((y_ > 0) == (y > 0))
    

def compute_congruency_acc(cmat,cmat_true):    
    """computes accuracy on congruent and incongruent trials

    Args:
        cmat (np array): choices
        cmat_true (np array): ground truth category labels

    Returns:
        int: accuracies on congruent and incongruent trials
    """
    c = (cmat>0) == (cmat_true>0)
    acc_congruent = (np.mean(c[:2,:2])+np.mean(c[3:,3:]))/2
    acc_incongruent = (np.mean(c[:2,3:])+np.mean(c[3:,:2]))/2
    return acc_congruent, acc_incongruent


def compute_sparsity_stats(yout):
    """ computes various sparsity statistics of the hidden layer

    Args:
        yout (np array): unit-x-feature matrix of hidden layer activity patterns

    Returns:
        n_dead (float): number of dead units
        n_local (float): number of task specific units
        n_only_A (float): number of task A selective units
        n_only_B (float): number of task B selective units 
        h_dotprod (float): dot product of hidden layer activity for task A and B as measure 
        of orthogonality
    """

    x = np.vstack((np.mean(yout[:, 0:25], 1).T, np.mean(yout[:, 25:-1], 1).T))
    n_dead = np.sum(~np.any(x, axis=0))
    n_local = np.sum(~(np.all(x, axis=0)) & np.any(x, axis=0))
    n_only_A = np.sum(np.all(np.vstack((x[0, :] > 0, x[1, :] == 0)), axis=0))
    n_only_B = np.sum(np.all(np.vstack((x[0, :] == 0, x[1, :] > 0)), axis=0))
    h_dotprod = np.dot(x[0, :], x[1, :].T)

    return n_dead, n_local, n_only_A, n_only_B, h_dotprod


def compute_relchange(w0, wt):
    """computes difference in norm of weight vectors between two time points

    Args:
        w0 (np array): weight matrix/vector at t0
        wt (np array): weight matrix/vector at later time point

    Returns:
        float: relative change in norm of weights
    """
    return (norm(wt.flatten())-norm(w0.flatten()))/norm(w0.flatten())


def mse(y_, y):
    '''
    computes mean squared error between targets and outputs
    '''
    return .5*np.linalg.norm(y_-y, 2)**2


def gen_modelrdms():
    # model rdms:
    a, b = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    # grid model
    gridm = np.concatenate(
        (a.flatten()[np.newaxis, :], b.flatten()[np.newaxis, :]), axis=0).T
    ctx = np.concatenate(
        (0*np.ones((25, 1)), 0*np.ones((25, 1))), axis=0).reshape(50, 1)
    gridm = np.concatenate((np.tile(gridm, (2, 1)), ctx), axis=1)
    grid_rdm = squareform(pdist(gridm, metric='euclidean'))

    # orthogonal model
    orthm = np.concatenate((np.concatenate((a.flatten()[np.newaxis, :], np.zeros((1, 25))), axis=0).T,
                            np.concatenate((np.zeros((1, 25)), b.flatten()[np.newaxis, :]), axis=0).T), axis=0)
    orthm = np.concatenate((orthm, ctx), axis=1)
    orth_rdm = squareform(pdist(orthm, metric='euclidean'))

    # parallel model
    theta = np.radians(90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    # parm = np.concatenate((R.dot(np.concatenate((a.flatten()[np.newaxis,:],np.zeros((1,25))),axis=0)).T,
    # np.concatenate((np.zeros((1,25)),b.flatten()[np.newaxis,:]),axis=0).T),axis=0)
    a = a.flatten()
    b = b.flatten()

    ta = np.stack((a, np.zeros((25))), axis=1)
    tb = np.stack((np.zeros(25), b), axis=1)
    theta = np.radians(-90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    parm = np.concatenate((ta.dot(R), tb), axis=0)
    parm = np.concatenate((parm, ctx), axis=1)
    par_rdm = squareform(pdist(parm, metric='euclidean'))

    dmat = np.asarray([zscore(grid_rdm[np.tril_indices(50, k=-1)].flatten()), zscore(
        orth_rdm[np.tril_indices(50, k=-1)].flatten()), zscore(par_rdm[np.tril_indices(50, k=-1)].flatten())]).T
    rdms = np.empty((3, 50, 50))
    data_orig = np.empty((3, 50, 3))
    rdms[0] = grid_rdm
    rdms[1] = orth_rdm
    rdms[2] = par_rdm
    data_orig[0, :, :] = gridm
    data_orig[1, :, :] = orthm
    data_orig[2, :, :] = parm

    return rdms, dmat, data_orig


def stats_fit_rdms(dmat, mlp_outputs):
    # stats

    regr = linear_model.LinearRegression()
    n_runs = mlp_outputs.shape[1]
    n_factors = mlp_outputs.shape[0]

    coeffs = np.empty((n_factors, n_runs, 3))
    # loop through scaling factors
    for ii in range(n_factors):
        for jj in range(n_runs):
            rdm = squareform(
                pdist(mlp_outputs[ii, jj, :, :].T, metric='euclidean'))
            y = zscore(rdm[np.tril_indices(50, k=-1)])
            regr.fit(dmat, y)
            coeffs[ii, jj, :] = np.asarray(regr.coef_)

    coeffs_mu = np.mean(coeffs, 1)
    coeffs_err = np.std(coeffs, 1)/np.sqrt(n_runs)
    return coeffs, coeffs_mu, coeffs_err


def compute_svd_EVs(x):
    """performs SVD and computes normalised eigenvalues of x

    Args:
        x (numpy array): feature-x-conditions array of inputs

    Returns:
        evs (numpy array): eigenvalues of x 
        evs_n (numpy array): normalised eigenvalues of x
    """
    u, s, v = np.linalg.svd(x, full_matrices=False)
    # turn singular values into eigenvalues and normalise
    evs = s**2/(len(s)-1)
    evs_n = np.asarray([ii/np.sum(evs) for ii in evs])
    return evs, evs_n


def reduce_dimensionality(x, n_dims=3):
    """performs SVD for dimensionality reduction

    Args:
        x (numpy array): feature-x-conditions array of inputs
        n_dims (int, optional): number of dimensions to keep. Defaults to 3.

    Returns:
        numpy array: X after removing all but n_dims dimensions
    """
    u, s, v = np.linalg.svd(x, full_matrices=False)
    s[n_dims:] = 0

    return u.dot(np.diag(s)).dot(v)


def compute_svd_betas(h):
    """performs RSA after successive removal of principal components 
       of h in order to estimate intrinsic dimensionality of data manifold

    Args:
        h (numpy arra): unit-x-feature matrix of hidden layer activity

    Returns:
        numpy array: estimated regression coefficients
    """
    _, dmat, _ = gen_modelrdms()
    coeffs = np.empty((h.shape[1], dmat.shape[1]))
    regr = linear_model.LinearRegression()
    for ii in range(1, h.shape[1]+1):
        h_reduced = reduce_dimensionality(h, n_dims=ii)
        rdm = squareform(pdist(h_reduced.T))
        y = zscore(rdm[np.tril_indices(50, k=-1)])
        regr.fit(dmat, y)
        coeffs[ii-1, :] = np.asarray(regr.coef_)

    return coeffs


def compute_svd_acc(nnet, y):
    """computes model accuracy after successive removal of principal components
       from hidden layer activity pattern

    Args:
        nnet (MLP object): instance of trained MLP
        y (numpy array): ground truth labels for test data

    Returns:
        ls (numpy array): losses for different dimensionalities of h
        ys (numpy array): predicted outputs for different dims of h
        accs (numpy array): test accuracies for different dims of h
    """
    h_out_0 = nnet.h_out
    ys = np.empty((h_out_0.shape[1], y.shape[1]))
    ls = np.empty((h_out_0.shape[1]))
    accs = np.empty((h_out_0.shape[1]))
    for ii in range(1, h_out_0.shape[1]+1):
        h_reduced = reduce_dimensionality(h_out_0, n_dims=ii)
        y_ = nnet.sigmoid(nnet.w_yh.dot(h_reduced)+nnet.b_yh)
        ls[ii-1] = nnet.loss(y, y_)
        ys[ii-1, :] = y_
        idcs = y != 0
        accs[ii-1] = np.mean((y_[idcs] > 0) == (y[idcs] > 0))
    return ls, ys, accs
