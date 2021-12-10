import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import torch
from numpy.linalg import norm
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
import pickle 
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from joblib import Parallel, delayed
import os

def load_dataset(phase='training', filepath='../../Data/Simulations/', shuffle=False, task='both'):
    if task == 'both':
        # north task
        with open(filepath + phase+'_data_north_withgarden.pkl', 'rb') as f:
            data_n = pickle.load(f)
        # south task
        with open(filepath + phase+'_data_south_withgarden.pkl', 'rb') as f:
            data_s = pickle.load(f)
        X = np.concatenate((data_n['images'].reshape(len(data_n['images']), 96, 96, 3), data_s['images'].reshape(
            len(data_s['images']), 96, 96, 3)), axis=0).astype(np.float32)/255.0
        # target labels
        y = np.concatenate(
            (data_n['rewards'][:, 0], data_s['rewards'][:, 0]), axis=0).astype(np.float32)
        
    else:
        with open(filepath + phase+'_data_' + task + '_withgarden.pkl', 'rb') as f:
            data = pickle.load(f)
        X = data['images'].reshape(
            len(data['images']), 96, 96, 3).astype(np.float32)/255.0
        y = data['rewards'][:, 0].astype(np.float32)

    # bring colour dim to front
    X = np.transpose(X, [0, 3, 1, 2])
    sc = StandardScaler()
    y = sc.fit_transform(y.reshape(-1, 1))
    if shuffle:
        # shuffle data from task A and B
        ii_shuff = np.random.permutation(len(y))
        X = X[ii_shuff, :]
        y = y[ii_shuff]
    return X, y


def gen_tensordataset(X, y):
    # turn into Pytorch dataset and also
    return TensorDataset(torch.tensor(X), torch.tensor(y))


def gen_dataloader(tsdataset, shuffle=True, bs=128):
    """creates a DataLoader object from a provided TensorDataset

    Args:
        tsdataset (TensorDataset): a pytorch tensordataset
        shuffle (bool, optional): whether or not to shuffle the exemplars. Defaults to True.
        bs (int, optional): batch size. Defaults to 128.

    Returns:
        DataLoader: a pytorch dataloader object
    """
    return DataLoader(tsdataset, batch_size=bs, shuffle=shuffle)


def gen_datasets(bs_train=128,bs_test=128,filepath='../../Data/Simulations/'):
    X_train, y_train = load_dataset(phase='training', shuffle=True,filepath=filepath)
    # training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.25)
    # test sets (separate for task A and B)
    X_test_n, y_test_n = load_dataset(phase='test',task='north',shuffle=False,filepath=filepath)
    X_test_s, y_test_s = load_dataset(phase='test',task='south',shuffle=False,filepath=filepath)
    # create data loaders 
    dl_train = gen_dataloader(gen_tensordataset(X_train,y_train),bs=bs_train)
    dl_val = gen_dataloader(gen_tensordataset(X_val, y_val),bs=bs_test)
    dl_tn  = gen_dataloader(gen_tensordataset(X_test_n,y_test_n),shuffle=False,bs=bs_test)
    dl_ts = gen_dataloader(gen_tensordataset(X_test_s,y_test_s),shuffle=False,bs=bs_test)
    return dl_train, dl_val, dl_tn, dl_ts




def mk_datasets_trees(doSuperimpose=1, num_jobs=6, numExemplars_traintest=[400, 200], path_trees='img/trees', path_gardens='img/contexts/', path_data='datasets/'):
    """
    generates datasets for CNN experiments with
    fractal tree images
    """

    # 1. load images and store all as vectors in file
    for gardenID in ['north', 'south']:
        for jj, expPhase in enumerate(['training', 'test']):
            numExemplars = numExemplars_traintest[jj]
            # process images
            results = Parallel(n_jobs=num_jobs)(delayed(helper_makeImgSet)(
                ii, doSuperimpose, gardenID, expPhase, path_trees, path_gardens) for ii in range(numExemplars))
            # stack em all
            allImgs = np.vstack(results)

            fileName = path_data + 'treeSet_' + gardenID + '_' + expPhase + \
                '_withgarden' if doSuperimpose else 'treeSet_' + gardenID + '_' + expPhase
            with open(fileName + '.pkl', 'wb') as f:
                pickle.dump(allImgs, f)

            # add vectors with stimulus and response information
            # 1. feature levels
            [leafiness, branchiness] = np.meshgrid(
                np.linspace(1, 5, 5), np.linspace(1, 5, 5))
            branchiness = np.tile(branchiness.reshape(25,), numExemplars)
            leafiness = np.tile(leafiness.reshape(25,), numExemplars)
            # 2. exemplar ids
            exemplars = np.repeat(np.expand_dims(
                np.arange(numExemplars, dtype='int'), axis=0), 25)
            # 3. context
            context = np.ones((numExemplars*25,), 'int') if (gardenID ==
                                                             'north') else np.ones((numExemplars*25,), 'int')*2
            # 4. rewards (c+,c-,d+,d-)
            rewards = genRewardVect(gardenID, numExemplars)
            # 5. categories (c+,c-,d+,d-)
            categories = genCategoryVect(rewards)
            # collect everything in one dictionary
            data = {'images':         allImgs,
                    'branchiness': branchiness,
                    'leafiness':    leafiness,
                    'exemplars':    exemplars,
                    'contexts':       context,
                    'rewards':        rewards,
                    'categories':   categories}
            # dump dict to file
            fileName = path_data + expPhase + '_data_' + gardenID + \
                '_withgarden' if doSuperimpose else expPhase + '_data_' + gardenID
            with open(fileName + '.pkl', 'wb') as f:
                pickle.dump(data, f)


def helper_makeImgSet(exemplarID, doSuperimpose, gardenID, taskID, path_trees, path_gardens):
    imgSet = np.array([], dtype='uint8')
    print('processing exemplar_set {}'.format(exemplarID))
    for ii_b in range(5):
        for ii_l in range(5):
            img = Image.open(os.path.join(path_trees + '_' + taskID + '/',
                             'B'+str(ii_b+1)+'L'+str(ii_l+1)+'_'+str(exemplarID)+'.png'))
            # resize image
            img = img.resize((96, 96))
            img = img.convert('RGBA')
            if doSuperimpose:
                # load garden image
                gardenIMG = Image.open(os.path.join(
                    path_gardens, gardenID + '_garden.png'))
                # preproc garden image
                gardenIMG = gardenIMG.resize((96, 96))
                gardenIMG = gardenIMG.convert('RGBA')
                # paste tree onto garden
                img = img.convert('RGBA')
                gardenIMG.paste(img, (10, 10), img)
                img = gardenIMG

            # transform image to rgb
            img = img.convert('RGB')
            # transform image to greyscale
            # img = img.convert('L')
            # convert to uint8 to save space
            img = np.asarray(img, dtype='uint8')
            # add to collection
            imgSet = np.concatenate((imgSet, np.expand_dims(img.flatten(
            ), axis=0)), axis=0) if imgSet.size else np.expand_dims(img.flatten(), axis=0)

    return imgSet


def genRewardVect(gardenID, numExemplars):
    rewards = np.array([[]])
    if(gardenID == 'north'):
        # cardinal +
        rewards = np.tile(np.tile(np.asarray([-50, -25, 0, 25, 50]), 5),
                          numExemplars).reshape(1, numExemplars*25)
        # cardinal-
        rewards = np.vstack((rewards, np.tile(np.tile(np.flipud(
            np.asarray([-50, -25, 0, 25, 50])), 5), numExemplars)))

    elif(gardenID == 'south'):
        x = np.asarray([-50, -50, -50, -50, -50])
        rewLabels = np.copy(x)
        for ii in range(1, 5):
            x += 25
            rewLabels = np.concatenate((rewLabels, x), axis=0)
        # cardinal +
        rewards = np.tile(rewLabels, numExemplars)
        x = np.asarray([50, 50, 50, 50, 50])
        rewLabels = np.copy(x)
        for ii in range(1, 5):
            x -= 25
            rewLabels = np.concatenate((rewLabels, x), axis=0)
        # cardinal -
        rewards = np.vstack((rewards, np.tile(rewLabels, numExemplars)))

    return np.transpose(rewards)


def genCategoryVect(rewards):
    categories = np.copy(rewards)
    categories[categories < 0] = -1
    categories[categories > 0] = 1
    return categories



def gen2Dgauss(x_mu=.0, y_mu=.0, xy_sigma=.1, n=20):
    '''
    generates two-dimensional gaussian blob
    '''
    xx, yy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    gausspdf = multivariate_normal(
        [x_mu, y_mu], [[xy_sigma, 0], [0, xy_sigma]])
    x_in = np.empty(xx.shape + (2,))
    x_in[:, :, 0] = xx
    x_in[:, :, 1] = yy
    return gausspdf.pdf(x_in)


def mk_block(garden, do_shuffle):
    """
    generates block of single task for experiment
    with gaussian blobs as inputs

    Input:
      - garden  : 'north' or 'south'
      - do_shuffle: True or False, shuffles  values
    """
    resolution = 5
    n_units = resolution**2
    l, b = np.meshgrid(np.linspace(0.2, .8, 5), np.linspace(0.2, .8, 5))
    b = b.flatten()
    l = l.flatten()
    r_n, r_s = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    r_s = r_s.flatten()
    r_n = r_n.flatten()
    val_l, val_b = np.meshgrid(np.linspace(1, 5, 5), np.linspace(1, 5, 5))
    val_b = val_b.flatten()
    val_l = val_l.flatten()

    ii_sub = 1
    blobs = np.empty((25, n_units))
    for ii in range(0, 25):
        blob = gen2Dgauss(x_mu=b[ii], y_mu=l[ii], xy_sigma=0.08, n=resolution)
        blob = blob / np.max(blob)
        ii_sub += 1
        blobs[ii, :] = blob.flatten()
    x1 = blobs
    reward = r_n if garden == 'north' else r_s

    feature_vals = np.vstack((val_b, val_l)).T
    if do_shuffle:
        ii_shuff = np.random.permutation(25)
        x1 = x1[ii_shuff, :]
        feature_vals = feature_vals[ii_shuff, :]
        reward = reward[ii_shuff]
    return x1, reward, feature_vals


def mk_experiment(whichtask='both'):
    """
    creates a whole training/test dataset of gaussian blobs with
    separate context vectors and rules for task a and b
    """
    # ------------------- Dataset----------------------
    if whichtask == 'both':
        x_north, y_north, _ = mk_block('north', 0)
        y_north = y_north[:, np.newaxis]
        c_north = np.repeat(np.array([[1, 0]]), 25, axis=0)

        x_south, y_south, _ = mk_block('south', 0)
        y_south = y_south[:, np.newaxis]

        c_south = np.repeat(np.array([[0, 1]]), 25, axis=0)

        x_in = np.concatenate((x_north, x_south), axis=0)
        y_rew = np.concatenate((y_north, y_south), axis=0)
        x_ctx = np.concatenate((c_north, c_south), axis=0)

    elif whichtask == 'north':
        x_in, y_rew, _ = mk_block('north', 0)
        y_rew = y_rew[:, np.newaxis]
        x_ctx = np.repeat(np.array([[0, 1]]), 25, axis=0)

    elif whichtask == 'south':
        x_in, y_rew, _ = mk_block('south', 0)
        y_rew = y_rew[:, np.newaxis]
        x_ctx = np.repeat(np.array([[0, 1]]), 25, axis=0)

    # normalise all inputs:
    x_in = x_in/norm(x_in)
    x_ctx = x_ctx/norm(x_ctx)

    # rename inputs
    return x_in.T, x_ctx.T, (-1)*(y_rew).T