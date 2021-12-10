'''
implementation of nnet simulations reported in the manuscript 
'''

# from utils import *
from utils.eval import perform_rsa_cnn
from utils.data import mk_experiment
from datetime import datetime
from models import MLP, MLP_L2
import torch
from torch import nn
import numpy as np

from utils.eval import *


def train_cnn(args, model,data) -> dict:
    """trains a CNN on the trees tasks

    Args:
        args (argparse): network and training parameters
        model (nn.Module): a pytorch neural netowrk

    Returns:
        dict: training and test statistics (loss, acc, patterns etc)
    """

    dl_train, dl_val, dl_tn, dl_ts = data

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    results = {}
    results['train_losses'] = []
    results['valid_losses'] = []
    results['train_accs'] = []
    results['valid_accs'] = []
    # send model to GPU
    model.to(args.device)

    for epoch in range(1, args.n_epochs + 1):

        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0
        task_a_test_acc = 0.0
        task_b_test_acc = 0.0

        model.train()
        for data, target in dl_train:
            data = data.to(args.device)
            target = target.to(args.device)
            optimiser.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimiser.step()
            # normalise loss by batch size
            train_loss += loss.item() * data.size(0)
            # class labels
            idcs = target != 0
            train_acc += torch.sum((output[idcs] > 0) == (target[idcs] > 0))

        model.eval()
        for data, target in dl_val:
            data = data.to(args.device)
            target = target.to(args.device)
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)
                # class labels
                idcs = target != 0
                valid_acc += torch.sum((output[idcs] > 0)
                                       == (target[idcs] > 0))

        # task a
        for data, target in dl_tn:
            data = data.to(args.device)
            target = target.to(args.device)
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)
                # class labels
                idcs = target != 0
                task_a_test_acc += torch.sum((output[idcs] > 0)
                                             == (target[idcs] > 0))

        # task b
        for data, target in dl_ts:
            data = data.to(args.device)
            target = target.to(args.device)
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)
                # class labels
                idcs = target != 0
                task_b_test_acc += torch.sum((output[idcs] > 0)
                                             == (target[idcs] > 0))

        train_loss = train_loss/len(dl_train.sampler)
        valid_loss = valid_loss/len(dl_val.sampler)
        results['train_losses'].append(train_loss)
        results['valid_losses'].append(valid_loss)
        train_acc = train_acc/len(dl_train.sampler)
        valid_acc = valid_acc/len(dl_val.sampler)
        task_a_test_acc = task_a_test_acc/len(dl_tn.sampler)
        task_b_test_acc = task_b_test_acc/len(dl_ts.sampler)
        results['train_accs'].append(train_acc)
        results['valid_accs'].append(valid_acc)

        print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f} \tTraining Accuracy: {:.3f} \tValidation Accuracy: {:.3f} \t Task A {:.3f} \t Task B {:.3f} '.format(
            epoch, train_loss, valid_loss, train_acc, valid_acc, task_a_test_acc, task_b_test_acc))
    
    # perform rsa
    results['outputs'], results['rdms'] = perform_rsa_cnn(args,model,filepath=args.datadir)
    return results


def run_simulation_diffdims(N_STIM, N_CTX, N_HIDDEN, N_OUT, N_RUNS, N_ITER, LRATE, SCALE_WHXS, SCALE_WHXC, SCALE_WYH, N_FACTORS):
    """runs simulations with different initial weight scales 
       and performs dimensionality analysis as well as ablation study on trained network

    Args:
        N_STIM (int): number of features
        N_CTX (int): number of context units
        N_HIDDEN (int): number of hidden units
        N_OUT (int): number of output units
        N_RUNS (int): number of independent training runs
        N_ITER (int): number of iterations per run
        LRATE (float): SGD learning rate
        SCALE_WHXS (float): init weight scale of input-to-hidden units
        SCALE_WHXC (float): init weight scale of context-to-hidden units
        SCALE_WYH (float): init weight scale of hidden-to-output units
        N_FACTORS (int): number of different init weight scales

    Returns:
        dict: results dictionary with logs for training and test phases
    """
    x_stim, x_ctx, y = mk_experiment('both')

    # ------------------ Simulation -------------------

    # init results dict
    results = {
        'all_losses': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'all_accuracies': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_relchange_hxc': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_relchange_hxs': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_relchange_yh': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_delta_hxc': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_delta_hxs': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_delta_yh': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'all_x_hidden': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, x_stim.shape[1])),
        'all_y_hidden': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, x_stim.shape[1])),
        'all_y_out': np.empty((N_FACTORS, N_RUNS, 1, x_stim.shape[1])),
        'all_w_hxs': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_STIM)),
        'all_w_hxc': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_CTX)),
        'all_w_yh': np.empty((N_FACTORS, N_RUNS, N_OUT, N_HIDDEN)),
        'w_hxc_0': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_CTX)),
        'w_hxs_0': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_STIM)),
        'w_yh_0': np.empty((N_FACTORS, N_RUNS, N_OUT, N_HIDDEN)),
        'n_dead': np.empty((N_FACTORS, N_RUNS, 2)),
        'n_local': np.empty((N_FACTORS, N_RUNS, 2)),
        'n_only_a': np.empty((N_FACTORS, N_RUNS, 2)),
        'n_only_b': np.empty((N_FACTORS, N_RUNS, 2)),
        'hidden_dotprod': np.empty((N_FACTORS, N_RUNS, 2)),
        'svd_scree': np.empty((N_FACTORS, N_RUNS, y.shape[1])),
        'svd_betas': np.empty((N_FACTORS, N_RUNS, y.shape[1], 3)),
        'svd_loss': np.empty((N_FACTORS, N_RUNS, y.shape[1])),
        'svd_y': np.empty((N_FACTORS, N_RUNS, y.shape[1], y.shape[1])),
        'svd_acc': np.empty((N_FACTORS, N_RUNS, y.shape[1])),
        'scale_whxs': SCALE_WHXS,
        'scale_whxc': SCALE_WHXC,
        'scale_wyh': SCALE_WYH,
        'n_hidden': N_HIDDEN,
        'lrate': LRATE,
        'corrs': np.empty((3, 2, N_FACTORS, N_RUNS)),
        'acc_y_ref': np.empty((N_FACTORS, N_RUNS, 1)),
        'acc_y_mixed': np.empty((N_FACTORS, N_RUNS, 1)),
        'acc_y_local': np.empty((N_FACTORS, N_RUNS, 1)),
        'acc_ablate_a': np.empty((N_FACTORS, N_RUNS, 1)),
        'acc_ablate_b': np.empty((N_FACTORS, N_RUNS, 1)),
        'acc_ablate_task': np.empty((N_FACTORS, N_RUNS, 1)),
        'acc_ablate_mixed': np.empty((N_FACTORS, N_RUNS, 1)),
        'acc_ablate_none_congr': np.empty((N_FACTORS, N_RUNS, 2)),
        'acc_ablate_a_congr': np.empty((N_FACTORS, N_RUNS, 2)),
        'acc_ablate_b_congr': np.empty((N_FACTORS, N_RUNS, 2)),
        'acc_ablate_task_congr': np.empty((N_FACTORS, N_RUNS, 2)),
        'acc_ablate_mixed_congr': np.empty((N_FACTORS, N_RUNS, 2)),
        'acc_ablate_none_both': np.empty((N_FACTORS, N_RUNS, 2)),
        'acc_ablate_a_both': np.empty((N_FACTORS, N_RUNS, 2)),
        'acc_ablate_b_both': np.empty((N_FACTORS, N_RUNS, 2)),        
        'acc_ablate_mixed_both': np.empty((N_FACTORS, N_RUNS, 2))

    }

    # loop through variance scaling factors
    for ii in range(N_FACTORS):
        print((datetime.now().strftime("%d/%m/%Y %H:%M:%S") +
              ' - scale factor {} / {}').format(ii+1, N_FACTORS))
        for jj in range(N_RUNS):
            print('... run {} / {}'.format(jj+1, N_RUNS))

            # initialise network and get starting values
            mlp = MLP(n_in=N_STIM, n_ctx=N_CTX, n_hidden=N_HIDDEN, n_out=N_OUT, lrate=LRATE,
                      scale_whxc=SCALE_WHXC[ii], scale_whxs=SCALE_WHXS[ii], scale_wyh=SCALE_WYH[ii])
            results['w_hxs_0'][ii, jj, :, :] = mlp.w_hxs
            results['w_hxc_0'][ii, jj, :, :] = mlp.w_hxc
            results['w_yh_0'][ii, jj, :, :] = mlp.w_yh
            # calculate sparsity and dead units at initialisation
            mlp.fprop(x_stim, x_ctx, y)
            n_dead, n_local, n_only_a, n_only_b, h_dotprod = compute_sparsity_stats(
                mlp.h_out)
            results['n_dead'][ii, jj, 0] = n_dead
            results['n_local'][ii, jj, 0] = n_local
            results['n_only_a'][ii, jj, 0] = n_only_a
            results['n_only_b'][ii, jj, 0] = n_only_b
            results['hidden_dotprod'][ii, jj, 0] = h_dotprod

            w_hxs_n1 = mlp.w_hxs
            w_hxc_n1 = mlp.w_hxc
            w_yh_n1 = mlp.w_yh

            # train network
            for kk in range(N_ITER):
                mlp.train(x_stim, x_ctx, y)

                # log data
                results['all_losses'][ii, jj, kk] = mlp.l
                results['all_accuracies'][ii, jj,
                                          kk] = compute_accuracy(mlp.y_, y)
                results['w_relchange_hxs'][ii, jj, kk] = compute_relchange(
                    results['w_hxs_0'][ii, jj, :, :], mlp.w_hxs)
                results['w_relchange_hxc'][ii, jj, kk] = compute_relchange(
                    results['w_hxc_0'][ii, jj, :, :], mlp.w_hxc)
                results['w_relchange_yh'][ii, jj, kk] = compute_relchange(
                    results['w_yh_0'][ii, jj, :, :], mlp.w_yh)

                results['w_delta_hxs'][ii, jj, kk] = compute_relchange(
                    w_hxs_n1, mlp.w_hxs)
                w_hxs_n1 = mlp.w_hxs
                results['w_delta_hxc'][ii, jj, kk] = compute_relchange(
                    w_hxc_n1, mlp.w_hxc)
                w_hxc_n1 = mlp.w_hxc
                results['w_delta_yh'][ii, jj, kk] = compute_relchange(
                    w_yh_n1, mlp.w_yh)
                w_yh_n1 = mlp.w_yh

            results['all_x_hidden'][ii, jj, :, :] = mlp.h_in
            results['all_y_hidden'][ii, jj, :, :] = mlp.h_out
            results['all_y_out'][ii, jj, :, :] = mlp.y_
            results['all_w_hxs'][ii, jj, :, :] = mlp.w_hxs
            results['all_w_hxc'][ii, jj, :, :] = mlp.w_hxc
            results['all_w_yh'][ii, jj, :, :] = mlp.w_yh
            # calculate endpoint sparsity and n_dead
            n_dead, n_local, n_only_a, n_only_b, h_dotprod = compute_sparsity_stats(
                mlp.h_out)
            results['n_dead'][ii, jj, 1] = n_dead
            results['n_local'][ii, jj, 1] = n_local
            results['n_only_a'][ii, jj, 1] = n_only_a
            results['n_only_b'][ii, jj, 1] = n_only_b
            results['hidden_dotprod'][ii, jj, 1] = h_dotprod
            # --- dimensionality ----
            # calculate endpoint Scree plot
            _, results['svd_scree'][ii, jj, :] = compute_svd_EVs(mlp.h_out)
            # calculate endpoint model correlations
            results['svd_betas'][ii, jj, :, :] = compute_svd_betas(mlp.h_out)
            # calculate endpoint model accuracy
            results['svd_loss'][ii, jj, :], results['svd_y'][ii, jj, :,:], results['svd_acc'][ii, jj, :] = compute_svd_acc(mlp, y)
            # ----- ctx weight correlations ----------
            yout = results['all_y_hidden'][ii, jj, :, :]
            x = np.vstack(
                (np.nanmean(yout[:, 0:25], 1).T, np.nanmean(yout[:, 25:], 1).T))
            # local units
            mask_local = ~(np.all(x, axis=0)) & np.any(x, axis=0)
            mask_a = np.all(np.vstack((x[0, :] > 0, x[1, :] == 0)), axis=0)
            mask_b = np.all(np.vstack((x[0, :] == 0, x[1, :] > 0)), axis=0)
            results['corrs'][0, 0, ii, jj] = np.corrcoef(
                results['w_hxc_0'][ii, jj, mask_local == 1, :].T)[0, 1]
            results['corrs'][0, 1, ii, jj] = np.corrcoef(
                results['all_w_hxc'][ii, jj, mask_local == 1, :].T)[0, 1]
            results['corrs'][1, 0, ii, jj] = np.corrcoef(
                results['w_hxc_0'][ii, jj, ~mask_local == 1, :].T)[0, 1]
            results['corrs'][1, 1, ii, jj] = np.corrcoef(
                results['all_w_hxc'][ii, jj, ~mask_local == 1, :].T)[0, 1]
            results['corrs'][2, 0, ii, jj] = np.corrcoef(
                results['w_hxc_0'][ii, jj, :, :].T)[0, 1]
            results['corrs'][2, 1, ii, jj] = np.corrcoef(
                results['all_w_hxc'][ii, jj, :, :].T)[0, 1]

            # ----- ablation study --------
            yout = results['all_y_hidden'][0, jj, :, :]
            x = np.vstack(
                (np.nanmean(yout[:, 0:25], 1).T, np.nanmean(yout[:, 25:], 1).T))
            mask_local = (~(np.all(x, axis=0)) & np.any(x, axis=0))
            mask_mixed = np.all(x, axis=0)

            # RETAINING only the units below:
            results['acc_y_ref'][ii, jj, :] = compute_accuracy(
                mlp.w_yh.dot(mlp.h_out)+mlp.b_yh, y)
            results['acc_y_mixed'][ii, jj, :] = compute_accuracy(
                mlp.w_yh.dot(mlp.h_out*mask_mixed[:, np.newaxis])+mlp.b_yh, y)
            results['acc_y_local'][ii, jj, :] = compute_accuracy(
                mlp.w_yh.dot(mlp.h_out*mask_local[:, np.newaxis])+mlp.b_yh, y)
            
            #REMOVING only the units below:             
            mask_only_A = np.all(np.vstack((x[0,:]>0,x[1,:]==0)),axis=0)            
            mask_only_B = np.all(np.vstack((x[0,:]==0,x[1,:]>0)),axis=0)
            ablate_a = ~mask_only_A
            ablate_b = ~mask_only_B
            ablate_mixed = ~mask_mixed
            ablate_task = ~mask_local

            results['acc_ablate_a'][ii, jj, :] = compute_accuracy(
                mlp.w_yh.dot(mlp.h_out*ablate_a[:, np.newaxis])+mlp.b_yh, y)
            results['acc_ablate_b'][ii, jj, :] = compute_accuracy(
                mlp.w_yh.dot(mlp.h_out*ablate_b[:, np.newaxis])+mlp.b_yh, y)
            results['acc_ablate_task'][ii, jj, :] = compute_accuracy(
                mlp.w_yh.dot(mlp.h_out*ablate_task[:, np.newaxis])+mlp.b_yh, y)
            results['acc_ablate_mixed'][ii, jj, :] = compute_accuracy(
                mlp.w_yh.dot(mlp.h_out*ablate_mixed[:, np.newaxis])+mlp.b_yh, y)
            

            #same as above but for ta and tb separately   
            ys = np.squeeze(y)
            ya = ys[:25].reshape(5,5)
            yb = ys[25:].reshape(5,5)
            
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out)+mlp.b_yh)            
            results['acc_ablate_none_both'][ii,jj,:] = [compute_accuracy(t[:25],ya),compute_accuracy(t[25:],yb)]
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out*ablate_a[:, np.newaxis])+mlp.b_yh)
            results['acc_ablate_a_both'][ii,jj,:] = [compute_accuracy(t[:25],ya),compute_accuracy(t[25:],yb)]
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out*ablate_b[:, np.newaxis])+mlp.b_yh)
            results['acc_ablate_b_both'][ii,jj,:] = [compute_accuracy(t[:25],ya),compute_accuracy(t[25:],yb)]         
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out*ablate_mixed[:, np.newaxis])+mlp.b_yh)
            results['acc_ablate_mixed_both'][ii,jj,:] = [compute_accuracy(t[:25],ya),compute_accuracy(t[25:],yb)]         

           
            # congruent vs incongruent stimuli:
            ys = np.squeeze(y)
            ya = ys[:25].reshape(5,5)
            yb = ys[25:].reshape(5,5)
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out)+mlp.b_yh)
            ta = t[:25].reshape(5,5)
            tb = t[25:].reshape(5,5)
            aca1,aca2 = compute_congruency_acc(ta,ya)        
            acb1,acb2 = compute_congruency_acc(tb,yb)
            results['acc_ablate_none_congr'][ii,jj,0] = (aca1+acb1)/2
            results['acc_ablate_none_congr'][ii,jj,1] = (aca2+acb2)/2
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out*ablate_a[:, np.newaxis])+mlp.b_yh)
            ta = t[:25].reshape(5,5)
            tb = t[25:].reshape(5,5)
            aca1,aca2 = compute_congruency_acc(ta,ya)        
            acb1,acb2 = compute_congruency_acc(tb,yb)
            results['acc_ablate_a_congr'][ii,jj,0] = (aca1+acb1)/2
            results['acc_ablate_a_congr'][ii,jj,1] = (aca2+acb2)/2
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out*ablate_b[:, np.newaxis])+mlp.b_yh)
            ta = t[:25].reshape(5,5)
            tb = t[25:].reshape(5,5)
            aca1,aca2 = compute_congruency_acc(ta,ya)        
            acb1,acb2 = compute_congruency_acc(tb,yb)
            results['acc_ablate_b_congr'][ii,jj,0] = (aca1+acb1)/2
            results['acc_ablate_b_congr'][ii,jj,1] = (aca2+acb2)/2
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out*ablate_task[:, np.newaxis])+mlp.b_yh)
            ta = t[:25].reshape(5,5)
            tb = t[25:].reshape(5,5)
            aca1,aca2 = compute_congruency_acc(ta,ya)        
            acb1,acb2 = compute_congruency_acc(tb,yb)
            results['acc_ablate_task_congr'][ii,jj,0] = (aca1+acb1)/2
            results['acc_ablate_task_congr'][ii,jj,1] = (aca2+acb2)/2
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out*ablate_mixed[:, np.newaxis])+mlp.b_yh)
            ta = t[:25].reshape(5,5)
            tb = t[25:].reshape(5,5)
            aca1,aca2 = compute_congruency_acc(ta,ya)        
            acb1,acb2 = compute_congruency_acc(tb,yb)
            results['acc_ablate_mixed_congr'][ii,jj,0] = (aca1+acb1)/2
            results['acc_ablate_mixed_congr'][ii,jj,1] = (aca2+acb2)/2
            
            

    return results


def run_simulation_norm(N_STIM, N_CTX, N_HIDDEN, N_OUT, N_RUNS, N_ITER, LRATE, SCALE_WHXS, SCALE_WHXC, SCALE_WYH, N_FACTORS, LAMBDA):
    """same as above, but with L2 regulariser

    Args:        
        LAMBDA (float): regularisation strength of L2 penalty

    """
    x_stim, x_ctx, y = mk_experiment('both')

    # ------------------ Simulation -------------------

    # init results dict

    results = {
        'all_losses': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_relchange_hxc': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_relchange_hxs': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_relchange_yh': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'all_x_hidden': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, x_stim.shape[1])),
        'all_y_hidden': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, x_stim.shape[1])),
        'all_y_out': np.empty((N_FACTORS, N_RUNS, 1, x_stim.shape[1])),
        'all_w_hxs': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_STIM)),
        'all_w_hxc': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_CTX)),
        'all_w_yh': np.empty((N_FACTORS, N_RUNS, N_OUT, N_HIDDEN)),
        'w_hxc_0': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_CTX)),
        'w_hxs_0': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_STIM)),
        'w_yh_0': np.empty((N_FACTORS, N_RUNS, N_OUT, N_HIDDEN)),
        'n_dead': np.empty((N_FACTORS, N_RUNS, 2)),
        'n_local': np.empty((N_FACTORS, N_RUNS, 2)),
        'n_only_a': np.empty((N_FACTORS, N_RUNS, 2)),
        'n_only_b': np.empty((N_FACTORS, N_RUNS, 2)),
        'hidden_dotprod': np.empty((N_FACTORS, N_RUNS, 2)),
        'scale_whxs': SCALE_WHXS,
        'scale_whxc': SCALE_WHXC,
        'scale_wyh': SCALE_WYH,
        'n_hidden': N_HIDDEN,
        'lrate': LRATE

    }

    # loop through variance scaling factors
    for ii in range(N_FACTORS):
        print((datetime.now().strftime("%d/%m/%Y %H:%M:%S") +
              ' - scale factor {} / {}').format(ii+1, N_FACTORS))
        for jj in range(N_RUNS):
            print('... run {} / {}'.format(jj+1, N_RUNS))

            # initialise network and get starting values
            mlp = MLP_L2(n_in=N_STIM, n_ctx=N_CTX, n_hidden=N_HIDDEN, n_out=N_OUT, lrate=LRATE,
                         scale_whxc=SCALE_WHXC[ii], scale_whxs=SCALE_WHXS[ii], scale_wyh=SCALE_WYH[ii], lmbd=LAMBDA[ii])
            results['w_hxs_0'][ii, jj, :, :] = mlp.w_hxs
            results['w_hxc_0'][ii, jj, :, :] = mlp.w_hxc
            results['w_yh_0'][ii, jj, :, :] = mlp.w_yh
            # calculate sparsity and dead units at initialisation
            mlp.fprop(x_stim, x_ctx, y)
            n_dead, n_local, n_only_a, n_only_b, h_dotprod = compute_sparsity_stats(
                mlp.h_out)
            results['n_dead'][ii, jj, 0] = n_dead
            results['n_local'][ii, jj, 0] = n_local
            results['n_only_a'][ii, jj, 0] = n_only_a
            results['n_only_b'][ii, jj, 0] = n_only_b
            results['hidden_dotprod'][ii, jj, 0] = h_dotprod

            # train network
            for kk in range(N_ITER):
                mlp.train(x_stim, x_ctx, y)

                # log data
                results['all_losses'][ii, jj, kk] = mlp.l
                results['w_relchange_hxs'][ii, jj, kk] = compute_relchange(
                    results['w_hxs_0'][ii, jj, :, :], mlp.w_hxs)
                results['w_relchange_hxc'][ii, jj, kk] = compute_relchange(
                    results['w_hxc_0'][ii, jj, :, :], mlp.w_hxc)
                results['w_relchange_yh'][ii, jj, kk] = compute_relchange(
                    results['w_yh_0'][ii, jj, :, :], mlp.w_yh)

            results['all_x_hidden'][ii, jj, :, :] = mlp.h_in
            results['all_y_hidden'][ii, jj, :, :] = mlp.h_out
            results['all_y_out'][ii, jj, :, :] = mlp.y_
            results['all_w_hxs'][ii, jj, :, :] = mlp.w_hxs
            results['all_w_hxc'][ii, jj, :, :] = mlp.w_hxc
            results['all_w_yh'][ii, jj, :, :] = mlp.w_yh
            # calculate endpoint sparsity and n_dead
            n_dead, n_local, n_only_a, n_only_b, h_dotprod = compute_sparsity_stats(
                mlp.h_out)
            results['n_dead'][ii, jj, 1] = n_dead
            results['n_local'][ii, jj, 1] = n_local
            results['n_only_a'][ii, jj, 1] = n_only_a
            results['n_only_b'][ii, jj, 1] = n_only_b
            results['hidden_dotprod'][ii, jj, 1] = h_dotprod

    return results


def run_simulation_noiselevel(N_STIM, N_CTX, N_HIDDEN, N_OUT, N_RUNS, N_ITER, LRATE, SCALE_WHXS, SCALE_WHXC, SCALE_WYH, SCALE_NOISE, N_FACTORS):
    """same as above, but with different levels of input noise during test phase

    Args:        
        SCALE_NOISE (float): strenght of noise pertubations
        N_FACTORS (int): number of noise levels

    """
    x_stim, x_ctx, y = mk_experiment('both')

    # ------------------ Simulation -------------------
    # init results dict

    results = {
        'all_losses': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_relchange_hxc': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_relchange_hxs': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_relchange_yh': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'all_x_hidden': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, x_stim.shape[1])),
        'all_y_hidden': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, x_stim.shape[1])),
        'all_y_out': np.empty((N_FACTORS, N_RUNS, 1, x_stim.shape[1])),
        'all_w_hxs': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_STIM)),
        'all_w_hxc': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_CTX)),
        'all_w_yh': np.empty((N_FACTORS, N_RUNS, N_OUT, N_HIDDEN)),
        'w_hxc_0': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_CTX)),
        'w_hxs_0': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_STIM)),
        'w_yh_0': np.empty((N_FACTORS, N_RUNS, N_OUT, N_HIDDEN)),
        'n_dead': np.empty((N_FACTORS, N_RUNS, 2)),
        'n_local': np.empty((N_FACTORS, N_RUNS, 2)),
        'n_only_a': np.empty((N_FACTORS, N_RUNS, 2)),
        'n_only_b': np.empty((N_FACTORS, N_RUNS, 2)),
        'hidden_dotprod': np.empty((N_FACTORS, N_RUNS, 2)),
        'scale_whxs': SCALE_WHXS,
        'scale_whxc': SCALE_WHXC,
        'scale_wyh': SCALE_WYH,
        'n_hidden': N_HIDDEN,
        'lrate': LRATE,
        'acc_noise': np.empty((N_FACTORS, N_RUNS, len(SCALE_NOISE), 1)),
        'loss_noise': np.empty((N_FACTORS, N_RUNS, len(SCALE_NOISE), 1))

    }

    # loop through variance scaling factors
    for ii in range(N_FACTORS):
        print((datetime.now().strftime("%d/%m/%Y %H:%M:%S") +
              ' - scale factor {} / {}').format(ii+1, N_FACTORS))
        for jj in range(N_RUNS):
            print('... run {} / {}'.format(jj+1, N_RUNS))

            # initialise network and get starting values
            mlp = MLP(n_in=N_STIM, n_ctx=N_CTX, n_hidden=N_HIDDEN, n_out=N_OUT, lrate=LRATE,
                      scale_whxc=SCALE_WHXC[ii], scale_whxs=SCALE_WHXS[ii], scale_wyh=SCALE_WYH[ii])
            results['w_hxs_0'][ii, jj, :, :] = mlp.w_hxs
            results['w_hxc_0'][ii, jj, :, :] = mlp.w_hxc
            results['w_yh_0'][ii, jj, :, :] = mlp.w_yh
            # calculate sparsity and dead units at initialisation
            mlp.fprop(x_stim, x_ctx, y)
            n_dead, n_local, n_only_a, n_only_b, h_dotprod = compute_sparsity_stats(
                mlp.h_out)
            results['n_dead'][ii, jj, 0] = n_dead
            results['n_local'][ii, jj, 0] = n_local
            results['n_only_a'][ii, jj, 0] = n_only_a
            results['n_only_b'][ii, jj, 0] = n_only_b
            results['hidden_dotprod'][ii, jj, 0] = h_dotprod

            # train network
            for kk in range(N_ITER):
                mlp.train(x_stim, x_ctx, y)

                # log data
                results['all_losses'][ii, jj, kk] = mlp.l
                results['w_relchange_hxs'][ii, jj, kk] = compute_relchange(
                    results['w_hxs_0'][ii, jj, :, :], mlp.w_hxs)
                results['w_relchange_hxc'][ii, jj, kk] = compute_relchange(
                    results['w_hxc_0'][ii, jj, :, :], mlp.w_hxc)
                results['w_relchange_yh'][ii, jj, kk] = compute_relchange(
                    results['w_yh_0'][ii, jj, :, :], mlp.w_yh)

            results['all_x_hidden'][ii, jj, :, :] = mlp.h_in
            results['all_y_hidden'][ii, jj, :, :] = mlp.h_out
            results['all_y_out'][ii, jj, :, :] = mlp.y_
            results['all_w_hxs'][ii, jj, :, :] = mlp.w_hxs
            results['all_w_hxc'][ii, jj, :, :] = mlp.w_hxc
            results['all_w_yh'][ii, jj, :, :] = mlp.w_yh
            # calculate endpoint sparsity and n_dead
            n_dead, n_local, n_only_a, n_only_b, h_dotprod = compute_sparsity_stats(
                mlp.h_out)
            results['n_dead'][ii, jj, 1] = n_dead
            results['n_local'][ii, jj, 1] = n_local
            results['n_only_a'][ii, jj, 1] = n_only_a
            results['n_only_b'][ii, jj, 1] = n_only_b
            results['hidden_dotprod'][ii, jj, 1] = h_dotprod

            # test with different noise levels
            for kk, nlvl in enumerate(SCALE_NOISE):
                noisemat_stim = nlvl * \
                    np.random.randn(x_stim.shape[0], x_stim.shape[1])
                noisemat_ctx = nlvl * \
                    np.random.randn(x_ctx.shape[0], x_ctx.shape[1])
                mlp.fprop(noisemat_stim+x_stim, noisemat_ctx+x_ctx, y)
                y_ = mlp.y_
                idcs = y != 0
                results['acc_noise'][ii, jj, kk, :] = np.mean(
                    (y_[idcs] > 0.5) == (y[idcs] > 0))

                results['loss_noise'][ii, jj, kk, :] = mlp.loss(y_, y)

    return results




def run_simulation_difflrs(N_STIM, N_CTX, N_HIDDEN, N_OUT, N_RUNS, N_ITER, LRATE, SCALE_WHXS, SCALE_WHXC, SCALE_WYH, N_FACTORS):
    """runs simulations with different initial weight scales 
       and performs dimensionality analysis as well as ablation study on trained network

    Args:
        N_STIM (int): number of features
        N_CTX (int): number of context units
        N_HIDDEN (int): number of hidden units
        N_OUT (int): number of output units
        N_RUNS (int): number of independent training runs
        N_ITER (int): number of iterations per run
        LRATE (list): SGD learning rate
        SCALE_WHXS (float): init weight scale of input-to-hidden units
        SCALE_WHXC (float): init weight scale of context-to-hidden units
        SCALE_WYH (float): init weight scale of hidden-to-output units
        N_FACTORS (int): number of different init weight scales

    Returns:
        dict: results dictionary with logs for training and test phases
    """
    x_stim, x_ctx, y = mk_experiment('both')

    # ------------------ Simulation -------------------

    # init results dict
    results = {
        'all_losses': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'all_accuracies': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_relchange_hxc': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_relchange_hxs': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_relchange_yh': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_delta_hxc': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_delta_hxs': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'w_delta_yh': np.empty((N_FACTORS, N_RUNS, N_ITER)),
        'all_x_hidden': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, x_stim.shape[1])),
        'all_y_hidden': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, x_stim.shape[1])),
        'all_y_out': np.empty((N_FACTORS, N_RUNS, 1, x_stim.shape[1])),
        'all_w_hxs': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_STIM)),
        'all_w_hxc': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_CTX)),
        'all_w_yh': np.empty((N_FACTORS, N_RUNS, N_OUT, N_HIDDEN)),
        'w_hxc_0': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_CTX)),
        'w_hxs_0': np.empty((N_FACTORS, N_RUNS, N_HIDDEN, N_STIM)),
        'w_yh_0': np.empty((N_FACTORS, N_RUNS, N_OUT, N_HIDDEN)),
        'n_dead': np.empty((N_FACTORS, N_RUNS, 2)),
        'n_local': np.empty((N_FACTORS, N_RUNS, 2)),
        'n_only_a': np.empty((N_FACTORS, N_RUNS, 2)),
        'n_only_b': np.empty((N_FACTORS, N_RUNS, 2)),
        'hidden_dotprod': np.empty((N_FACTORS, N_RUNS, 2)),
        'svd_scree': np.empty((N_FACTORS, N_RUNS, y.shape[1])),
        'svd_betas': np.empty((N_FACTORS, N_RUNS, y.shape[1], 3)),
        'svd_loss': np.empty((N_FACTORS, N_RUNS, y.shape[1])),
        'svd_y': np.empty((N_FACTORS, N_RUNS, y.shape[1], y.shape[1])),
        'svd_acc': np.empty((N_FACTORS, N_RUNS, y.shape[1])),
        'scale_whxs': SCALE_WHXS,
        'scale_whxc': SCALE_WHXC,
        'scale_wyh': SCALE_WYH,
        'n_hidden': N_HIDDEN,
        'lrate': LRATE,
        'corrs': np.empty((3, 2, N_FACTORS, N_RUNS)),
        'acc_y_ref': np.empty((N_FACTORS, N_RUNS, 1)),
        'acc_y_mixed': np.empty((N_FACTORS, N_RUNS, 1)),
        'acc_y_local': np.empty((N_FACTORS, N_RUNS, 1)),
        'acc_ablate_a': np.empty((N_FACTORS, N_RUNS, 1)),
        'acc_ablate_b': np.empty((N_FACTORS, N_RUNS, 1)),
        'acc_ablate_task': np.empty((N_FACTORS, N_RUNS, 1)),
        'acc_ablate_mixed': np.empty((N_FACTORS, N_RUNS, 1)),
        'acc_ablate_none_congr': np.empty((N_FACTORS, N_RUNS, 2)),
        'acc_ablate_a_congr': np.empty((N_FACTORS, N_RUNS, 2)),
        'acc_ablate_b_congr': np.empty((N_FACTORS, N_RUNS, 2)),
        'acc_ablate_task_congr': np.empty((N_FACTORS, N_RUNS, 2)),
        'acc_ablate_mixed_congr': np.empty((N_FACTORS, N_RUNS, 2)),
        'acc_ablate_none_both': np.empty((N_FACTORS, N_RUNS, 2)),
        'acc_ablate_a_both': np.empty((N_FACTORS, N_RUNS, 2)),
        'acc_ablate_b_both': np.empty((N_FACTORS, N_RUNS, 2)),        
        'acc_ablate_mixed_both': np.empty((N_FACTORS, N_RUNS, 2))

    }

    # loop through variance scaling factors
    for ii in range(N_FACTORS):
        print((datetime.now().strftime("%d/%m/%Y %H:%M:%S") +
              ' - scale factor {} / {}').format(ii+1, N_FACTORS))
        for jj in range(N_RUNS):
            print('... run {} / {}'.format(jj+1, N_RUNS))

            # initialise network and get starting values
            mlp = MLP(n_in=N_STIM, n_ctx=N_CTX, n_hidden=N_HIDDEN, n_out=N_OUT, lrate=LRATE[ii],
                      scale_whxc=SCALE_WHXC, scale_whxs=SCALE_WHXS, scale_wyh=SCALE_WYH)
            results['w_hxs_0'][ii, jj, :, :] = mlp.w_hxs
            results['w_hxc_0'][ii, jj, :, :] = mlp.w_hxc
            results['w_yh_0'][ii, jj, :, :] = mlp.w_yh
            # calculate sparsity and dead units at initialisation
            mlp.fprop(x_stim, x_ctx, y)
            n_dead, n_local, n_only_a, n_only_b, h_dotprod = compute_sparsity_stats(
                mlp.h_out)
            results['n_dead'][ii, jj, 0] = n_dead
            results['n_local'][ii, jj, 0] = n_local
            results['n_only_a'][ii, jj, 0] = n_only_a
            results['n_only_b'][ii, jj, 0] = n_only_b
            results['hidden_dotprod'][ii, jj, 0] = h_dotprod

            w_hxs_n1 = mlp.w_hxs
            w_hxc_n1 = mlp.w_hxc
            w_yh_n1 = mlp.w_yh

            # train network
            for kk in range(N_ITER):
                mlp.train(x_stim, x_ctx, y)

                # log data
                results['all_losses'][ii, jj, kk] = mlp.l
                results['all_accuracies'][ii, jj,
                                          kk] = compute_accuracy(mlp.y_, y)
                results['w_relchange_hxs'][ii, jj, kk] = compute_relchange(
                    results['w_hxs_0'][ii, jj, :, :], mlp.w_hxs)
                results['w_relchange_hxc'][ii, jj, kk] = compute_relchange(
                    results['w_hxc_0'][ii, jj, :, :], mlp.w_hxc)
                results['w_relchange_yh'][ii, jj, kk] = compute_relchange(
                    results['w_yh_0'][ii, jj, :, :], mlp.w_yh)

                results['w_delta_hxs'][ii, jj, kk] = compute_relchange(
                    w_hxs_n1, mlp.w_hxs)
                w_hxs_n1 = mlp.w_hxs
                results['w_delta_hxc'][ii, jj, kk] = compute_relchange(
                    w_hxc_n1, mlp.w_hxc)
                w_hxc_n1 = mlp.w_hxc
                results['w_delta_yh'][ii, jj, kk] = compute_relchange(
                    w_yh_n1, mlp.w_yh)
                w_yh_n1 = mlp.w_yh

            results['all_x_hidden'][ii, jj, :, :] = mlp.h_in
            results['all_y_hidden'][ii, jj, :, :] = mlp.h_out
            results['all_y_out'][ii, jj, :, :] = mlp.y_
            results['all_w_hxs'][ii, jj, :, :] = mlp.w_hxs
            results['all_w_hxc'][ii, jj, :, :] = mlp.w_hxc
            results['all_w_yh'][ii, jj, :, :] = mlp.w_yh
            # calculate endpoint sparsity and n_dead
            n_dead, n_local, n_only_a, n_only_b, h_dotprod = compute_sparsity_stats(
                mlp.h_out)
            results['n_dead'][ii, jj, 1] = n_dead
            results['n_local'][ii, jj, 1] = n_local
            results['n_only_a'][ii, jj, 1] = n_only_a
            results['n_only_b'][ii, jj, 1] = n_only_b
            results['hidden_dotprod'][ii, jj, 1] = h_dotprod
            # --- dimensionality ----
            # calculate endpoint Scree plot
            _, results['svd_scree'][ii, jj, :] = compute_svd_EVs(mlp.h_out)
            # calculate endpoint model correlations
            results['svd_betas'][ii, jj, :, :] = compute_svd_betas(mlp.h_out)
            # calculate endpoint model accuracy
            results['svd_loss'][ii, jj, :], results['svd_y'][ii, jj, :,:], results['svd_acc'][ii, jj, :] = compute_svd_acc(mlp, y)
            # ----- ctx weight correlations ----------
            yout = results['all_y_hidden'][ii, jj, :, :]
            x = np.vstack(
                (np.nanmean(yout[:, 0:25], 1).T, np.nanmean(yout[:, 25:], 1).T))
            # local units
            mask_local = ~(np.all(x, axis=0)) & np.any(x, axis=0)
            mask_a = np.all(np.vstack((x[0, :] > 0, x[1, :] == 0)), axis=0)
            mask_b = np.all(np.vstack((x[0, :] == 0, x[1, :] > 0)), axis=0)
            results['corrs'][0, 0, ii, jj] = np.corrcoef(
                results['w_hxc_0'][ii, jj, mask_local == 1, :].T)[0, 1]
            results['corrs'][0, 1, ii, jj] = np.corrcoef(
                results['all_w_hxc'][ii, jj, mask_local == 1, :].T)[0, 1]
            results['corrs'][1, 0, ii, jj] = np.corrcoef(
                results['w_hxc_0'][ii, jj, ~mask_local == 1, :].T)[0, 1]
            results['corrs'][1, 1, ii, jj] = np.corrcoef(
                results['all_w_hxc'][ii, jj, ~mask_local == 1, :].T)[0, 1]
            results['corrs'][2, 0, ii, jj] = np.corrcoef(
                results['w_hxc_0'][ii, jj, :, :].T)[0, 1]
            results['corrs'][2, 1, ii, jj] = np.corrcoef(
                results['all_w_hxc'][ii, jj, :, :].T)[0, 1]

            # ----- ablation study --------
            yout = results['all_y_hidden'][0, jj, :, :]
            x = np.vstack(
                (np.nanmean(yout[:, 0:25], 1).T, np.nanmean(yout[:, 25:], 1).T))
            mask_local = (~(np.all(x, axis=0)) & np.any(x, axis=0))
            mask_mixed = np.all(x, axis=0)

            # RETAINING only the units below:
            results['acc_y_ref'][ii, jj, :] = compute_accuracy(
                mlp.w_yh.dot(mlp.h_out)+mlp.b_yh, y)
            results['acc_y_mixed'][ii, jj, :] = compute_accuracy(
                mlp.w_yh.dot(mlp.h_out*mask_mixed[:, np.newaxis])+mlp.b_yh, y)
            results['acc_y_local'][ii, jj, :] = compute_accuracy(
                mlp.w_yh.dot(mlp.h_out*mask_local[:, np.newaxis])+mlp.b_yh, y)
            
            #REMOVING only the units below:             
            mask_only_A = np.all(np.vstack((x[0,:]>0,x[1,:]==0)),axis=0)            
            mask_only_B = np.all(np.vstack((x[0,:]==0,x[1,:]>0)),axis=0)
            ablate_a = ~mask_only_A
            ablate_b = ~mask_only_B
            ablate_mixed = ~mask_mixed
            ablate_task = ~mask_local

            results['acc_ablate_a'][ii, jj, :] = compute_accuracy(
                mlp.w_yh.dot(mlp.h_out*ablate_a[:, np.newaxis])+mlp.b_yh, y)
            results['acc_ablate_b'][ii, jj, :] = compute_accuracy(
                mlp.w_yh.dot(mlp.h_out*ablate_b[:, np.newaxis])+mlp.b_yh, y)
            results['acc_ablate_task'][ii, jj, :] = compute_accuracy(
                mlp.w_yh.dot(mlp.h_out*ablate_task[:, np.newaxis])+mlp.b_yh, y)
            results['acc_ablate_mixed'][ii, jj, :] = compute_accuracy(
                mlp.w_yh.dot(mlp.h_out*ablate_mixed[:, np.newaxis])+mlp.b_yh, y)
            

            #same as above but for ta and tb separately   
            ys = np.squeeze(y)
            ya = ys[:25].reshape(5,5)
            yb = ys[25:].reshape(5,5)
            
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out)+mlp.b_yh)            
            results['acc_ablate_none_both'][ii,jj,:] = [compute_accuracy(t[:25],ya),compute_accuracy(t[25:],yb)]
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out*ablate_a[:, np.newaxis])+mlp.b_yh)
            results['acc_ablate_a_both'][ii,jj,:] = [compute_accuracy(t[:25],ya),compute_accuracy(t[25:],yb)]
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out*ablate_b[:, np.newaxis])+mlp.b_yh)
            results['acc_ablate_b_both'][ii,jj,:] = [compute_accuracy(t[:25],ya),compute_accuracy(t[25:],yb)]         
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out*ablate_mixed[:, np.newaxis])+mlp.b_yh)
            results['acc_ablate_mixed_both'][ii,jj,:] = [compute_accuracy(t[:25],ya),compute_accuracy(t[25:],yb)]         

           
            # congruent vs incongruent stimuli:
            ys = np.squeeze(y)
            ya = ys[:25].reshape(5,5)
            yb = ys[25:].reshape(5,5)
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out)+mlp.b_yh)
            ta = t[:25].reshape(5,5)
            tb = t[25:].reshape(5,5)
            aca1,aca2 = compute_congruency_acc(ta,ya)        
            acb1,acb2 = compute_congruency_acc(tb,yb)
            results['acc_ablate_none_congr'][ii,jj,0] = (aca1+acb1)/2
            results['acc_ablate_none_congr'][ii,jj,1] = (aca2+acb2)/2
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out*ablate_a[:, np.newaxis])+mlp.b_yh)
            ta = t[:25].reshape(5,5)
            tb = t[25:].reshape(5,5)
            aca1,aca2 = compute_congruency_acc(ta,ya)        
            acb1,acb2 = compute_congruency_acc(tb,yb)
            results['acc_ablate_a_congr'][ii,jj,0] = (aca1+acb1)/2
            results['acc_ablate_a_congr'][ii,jj,1] = (aca2+acb2)/2
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out*ablate_b[:, np.newaxis])+mlp.b_yh)
            ta = t[:25].reshape(5,5)
            tb = t[25:].reshape(5,5)
            aca1,aca2 = compute_congruency_acc(ta,ya)        
            acb1,acb2 = compute_congruency_acc(tb,yb)
            results['acc_ablate_b_congr'][ii,jj,0] = (aca1+acb1)/2
            results['acc_ablate_b_congr'][ii,jj,1] = (aca2+acb2)/2
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out*ablate_task[:, np.newaxis])+mlp.b_yh)
            ta = t[:25].reshape(5,5)
            tb = t[25:].reshape(5,5)
            aca1,aca2 = compute_congruency_acc(ta,ya)        
            acb1,acb2 = compute_congruency_acc(tb,yb)
            results['acc_ablate_task_congr'][ii,jj,0] = (aca1+acb1)/2
            results['acc_ablate_task_congr'][ii,jj,1] = (aca2+acb2)/2
            t = np.squeeze(mlp.w_yh.dot(mlp.h_out*ablate_mixed[:, np.newaxis])+mlp.b_yh)
            ta = t[:25].reshape(5,5)
            tb = t[25:].reshape(5,5)
            aca1,aca2 = compute_congruency_acc(ta,ya)        
            acb1,acb2 = compute_congruency_acc(tb,yb)
            results['acc_ablate_mixed_congr'][ii,jj,0] = (aca1+acb1)/2
            results['acc_ablate_mixed_congr'][ii,jj,1] = (aca2+acb2)/2
            
            

    return results