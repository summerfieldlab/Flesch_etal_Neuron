import pickle
import tensorflow as tf 
import numpy as np
from datetime import datetime
from rdmloss_utils import *


def main():
    x_north,y_north,f_north = mk_block('north',0)
    y_north = y_north[:,np.newaxis]
    l_north = (y_north>0).astype('int')
    c_north = np.repeat(np.array([[1,0]]),25,axis=0)

    x_south,y_south,f_south = mk_block('south',0)
    y_south = y_south[:,np.newaxis]
    l_south = (y_south>0).astype('int')
    c_south = np.repeat(np.array([[0,1]]),25,axis=0)

    x_in = np.concatenate((x_north,x_south),axis=0)
    y_rew = np.concatenate((y_north,y_south), axis=0)
    y_true = np.concatenate((l_north,l_south), axis=0)
    x_ctx = np.concatenate((c_north,c_south), axis=0)
    # define datasets (duplicates for shuffling)
    x_train = x_in
    y_train = y_rew
    l_train = y_true
    c_train = x_ctx


    # ------------------------------ PARAMETERS ---------------------------------------------
    # modelrdms 
    rdms,dmat = gen_modelrdms()
    N_MODELS = dmat.shape[1]
    # # which model rdm (0 grid, 1 orthogonal, 2 parallel)
    rdm_constr = dmat[:,0][:,np.newaxis].T
    # parameters 
    N_EPISODES = 50000
    N_RUNS = 10
    N_HIDDEN = 100
    EPSILON = 1e-3
    WEIGHT_INIT = 1e-3#1e-5 for h1
    WEIGHT_LL = 1
    WEIGHT_RDM = 0.3 # was 0.05 for h1
    dmat.shape[0]

    # ---------------------------- COMPUTATIONAL GRAPH ------------------------------------
    # define neural network 
    x_features = tf.placeholder(tf.float32, [None,25],'inputs')
    x_garden = tf.placeholder(tf.float32, [None,2],'ctx_node')
    y_reward  = tf.placeholder(tf.float32,[None,1],'reward')

    y_rdm     = tf.placeholder(tf.float32,[None,dmat.shape[0]],'model_rdm')

    w_hf = var_weights((25,N_HIDDEN),stdval=np.sqrt(WEIGHT_INIT))
    b_hf = var_bias((1,N_HIDDEN))

    w_hg = var_weights((2,N_HIDDEN),stdval=np.sqrt(1/2))

    x_h1 = tf.add(tf.matmul(x_features,w_hf),tf.matmul(x_garden,w_hg)) #+ b_hf
    y_h1 = tf.nn.relu(x_h1)
    w_h = var_weights((N_HIDDEN,N_HIDDEN),stdval=np.sqrt(1/N_HIDDEN))
    x_h = tf.matmul(y_h1,w_h)
    

    y_h = tf.nn.relu(x_h)
    y_h_rdm = tf_compute_rdm(tf.transpose(y_h))
    y_h_flzs = tf_zscore(tf_compute_lowertriang_flattened(y_h_rdm))
    w_out = var_weights((N_HIDDEN,1),stdval=np.sqrt(1/N_HIDDEN))
    b_out = var_bias((1,1))
    
    y_pred = tf.add(tf.matmul(y_h,w_out),b_out)


    
    loss_supervised = tf.reduce_sum(tf.pow(y_pred - y_reward,2),name='supervised_loss')
    loss_rdm = tf.reduce_sum(tf.pow(y_h_flzs - y_rdm,2),name='rdm_loss')
    loss =  WEIGHT_LL*loss_supervised + WEIGHT_RDM*loss_rdm 
    run_sgd = tf.train.GradientDescentOptimizer(EPSILON).minimize(loss)
    init_vars = tf.global_variables_initializer()  
    varstoquery = [y_pred,w_hf,w_hg,w_out,y_h,x_h,y_h1,x_h1] 

    #----------------------------- TRAINING -------------------------------------------------
    # train on n episodes (each episode = 50 trials)

    results = {
                'losses_total' : np.empty((N_MODELS,N_RUNS,N_EPISODES)),
                'losses_rdm' : np.empty((N_MODELS,N_RUNS,N_EPISODES)),
                'losses_supervised' : np.empty((N_MODELS,N_RUNS,N_EPISODES)),
                'all_x_hidden1' : np.empty((N_MODELS,N_RUNS,N_HIDDEN,x_train.shape[0])),
                'all_y_hidden1' : np.empty((N_MODELS,N_RUNS,N_HIDDEN,x_train.shape[0])),
                'all_x_hidden2' : np.empty((N_MODELS,N_RUNS,N_HIDDEN,x_train.shape[0])),
                'all_y_hidden2' : np.empty((N_MODELS,N_RUNS,N_HIDDEN,x_train.shape[0])),
                'all_y_out' : np.empty((N_MODELS,N_RUNS,1,x_train.shape[0])),            
                'n_dead1': np.empty((N_MODELS,N_RUNS,2)),
                'n_local1': np.empty((N_MODELS,N_RUNS,2)),
                'n_only_a1': np.empty((N_MODELS,N_RUNS,2)),
                'n_only_b1': np.empty((N_MODELS,N_RUNS,2)),
                'hidden_dotprod1' : np.empty((N_MODELS,N_RUNS,2)),
                'n_dead2': np.empty((N_MODELS,N_RUNS,2)),
                'n_local2': np.empty((N_MODELS,N_RUNS,2)),
                'n_only_a2': np.empty((N_MODELS,N_RUNS,2)),
                'n_only_b2': np.empty((N_MODELS,N_RUNS,2)),
                'hidden_dotprod2' : np.empty((N_MODELS,N_RUNS,2)),
                'weight_ll': WEIGHT_LL,
                'weight_rdm': WEIGHT_RDM,            
                'n_hidden': N_HIDDEN,
                'weight_init': WEIGHT_INIT,
                'lrate': EPSILON

                }
    for ii in range(N_MODELS):
        # which model rdm (0 grid, 1 orthogonal, 2 parallel)
        
        rdm_constr = dmat[:,ii][:,np.newaxis].T
        print((datetime.now().strftime("%d/%m/%Y %H:%M:%S") +  ' - model rdm {} / {}').format(ii+1,N_MODELS))
        for jj in range(N_RUNS):
            print('... run {} / {}'.format(jj+1,N_RUNS))
                

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True            
            with tf.Session(config=config) as sess:
                sess.run(init_vars)
                losses_supervised = []
                losses_rdm = []
                losses_total = []

                # stats at init 
                responses,w_h1,w_h2,w_o,y_hidden,x_hidden,y_hidden1,x_hidden1 = sess.run(varstoquery,feed_dict={x_features:x_in,x_garden:x_ctx})
                results['n_dead1'][ii,jj,0],results['n_local1'][ii,jj,0],results['n_only_a1'][ii,jj,0],results['n_only_b1'][ii,jj,0], results['hidden_dotprod1'][ii,jj,0] = compute_sparsity_stats(y_hidden1)
                results['n_dead2'][ii,jj,0],results['n_local2'][ii,jj,0],results['n_only_a2'][ii,jj,0],results['n_only_b2'][ii,jj,0], results['hidden_dotprod2'][ii,jj,0] = compute_sparsity_stats(y_hidden) 
                for ep in range(N_EPISODES):
                    # minibatch training. one batch == entire 5x5x2 stimulus space        
                    sess.run(run_sgd, feed_dict={x_features:x_train,
                                                y_reward:y_train,
                                                x_garden:c_train,
                                                y_rdm:rdm_constr})
                                                
                    # evaluation
                    rl,sl,tl = sess.run([loss_rdm,loss_supervised,loss], feed_dict={x_features:x_train,
                                                                        x_garden:c_train,
                                                                        y_reward:y_train,                                                            
                                                                        y_rdm:rdm_constr})
                    
                    losses_rdm.append(rl)
                    losses_supervised.append(sl)
                    losses_total.append(tl)
                    if (ep%1000)==0:
                        print('episode {}, interleaved, rdm loss {:.2f}, supervised loss {:.2f}, weighted total loss {:.2f}'.format(ep,rl, sl, tl))
            
                # EVALUATION            
                responses,w_h1,w_h2,w_o,y_hidden,x_hidden,y_hidden1,x_hidden1 = sess.run(varstoquery,feed_dict={x_features:x_in,x_garden:x_ctx})
                results['n_dead1'][ii,jj,1],results['n_local1'][ii,jj,1],results['n_only_a1'][ii,jj,1],results['n_only_b1'][ii,jj,1], results['hidden_dotprod1'][ii,jj,1] = compute_sparsity_stats(y_hidden1.T)
                results['n_dead2'][ii,jj,1],results['n_local2'][ii,jj,1],results['n_only_a2'][ii,jj,1],results['n_only_b2'][ii,jj,1], results['hidden_dotprod2'][ii,jj,1] = compute_sparsity_stats(y_hidden.T)
                results['losses_total'][ii,jj,:] = losses_total 
                results['losses_rdm'][ii,jj,:] = losses_rdm 
                results['losses_supervised'][ii,jj,:] = losses_supervised
                results['all_y_hidden2'][ii,jj,:] = y_hidden.T
                results['all_x_hidden2'][ii,jj,:] = x_hidden.T
                results['all_y_hidden1'][ii,jj,:] = y_hidden1.T
                results['all_x_hidden1'][ii,jj,:] = x_hidden1.T
                results['all_y_out'][ii,jj,:] = responses.T
                sess.close()
                # tf.reset_default_graph()

    with open('final_blob_mlp_2h_rdm.pickle','wb') as f:
        pickle.dump(results,f)
if __name__ == "__main__":
    main()