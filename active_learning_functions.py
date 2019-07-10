from __future__ import division
from p_vae import *
from codings import *
import numpy as np
import tensorflow as tf
from scipy.stats import bernoulli
import argparse
import os
import random
from random import sample

#### parser configurations
parser = argparse.ArgumentParser(
    description='EDDI')
parser.add_argument(
    '--epochs',
    type=int,
    default=3000,
    metavar='N_eps',
    help='number of epochs to train (default: 3000)')
parser.add_argument(
    '--latent_dim',
    type=int,
    default=10,
    metavar='LD',
    help='latent dimension (default: 10)')
parser.add_argument(
    '--p',
    type=float,
    default=0.7,
    metavar='probability',
    help='dropout probability of artificial missingness during training')
parser.add_argument(
    '--iteration',
    type=int,
    default=-1,
    metavar='it',
    help='iterations per epoch. set to -1 to run the full epoch. ')
parser.add_argument(
    '--batch_size',
    type=int,
    default=100,
    metavar='batch',
    help='Mini Batch size per epoch.  ')
parser.add_argument(
    '--K',
    type=int,
    default=20,
    metavar='K',
    help='Dimension of PNP feature map ')
parser.add_argument(
    '--M',
    type=int,
    default=50,
    metavar='M',
    help='Number of MC samples when perform imputing')
parser.add_argument(
    '--eval',
    type=str,
    default='rmse',
    metavar='eval',
    help='eval: evaluation metric of active learning. ''rmse'':rmse; ''nllh'':negative log likelihood')
parser.add_argument(
    '--repeat',
    type=int,
    default=5,
    metavar='repeat',
    help='Number of repeats of the active learning experiment')
parser.add_argument(
    '--output_dir',
    type=str,
    default=os.getenv('PT_OUTPUT_DIR', '/tmp'))
parser.add_argument(
    '--data_dir',
    type=str,
    default=os.getenv('PT_DATA_DIR', 'data'),
    help='Directory where UCI dataset is stored.')
args = parser.parse_args()

#### Set directories
UCI = args.data_dir
ENCODER_WEIGHTS = os.path.join(args.output_dir, 'encoder.tensorflow')
FINETUNED_DECODER_WEIGHTS = os.path.join(args.output_dir, 'generator.tensorflow')
rs = 42 # random seed

def p_vae_active_learning(Data_train,mask_train,Data_test,mask_test,epochs,latent_dim,batch_size,p,K,M,eval,Repeat,estimation_method=0):
    '''
    This function loads a pretrained p-VAE model, and performs active learning using single global strategy.
    Note that we assume that the last column of x is the target variable of interest
    :param Data_train: training data matrix
    :param mask_train: mask matrix that indicates the missingness of training data. 1=observed, 0 = missing
    :param Data_test: test data matrix
    :param mask_test: mask matrix that indicates the missingness of test data. 1=observed, 0 = missing
    :param latent_dim: latent dimension of partial VAE.
    :param K: dimension of feature map of PNP encoder
    :param M: number of samples used for MC sampling
    :param eval: evaluation metric of active learning. 'rmse':rmse; 'nllh':negative log likelihood
    :param Repeat: number of repeats.
    :param estimation_method: what method to use for single ordering information reward estimation.
            In order to calculate the single best ordering, we need to somehow marginalize (average) the
            information reward over the data set (in this case, the test set).
            we provide two methods of marginalization.
            - estimation_method = 0: information reward marginalized using the model distribution p_{vae_model}(x_o).
            - estimation_method = 1: information reward marginalized using the data distribution p_{data}(x_o)
    :return: None (active learning results are saved to args.output_dir)
    '''

    for r in range(Repeat):
        ## train partial VAE
        tf.reset_default_graph()
        vae = train_p_vae(Data_train,mask_train, epochs, latent_dim,batch_size, p, K,10)
        n_test = Data_test.shape[0]
        n_train = Data_train.shape[0]
        OBS_DIM = Data_test.shape[1]

        # kwargs = {
        #     'K': K,
        #     'obs_distrib': "Gaussian",
        #     'latent_dim': latent_dim,
        #     'encoder': PNP_fc_uci_encoder,
        #     'decoder': fc_uci_decoder,
        #     'obs_dim': OBS_DIM,
        #     'load_model': 1,
        #     'decoder_path': FINETUNED_DECODER_WEIGHTS,
        #     'encoder_path': ENCODER_WEIGHTS,
        # }
        # vae = PN_Plus_VAE(**kwargs)

        ## create arrays to store results
        if r == 0:
            # information curves
            information_curve_RAND = np.zeros(
                (Repeat, n_test, OBS_DIM - 1  + 1))
            information_curve_SING = np.zeros(
                (Repeat, n_test, OBS_DIM - 1  + 1))
            information_curve_CHAI = np.zeros(
                (Repeat, n_test, OBS_DIM - 1 + 1))

            # history of optimal actions
            action_SING = np.zeros((Repeat, n_test,
                                    OBS_DIM - 1 ))
            action_CHAI = np.zeros((Repeat, n_test,
                                    OBS_DIM - 1))

            # history of information reward values
            R_hist_SING = np.zeros(
                (Repeat, OBS_DIM - 1 , n_test,
                 OBS_DIM - 1 ))
            R_hist_CHAI = np.zeros(
                (Repeat, OBS_DIM - 1, n_test,
                 OBS_DIM - 1))

            # history of posterior samples of partial inference
            im_SING = np.zeros((Repeat, OBS_DIM - 1 , M,
                                n_test, OBS_DIM ))
            im_CHAI = np.zeros((Repeat, OBS_DIM - 1, M,
                                n_test, OBS_DIM))

        ## Perform active variable selection with random ordiner and SING (single sequence)
        for strategy in range(3):

            if strategy == 0:### random strategy
                ## create arrays to store data and missingness
                x = Data_test[:, :]  #
                x = np.reshape(x, [n_test, OBS_DIM])
                mask = np.zeros((n_test, OBS_DIM))
                mask[:, -1] = 0  # we will never observe target value

                ## initialize array that stores optimal actions (i_optimal)
                i_optimal = [
                    nums for nums in range(OBS_DIM - 1 )
                ]
                i_optimal = np.tile(i_optimal, [n_test, 1])
                random.shuffle([random.shuffle(c) for c in i_optimal])

                ## evaluate likelihood at initial stage (no observation)
                negative_predictive_llh, uncertainty = vae.predictive_loss(
                    x, mask,eval, M)
                information_curve_RAND[r, :, 0] = negative_predictive_llh
                for t in range(OBS_DIM - 1 ):
                    print("Repeat = {:.1f}".format(r))
                    print("Strategy = {:.1f}".format(strategy))
                    print("Step = {:.1f}".format(t))
                    io = np.eye(OBS_DIM)[i_optimal[:, t]]
                    mask = mask + io
                    negative_predictive_llh, uncertainty = vae.predictive_loss(
                        x, mask,eval, M)
                    information_curve_RAND[r, :, t +
                                           1] = negative_predictive_llh

            elif strategy == 1:### single ordering strategy "SING"
                #SING is obtrained by maximize mean information reward for each step for the test set to be consistant with the description in the paper.
                #We can also get this order by using a subset of training set to obtain the optimal ordering and apply this to the testset.
                x = Data_test[:, :]  #
                x = np.reshape(x, [n_test, OBS_DIM])
                mask = np.zeros((n_test, OBS_DIM)) # this stores the mask of missingness (stems from both test data missingness and unselected features during active learing)
                mask2 = np.zeros((n_test, OBS_DIM)) # this stores the mask indicating that which features has been selected of each data
                mask[:, -1] = 0  # Note that no matter how you initialize mask, we always keep the target variable (last column) unobserved.

                negative_predictive_llh, uncertainty = vae.predictive_loss(
                    x, mask,eval, M)
                information_curve_SING[r, :, 0] = negative_predictive_llh

                for t in range(OBS_DIM - 1 ): # t is a indicator of step
                    print("Repeat = {:.1f}".format(r))
                    print("Strategy = {:.1f}".format(strategy))
                    print("Step = {:.1f}".format(t))
                    ## note that for single ordering, there are two rewards.
                    # The first one (R) is calculated based on no observations.
                    # This is used for active learning phase, since single ordering should not depend on observations.
                    # The second one (R_eval) is calculated in the same way as chain rule approximation. This is only used for visualization.
                    R = -1e4 * np.ones(
                        (n_test, OBS_DIM - 1)
                    )


                    im_0 = completion(x, mask*0, M, vae) # sample from model prior
                    im = completion(x, mask, M, vae) # sample conditional on observations
                    im_SING[r, t, :, :, :] = im

                    for u in range(OBS_DIM - 1): # u is the indicator for features. calculate reward function for each feature candidates
                        loc = np.where(mask2[:, u] == 0)[0]


                        if estimation_method == 0:
                            R[loc, u] = R_lindley_chain(u, x, mask, M, vae, im_0,loc)
                        else:
                            R[loc, u] = R_lindley_chain(u, x, mask, M, vae, im, loc)

                    R_hist_SING[r, t, :, :] = R
                    i_optimal = (R.mean(axis=0)).argmax() # optimal decision based on reward averaged on all data
                    i_optimal = np.tile(i_optimal, [n_test])
                    io = np.eye(OBS_DIM)[i_optimal]

                    action_SING[r, :, t] = i_optimal
                    mask = mask + io*mask_test # this mask takes into account both data missingness and missingness of unselected features
                    negative_predictive_llh, uncertainty = vae.predictive_loss(
                        x, mask,eval, M)
                    mask2 = mask2 + io # this mask only stores missingess of unselected features, i.e., which features has been selected of each data
                    information_curve_SING[r, :, t +
                                           1] = negative_predictive_llh


            elif strategy == 2:  ### EDDI strategy (chain rule approximation)
                # personalized active feature selection strategy
                ## create arrays to store data and missingness
                x = Data_test[:, :]  #
                x = np.reshape(x, [n_test, OBS_DIM])
                mask = np.zeros((n_test, OBS_DIM))  # this stores the mask of missingness (stems from both test data missingness and unselected features during active learing)
                mask2 = np.zeros((n_test,OBS_DIM))  # this stores the mask indicating that which features has been selected of each data
                mask[:,-1] = 0  # Note that no matter how you initialize mask, we always keep the target variable (last column) unobserved.
                ## evaluate likelihood at initial stage (no observation)
                negative_predictive_llh, uncertainty = vae.predictive_loss(
                    x, mask, eval,M)
                information_curve_CHAI[r, :, 0] = negative_predictive_llh

                for t in range(OBS_DIM - 1): # t is a indicator of step
                    print("Repeat = {:.1f}".format(r))
                    print("Strategy = {:.1f}".format(strategy))
                    print("Step = {:.1f}".format(t))
                    R = -1e4 * np.ones((n_test, OBS_DIM - 1))
                    im = completion(x, mask, M, vae)
                    im_CHAI[r, t, :, :, :] = im

                    for u in range(OBS_DIM - 1): # u is the indicator for features. calculate reward function for each feature candidates
                        loc = np.where(mask[:, u] == 0)[0]

                        R[loc, u] = R_lindley_chain(u, x, mask, M, vae, im,
                                                    loc)
                    R_hist_CHAI[r, t, :, :] = R
                    i_optimal = R.argmax(axis=1)
                    io = np.eye(OBS_DIM)[i_optimal]

                    action_CHAI[r, :, t] = i_optimal

                    mask = mask + io # this mask takes into account both data missingness and missingness of unselected features
                    negative_predictive_llh, uncertainty = vae.predictive_loss(
                        x, mask, eval,M)
                    mask2 = mask2 + io # this mask only stores missingess of unselected features, i.e., which features has been selected of each data
                    print(mask2[0:5, :])

                    information_curve_CHAI[r, :, t +
                                                 1] = negative_predictive_llh

    # Save results
    np.savez(
        os.path.join(
            args.output_dir,
            'UCI_information_curve_RAND.npz'),
        information_curve=information_curve_RAND)
    np.savez(
        os.path.join(
            args.output_dir,
            'UCI_information_curve_SING.npz'),
        information_curve=information_curve_SING)
    np.savez(
        os.path.join(
            args.output_dir,
            'UCI_information_curve_CHAI.npz'),
        information_curve=information_curve_CHAI)
    np.savez(
        os.path.join(args.output_dir,
                     'UCI_action_SING.npz'),
        action=action_SING)
    np.savez(
        os.path.join(args.output_dir,
                     'UCI_action_CHAI.npz'),
        action=action_CHAI)
    np.savez(
        os.path.join(args.output_dir,
                     'UCI_R_hist_SING.npz'),
        R_hist=R_hist_SING)
    np.savez(
        os.path.join(args.output_dir,
                     'UCI_R_hist_CHAI.npz'),
        R_hist=R_hist_CHAI)
    np.savez(
        os.path.join(args.output_dir,
                     'UCI_im_SING.npz'),
        im=im_SING)
    np.savez(
        os.path.join(args.output_dir,
                     'UCI_im_CHAI.npz'),
        im=im_CHAI)

    return None



def train_p_vae(Data_train,mask_train, epochs, latent_dim,batch_size, p, K,iteration):
    '''
        This function trains the partial VAE.
        :param Data_train: training Data matrix, N by D
        :param mask_train: mask matrix that indicates the missingness. 1=observed, 0 = missing
        :param epochs: number of epochs of training
        :param LATENT_DIM: latent dimension for partial VAE model
        :param p: dropout rate for creating additional missingness during training
        :param K: dimension of feature map of PNP encoder
        :param iteration: how many mini-batches are used each epoch. set to -1 to run the full epoch.
        :return: trained VAE, together with the test data used for testing.
        '''

    obs_dim = Data_train.shape[1]
    n_train = Data_train.shape[0]
    list_train = np.arange(n_train)
    ####### construct
    kwargs = {
        'K': K,
        'obs_distrib': "Gaussian",
        'latent_dim': latent_dim,
        'batch_size': batch_size,
        'encoder': PNP_fc_uci_encoder,
        'decoder': fc_uci_decoder,
        'obs_dim': obs_dim,
        'load_model':0,
        'decoder_path': FINETUNED_DECODER_WEIGHTS,
        'encoder_path': ENCODER_WEIGHTS,
    }
    vae = PN_Plus_VAE(**kwargs)



    if iteration == -1:
        n_it = int(np.ceil(n_train / float(kwargs['batch_size'])))
    else:
        n_it = iteration

    for epoch in range(epochs):
        training_loss_full = 0.

        # test_loss, test_kl, test_recon = vae.full_batch_loss(Data_test,mask_test)
        # test_loss = test_loss
        # test_kl = test_kl / n_test
        # test_recon = test_recon / n_test

        # iterate through batches

        # np.random.shuffle(list_train)
        for it in range(n_it):

            if iteration == -1:
                batch_indices = list_train[it*kwargs['batch_size']:min(it*kwargs['batch_size'] + kwargs['batch_size'], n_train - 1)]
            else:
                batch_indices = sample(range(n_train), kwargs['batch_size'])

            x = Data_train[batch_indices, :]
            mask_train_batch = mask_train[batch_indices, :]
            # DROPOUT_TRAIN = np.minimum(np.random.rand(1), p)
            # while True:
            #     mask_drop = np.array([bernoulli.rvs(1 - DROPOUT_TRAIN, size=obs_dim)] *
            #                          kwargs['batch_size'])
            #     if np.sum(mask_drop>0):
            #         break

            DROPOUT_TRAIN = np.minimum(np.random.rand(mask_train_batch.shape[0], obs_dim), p)
            while True:
                # mask_drop = np.array([bernoulli.rvs(1 - DROPOUT_TRAIN)] )
                mask_drop = bernoulli.rvs(1 - DROPOUT_TRAIN)
                if np.sum(mask_drop > 0):
                    break

            mask_drop = mask_drop.reshape([kwargs['batch_size'], obs_dim])
            _ = vae.update(x, mask_drop*mask_train_batch)
            loss_full, _, _ = vae.full_batch_loss(x,mask_drop*mask_train_batch)
            training_loss_full += loss_full

        # average loss over most recent epoch
        training_loss_full /= n_it
        print(
            'Epoch: {} \tnegative training ELBO per observed feature: {:.2f}'
            .format(epoch, training_loss_full))

    vae.save_generator(FINETUNED_DECODER_WEIGHTS)
    vae.save_encoder(ENCODER_WEIGHTS)

    return vae