
from __future__ import division
from p_vae import *
from codings import *
import numpy as np
import tensorflow as tf
from scipy.stats import bernoulli
import argparse
import os
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
            DROPOUT_TRAIN = np.minimum(np.random.rand(mask_train_batch.shape[0],obs_dim), p)
            while True:
                # mask_drop = np.array([bernoulli.rvs(1 - DROPOUT_TRAIN)] )
                mask_drop = bernoulli.rvs(1 - DROPOUT_TRAIN)
                if np.sum(mask_drop>0):
                    break

            # mask_drop = mask_drop.reshape([kwargs['batch_size'], obs_dim])
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

def test_p_vae_marginal_elbo(Data_test,mask_test,latent_dim,K):
    '''
    This function computes the marginal (negative) ELBO of observed test data
    Note that this function does not perform imputing.
    :param Data_test: test data matrix
    :param mask_test: mask matrix that indicates the missingness. 1=observed, 0 = missing
    :param latent_dim: latent dimension for partial VAE model
    :param K:dimension of feature map of PNP encoder
    :return: test negative ELBO
    '''

    obs_dim = Data_test.shape[1]
    ####### construct
    kwargs = {
        'K': K,
        'obs_distrib': "Gaussian",
        'latent_dim': latent_dim,
        'encoder': PNP_fc_uci_encoder,
        'decoder': fc_uci_decoder,
        'obs_dim': obs_dim,
        'load_model':1,
        'decoder_path': FINETUNED_DECODER_WEIGHTS,
        'encoder_path': ENCODER_WEIGHTS,
    }
    vae = PN_Plus_VAE(**kwargs)

    test_loss, _, _ = vae.full_batch_loss(Data_test,mask_test)


    print('test negative ELBO per feature: {:.2f}'
            .format(test_loss))


    return test_loss

def impute_p_vae(Data_observed,mask_observed,Data_ground_truth,mask_target,latent_dim,batch_size,K,M):
    '''
    This function loads a pretrained p-VAE model, and performs imputation and returns RMSE.
    :param Data_observed: observed data matrix
    :param mask_observed: mask matrix that indicates the missingness of observed data. 1=observed, 0 = missing
    :param Data_ground_truth: ground truth data for calculating RMSE performance. It can also contain missing data.
    :param mask_target: ask matrix that indicates the missingness of ground truth data. 1=observed, 0 = missing
    :param latent_dim: latent dimension for partial VAE model
    :param batch_size: Mini-batch size. We evaluate test RMSE in batch mode in order to handle large dataset.
    :param K: dimension of feature map of PNP encoder
    :param M: number of samples used for MC sampling
    :return: RMSE
    '''
    obs_dim = Data_observed.shape[1]
    ####### construct
    n_data = Data_observed.shape[0]
    # mask_test_obs = 1 * (Data_observed != missing_value_indicator)  # mask of test observed entries
    # mask_test_target = 1 * (Data_ground_truth != missing_value_indicator)  # mask of test missingness to be imputed
    mask_test_obs = mask_observed
    mask_test_target = mask_target
    kwargs = {
        'K': K,
        'obs_distrib': "Gaussian",
        'latent_dim': latent_dim,
        'batch_size': batch_size,
        'encoder': PNP_fc_uci_encoder,
        'decoder': fc_uci_decoder,
        'obs_dim': obs_dim,
        'load_model': 1,
        'decoder_path': FINETUNED_DECODER_WEIGHTS,
        'encoder_path': ENCODER_WEIGHTS,
        'M': M,
    }
    vae = PN_Plus_VAE(**kwargs)

    impute_loss_SE = 0.
    list_data = np.arange(n_data)
    # np.random.shuffle(list_data)

    n_it = int(np.ceil(n_data / float(kwargs['batch_size'])))
    # iterate through batches
    for it in range(n_it):
        batch_indices = list_data[it*kwargs['batch_size']:min(it*kwargs['batch_size'] + kwargs['batch_size'], n_data - 1)]

        impute_loss_SE_batch, impute_loss_RMSE_batch = vae.impute_losses(Data_ground_truth[batch_indices, :], mask_test_obs[batch_indices, :],
                                                        mask_test_target[batch_indices, :])

        impute_loss_SE += impute_loss_SE_batch

    impute_loss_RMSE = np.sqrt(impute_loss_SE/(np.sum(mask_test_target)))


    X_fill = vae.get_imputation( Data_ground_truth, mask_test_obs)

    print('test impute RMSE eddi (estimation 1): {:.2f}'
            .format(impute_loss_RMSE))


    return impute_loss_RMSE, X_fill
