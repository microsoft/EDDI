'''
EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE
This code implememts partial VAE (PNP) part demonstrated on a UCI dataset.

To run this code:
python main_train_and_impute.py  --epochs 3000  --latent_dim 10 --p 0.99 --data_dir your_directory/data/boston --output_dir your_directory/model

possible arguments:
- epochs: number of epochs.
- latent_dim: size of latent space of partial VAE.
- p: upper bound for artificial missingness probability. For example, if set to 0.9, then during each training epoch, the algorithm will
  randomly choose a probability smaller than 0.9, and randomly drops observations according to this probability.
  Our suggestion is that if original dataset already contains missing data, you can just set p to 0.
- batch_size: mini batch size for training. default: 100
- iteration: iterations (number of minibatches) used per epoch. set to -1 to run the full epoch.
  If your dataset is large, please set to other values such as 10.
- K: dimension of the feature map (h) dimension of PNP encoder. Default: 20
- M: Number of MC samples when perform imputing. Default: 50
- data_dir: Directory where UCI dataset is stored.
- output_dir: Directory where the trained model will be stored and loaded.

Other comments:
- We assume that the data is stored in an excel file named d0.xls,
   and we assume that the last column is the target variable of interest (only used in active learning)
   you should modify the load data section according to your data.
- Note that this code assumes a Gaussian noise real valued data. You may need to modify the likelihood function for other types of data.
- In preprocessing, we chose to squash the data to the range of 0 and 1. Therefore our decoder output has also been squashed
  by a sigmoid function. If you wish to change the preprocessing setting, you may also need to change the decoder setting accordingly.
  This can be found in coding.py.

File Structure:
- main functions:
  main_train_impute.py: implements the training of partial VAE (PNP) part demonstrated on a UCI dataset.
  main_active_learning.py: implements the EDDI active learning strategy, together with a global single ordering strategy based on partial VAE demonstrated on a UCI dataset
                           it will also generate a information curve plot.
- decoder-encoder functions: coding.py
- partial VAE class:p_vae.py
- training-impute functions: train_and_test_functions.py
- training-active learning functions:active_learning_functions.py
- active learning visualization: boston_bar_plot.py, this will visualize the decision process of eddi on Boston Housing data.
- data: data/boston/d0.xls

'''
### load models and functions
from train_and_test_functions_dropfix import *
#### Import libraries
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from random import sample
import pandas as pd
import sklearn.preprocessing as preprocessing
plt.switch_backend('agg')
tfd = tf.contrib.distributions

### load data
Data = pd.read_excel(UCI + '/d0.xls')
Data = Data.as_matrix()
### data preprocess
max_Data = 1  #
min_Data = 0  #
Data_std = (Data - Data.min(axis=0)) / (Data.max(axis=0) - Data.min(axis=0))
Data = Data_std * (max_Data - min_Data) + min_Data
Mask = np.ones(Data.shape) # This is a mask indicating missingness, 1 = observed, 0 = missing.
# this UCI data is fully observed. you should modify the set up of Mask if your data contains missing data.

### split the data into train and test sets
Data_train, Data_test, mask_train, mask_test = train_test_split(
        Data, Mask, test_size=0.1, random_state=rs)

### Train the model and save the trained model.
vae = train_p_vae(Data_train,mask_train, args.epochs, args.latent_dim, args.batch_size,args.p, args.K,args.iteration)

### Test imputating model on the test set
## Calculate test ELBO of observed test data (will load the pre-trained model). Note that this is NOT imputing.
tf.reset_default_graph()
test_loss = test_p_vae_marginal_elbo(Data_test,mask_test, args.latent_dim,args.K)
## Calculate imputation RMSE (conditioned on observed data. will load the pre-trained model)
## Note that here we perform imputation on a new dataset, whose observed entries are not used in training.
## this will under estimate the imputation performance, since in principle all observed entries should be used to train the model.
tf.reset_default_graph()
Data_ground_truth = Data_test
mask_obs = np.array([bernoulli.rvs(1 - 0.3, size=Data_ground_truth.shape[1]*Data_ground_truth.shape[0])]) # manually create missing data on test set
mask_obs = mask_obs.reshape(Data_ground_truth.shape)
Data_observed = Data_ground_truth*mask_obs

mask_target = 1-mask_obs
# This line below is optional. Turn on this line means that we use the new comming testset to continue update the imputing model. Turn off this linea means that we only use the pre-trained model to impute without futher updating the model.
# vae = train_p_vae(Data_ground_truth,mask_obs, args.epochs, args.latent_dim, args.batch_size,0, args.K,args.iteration)
tf.reset_default_graph()
RMSE = impute_p_vae(Data_observed,mask_obs,Data_ground_truth,mask_target,args.latent_dim,args.batch_size,args.K,args.M)





