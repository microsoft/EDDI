'''
EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE
This code implements EDDI global single ordering strategy
 based on partial VAE (PNP), demonstrated on a UCI dataset.

To run this code:
python main_active_learning.py  --epochs 3000  --latent_dim 10 --p 0.99 --data_dir your_directory/data/boston --output_dir your_directory/model


possible arguments:
- epochs: number of epochs.
- latent_dim: size of latent space of partial VAE.
- p: upper bound for artificial missingness probability. For example, if set to 0.9, then during each training epoch, the algorithm will
  randomly choose a probability smaller than 0.9, and randomly drops observations according to this probability.
  Our suggestion is that if original dataset already contains missing data, you can just set p to 0.
- batch_size: mini batch size for training. default: 100
- iteration: iterations (number of minibatches) used per epoch. set to -1 to run the full epoch.
  If your dataset is large, please set this to other values such as 10.
- K: dimension of the feature map (h) dimension of PNP encoder. Default: 20
- M: Number of MC samples when perform imputing. Default: 50
- eval: evaluation metric of active learning. 'rmse':rmse; 'nllh':negative log likelihood
- repeat: Number of repeats of the active learning experiment
- data_dir: Directory where UCI dataset is stored.
- output_dir: Directory where the trained model will be stored and loaded.

Other comments:
- We assume that the data is stored in an excel file named d0.xls,
   and we assume that the last column is the target variable of interest (only used in active learning)
   you may need to modify the load data section according to your task.
- Note that this code assumes a Gaussian noise real valued data. You may need to modify the likelihood function for other types of data.
- In preprocessing, we chose to squash the data to the range of 0 and 1. Therefore our decoder output has also been squashed
  by a sigmoid function. If you wish to change the preprocessing setting, you may also need to change the decoder setting accordingly.
  This can be found in coding.py.

File Structure:
- main functions:
  main_train_impute.py: implements the training of partial VAE (PNP) part demonstrated on a UCI dataset.
  main_active_learning.py: implements a global single ordering strategy based on partial VAE demonstrated on a UCI dataset
                           it will also generate a information curve plot.
- decoder-encoder functions: coding.py
- partial VAE class:p_vae.py
- training-impute functions: train_and_test_functions.py
- training-active learning functions:active_learning_functions.py
- active learning visualization: boston_bar_plot.py, this will visualize the decision process of eddi on Boston Housing data.
- data: data/boston/d0.xls
'''
### load models and functions
from active_learning_functions import *
#### Import libraries
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
Mask = np.ones(Data.shape) # this UCI data is fully observed
Data_train, Data_test, mask_train, mask_test = train_test_split(
        Data, Mask, test_size=0.1, random_state=rs)


### run training and active learning
# #number of experiments that you want to repeat
Repeat = args.repeat
#Training Partial VAE and then apply Random order feature selection (RAND in the paper) and single order feature selection (SING in the paper)
#generate information curve and per step information gain with RAND and SING.
p_vae_active_learning(Data_train,mask_train,Data_test,mask_test,args.epochs,args.latent_dim,args.batch_size,args.p,args.K,args.M,args.eval,Repeat)


### visualize active learning
npzfile = np.load(args.output_dir+'/UCI_information_curve_RAND.npz')
IC_RAND=npzfile['information_curve']

npzfile = np.load(args.output_dir+'/UCI_information_curve_SING.npz')
IC_SING=npzfile['information_curve']

npzfile = np.load(args.output_dir+'/UCI_information_curve_CHAI.npz')
IC_CHAI=npzfile['information_curve']


import matplotlib.pyplot as plt
plt.figure(0)
L = IC_SING.shape[1]
fig, ax1 = plt.subplots()

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.45, 0.4, 0.45, 0.45]

if args.eval == 'rmse':
    ax1.plot(np.sqrt((IC_RAND[:,:,0:].mean(axis = 0)).mean(axis=0)), 'gs', linestyle='-.', label='PNP+RAND')
    ax1.errorbar(np.arange(IC_RAND.shape[2]), np.sqrt((IC_RAND[:,:,0:].mean(axis = 0)).mean(axis=0)),
                 yerr=np.sqrt((IC_RAND[:,:,0:]).mean(axis=1)).std(axis=0) / np.sqrt(IC_SING.shape[0]), ecolor='g', fmt='gs')
    ax1.plot(np.sqrt((IC_SING[:,:,0:].mean(axis = 0)).mean(axis=0)), 'ms', linestyle='-.', label='PNP+SING')
    ax1.errorbar(np.arange(IC_SING.shape[2]),np.sqrt((IC_SING[:,:,0:].mean(axis = 0)).mean(axis=0)),
                 yerr=np.sqrt((IC_SING[:,:,0:]).mean(axis=1)).std(axis=0) / np.sqrt(IC_SING.shape[0]), ecolor='m', fmt='ms')
    ax1.plot(np.sqrt((IC_CHAI[:,:,0:].mean(axis = 0)).mean(axis=0)), 'ks', linestyle='-.', label='PNP+EDDI')
    ax1.errorbar(np.arange(IC_CHAI.shape[2]), np.sqrt((IC_CHAI[:,:,0:].mean(axis = 0)).mean(axis=0)),
                 yerr=np.sqrt((IC_CHAI[:,:,0:]).mean(axis=1)).std(axis=0) / np.sqrt(IC_CHAI.shape[0]), ecolor='k', fmt='ks')

    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('avg. test RMSE', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    ax1.legend(bbox_to_anchor=(0.0, 1.02, 1., .102), mode="expand", loc=3,
               ncol=1, borderaxespad=0., prop={'size': 20}, frameon=False)
    plt.show()
    plt.savefig(args.output_dir + '/PNP_all_IC_curves.png', format='png', dpi=200, bbox_inches='tight')
else:
    ax1.plot((IC_RAND[:,:,0:].mean(axis=1)).mean(axis = 0),'gs',linestyle = '-.', label =  'PNP+RAND')
    ax1.errorbar(np.arange(IC_RAND.shape[2]), (IC_RAND[:,:,0:].mean(axis=1)).mean(axis = 0), yerr=(IC_RAND[:,:,0:].mean(axis=1)).std(axis = 0)/np.sqrt(IC_SING.shape[0]),ecolor='g',fmt = 'gs')
    ax1.plot((IC_SING[:,:,0:].mean(axis=1)).mean(axis = 0),'ms',linestyle = '-.', label = 'PNP+SING')
    ax1.errorbar(np.arange(IC_SING.shape[2]), (IC_SING[:,:L,0:].mean(axis=1)).mean(axis = 0), yerr=(IC_SING[:,0:L,0:].mean(axis=1)).std(axis = 0)/np.sqrt(IC_SING.shape[0]),ecolor='m',fmt = 'ms')
    ax1.plot((IC_CHAI[:,:,0:].mean(axis=1)).mean(axis = 0),'ks',linestyle = '-.', label = 'PNP+EDDI')
    ax1.errorbar(np.arange(IC_CHAI.shape[2]), (IC_CHAI[:,:L,0:].mean(axis=1)).mean(axis = 0), yerr=(IC_CHAI[:,0:L,0:].mean(axis=1)).std(axis = 0)/np.sqrt(IC_CHAI.shape[0]),ecolor='k',fmt = 'ks')

    plt.xlabel('Steps',fontsize=18)
    plt.ylabel('avg. neg. test likelihood',fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    ax1.legend(bbox_to_anchor=(0.0, 1.02, 1., .102), mode = "expand", loc=3,
               ncol=1, borderaxespad=0.,prop={'size': 20}, frameon=False)
    plt.show()
    plt.savefig(args.output_dir+'/PNP_all_IC_curves.png', format='png', dpi=200,bbox_inches='tight')


