
# EDDI

EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE (https://arxiv.org/pdf/1809.11142.pdf) 



For the ease of evaluation, we provide two versions of the code.

The first version only contains the missing data imputation model using Partial VAE. 

The second version presents both training and partial VAE and use the EDDI for feature selection (EDDI and SING in the paper). 



*****Version One: Missing Data Imputation*****

This code implements partial VAE (PNP) part demonstrated on a UCI dataset.



To run this code:

python main_train_and_impute.py  --epochs 3000  --latent_dim 10 --p 0.99 --data_dir your_directory/data/boston/ --output_dir your_directory/model/



Input: 

In this example, we use the Boston house dataset. (https://archive.ics.uci.edu/ml/machine-learning-databases/housing/)

We split the dataset into train and test set in our code. 

Random test set entries are removed for imputation quality evaluation. 



Output:

This code will save the trained model to your_directory/model. To impute new data (test dataset in this case), we can load the pre-trained model. 

In case of missing data in the training set, you can use the same train set as a test set to obtain the imputing results. 

The code also computes the test RMSE for imputed missing data and the result is printed on the console. 



*****Version Two: EDDI Optimal Feature Ordering*****

This code implements partial VAE (PNP) + EDDI/SING/RAND together demonstrated on a UCI dataset.



To run this code:



python main_active_learning.py  --epochs 3000  --latent_dim 10 --p 0.99 --data_dir your_directory/data/boston/ --output_dir your_directory/model/



Input: 

In this example, we use the Boston house dataset. (https://archive.ics.uci.edu/ml/machine-learning-databases/housing/)

We split the dataset into train and test set in our code. 

Random test set entries are removed for imputation quality evaluation. 





Output:

This code will run Partial VAE on the training set and use the trained model for the sequential feature selection as we presented in the paper. 

This code will output the result comparing EDDI, RAND and SING in the paper in the form of information curve as presented in the paper. 

This code will also output the information gain for each feature selection step. These are stored in your_directory/model.





******* Possible  Arguments ****

- epochs: number of epochs.

- latent_dim: size of latent space of partial VAE.

- p: upper bound for artificial missingness probability. For example, if set to 0.9, then during each training epoch, the algorithm will

  randomly choose a probability smaller than 0.9, and randomly drops observations according to this probability.

  Our suggestion is that if original dataset already contains missing data, you can just set p to 0.

- batch_size: minibatch size for training. default: 100

- iteration: iterations (number of mini batches) used per epoch. set to -1 to run the full epoch.

  If your dataset is large, please set to other values such as 10.

- K: the dimension of the feature map (h) dimension of PNP encoder. Default: 20

- eval: only for our active learning module. evaluation metric of active learning. 'rmse':rmse (default); 'nllh':negative log likelihood

- M: Number of MC samples when perform imputing. Default: 50

- data_dir: Directory where UCI dataset is stored.

- output_dir: Directory where the trained model will be stored and loaded.



Other comments:

- We assume that the data is stored in an excel file named d0.xls,

   and we assume that the last column is the target variable of interest (only used in active learning)

   you should modify the load data section according to your data.

- Note that this code assumes a Gaussian noise real-valued data. You may need to modify the likelihood function for other types of data.

- In preprocessing, we chose to squash the data to the range of 0 and 1. Therefore our decoder output has also been squashed

  by a sigmoid function. If you wish to change the preprocessing setting, you may also need to change the decoder setting accordingly.

  This can be found in coding.py.

  

File Structure:

- main functions:

  main_train_impute.py: implements the training of partial VAE (PNP) part demonstrated on a UCI dataset.

  main_active_learning.py: implements EDDI strategy, together with a global single ordering strategy based on partial VAE demonstrated on a UCI dataset

                           it will also generate a information curve plot.

- decoder-encoder functions: coding.py

- partial VAE class:p_vae.py

- training-impute functions: train_and_test_functions.py

- training-active learning functions:active_learning_functions.py

- active learning visualization: boston_bar_plot.py, this will visualize the decision process of eddi on Boston Housing data.

- data: data/boston/d0.xls



Dependencies:

- tensorflow 1.4.0

- scipy 1.10

- sklearn 0.19.1

- argparse 1.1


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
