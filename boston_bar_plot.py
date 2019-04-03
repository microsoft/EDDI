'''
- This file demonstrates the visualization of active learning process of Boston Housing.

    Please first run the active learning in main_active_learning.py
    Then, to run this file:
    python boston_bar_plot.py -- data_dir your_directory/data/boston/ -- output_dir your_directory/model/

- Each figure named in the format "_SING_avg_bars_x.png" (where x is the step of active learning) shows the distribution
of information reward function of each feature at this specific step on the y-axis.
All unobserved variables start with green bars, and turns purple once selected by the algorithm. . Step 0 corresponds to no features has been selected.
All reward will be updated after each decision of feature selection, and the reward of selected features will
will be fixed.

- Each figure named in the format "_SING_avg_violins_x.png" (where x is the step of active learning) shows the
violin plot of the posterior density estimations of remaining unobserved variables at this specific step

'''
import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing


parser = argparse.ArgumentParser(
    description='EDDI')
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


#### load data
data_grid = np.array([0,1,2,3,4,5,6,9,10,12])
data = 1
rs = 42
UCI = args.data_dir
Data = pd.read_excel(UCI + 'd0.xls')
Data = Data.as_matrix()
OBS_DIM = Data.shape[1]
### UCI data preprocess
if data_grid[data-1] == 12:
    size_feature = Data.shape[1]
    n_data = Data.shape[0]
else:
    max_Data = 1  #
    min_Data = 0  #
    Data_std = (Data - Data.min(axis=0)) / (Data.max(axis=0) - Data.min(axis=0))
    Data = Data_std * (max_Data - min_Data) + min_Data
    size_feature = Data.shape[1]
    n_data = Data.shape[0]


Data_train, Data_test, Data_train, Data_test = train_test_split(
        Data, Data, test_size=0.1, random_state=rs)





UCI_SING = args.output_dir


npzfile = np.load(UCI_SING+'UCI_R_hist_SING.npz')
R_SING=npzfile['R_hist']
R_SING = np.maximum(R_SING,0)
npzfile = np.load(UCI_SING+'UCI_action_SING.npz')
A_SING=npzfile['action']
npzfile = np.load(UCI_SING+'UCI_im_SING.npz')
im_SING=npzfile['im']

K = 3
T=10
repeat = 0
        



### SING strategy over all data

pos = np.arange(R_SING.shape[1])+1
A_SING_0_t = []
A_SING_t_minus_1 = []
R_SING_t_minus_1 = []
bars1 = []
r1 = []
r2 = pos
for t in range(T):
    # Create bars T=1
    A_SING_t = A_SING[repeat,K,t]
    R_SING_t = R_SING[repeat,t,:,:].mean(axis=0)*100
    barWidth = 0.9
    bars2 = R_SING_t
    if t >0:
        bars1 = np.append(bars1, R_SING_t_minus_1[int(A_SING_t_minus_1)])
        bars2 = np.delete(bars2,A_SING_0_t.astype(int))
     
    # The X position of bars
    if t>0:
        r1 = np.append(r1,int(A_SING_t_minus_1)+1)
        r2 = np.delete(r2,np.argwhere(r2==int(A_SING_t_minus_1)+1))
     
    # Create barplot
    if t>0:
        plt.bar(r1, bars1, width = barWidth, color = (0.3,0.1,0.4,0.6), label='Selected')
    plt.bar(r2, bars2, width = barWidth, color = (0.3,0.9,0.4,0.6), label='Unselected')
    # Note: the barplot could be created easily. See the barplot section for other examples.
     
    # Create legend
    plt.legend()
     
    # Text below each barplot with a rotation at 90°
    #plt.xticks([r + barWidth for r in range(R_SING.shape[1])], ['crime rate','proportion of residential land zoned','proportion of non-retail business acres',' Charles River',' nitric oxides concentration','average number of rooms','owner-occupied units built prior to 1940','distances to Boston centres','accessibility to radial highways','property-tax rate',' pupil-teacher ratio','proportion of blacks','lower status of the population'], rotation=90)
    plt.xticks([r + barWidth for r in range(R_SING.shape[1])], ['CR','PRD','PNB',' CHR',' NOC','ANR','OUB','DTB','ARH','TAX',' OTR','PB','LSP',])
    
    # Create labels
     
    # Adjust the margins
    #plt.subplots_adjust(bottom= 0.3, top = 0.98)
     
    # Show graphic
    plt.savefig(UCI_SING+'_SING_avg_bars_'+str(t)+'.png', format='png', dpi=200)
    plt.show()
    
    
    A_SING_t_minus_1 = A_SING_t
    R_SING_t_minus_1 = R_SING_t
    A_SING_0_t = np.append(A_SING_0_t,A_SING_t)
    
    
     ## plot voilin plot
    im = im_SING[repeat,t,:,K,:]
    target = Data_test[K,:]
    if t >0:
        im[:,(r1-1).astype(int)] = target[(r1-1).astype(int)]
        
    M = im.shape[0]
    obs_dim = im.shape[1]
    GR_name = np.array(['CR','PRD','PNB',' CHR',' NOC','ANR','OUB','DTB','ARH','TAX',' OTR','PB','LSP','PRC'])
    GR_label_im = np.array([])
    GR_label_target = np.array([])
    IM = np.reshape(im.T,(M*obs_dim,))
    for i in range(M*obs_dim):
        GR_label_im = np.append(GR_label_im,   GR_name[int(np.floor(i/M))])
    df_im = pd.DataFrame(dict(Score = IM, Group = GR_label_im))
    
    
    for i in range(obs_dim):
        GR_label_target = np.append(GR_label_target,   GR_name[int(i)])
    df_target = pd.DataFrame(dict(Score = target, Group = GR_label_target))
    
    
    # library &amp; dataset
    import seaborn as sns
     
    
    # Use a color palette
    plt.legend()
    ax1 = sns.violinplot( x=df_im["Group"], y=df_im["Score"],palette="Blues",scale="count",bw=0.5)
    ax1.set_ylabel('') 
    ax2 = sns.pointplot(x=df_target["Group"], y=df_target["Score"], join = False,markers = 'x',scatter_kws={"s": 0.1},color = 'black')
    ax2.set_ylabel('') 
    plt.savefig(UCI_SING+'SING_avg_violins_'+str(t)+'.png', format='png', dpi=200)
    plt.show()
