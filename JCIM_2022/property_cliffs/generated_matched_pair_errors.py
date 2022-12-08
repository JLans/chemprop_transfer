# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 18:57:02 2022

@author: joshua.l.lansford
"""
import pandas as pd
from chemprop_transfer import plotting_tools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plotting_tools.set_figure_settings('paper')
df = pd.read_csv('./Mathieu_self_similar.csv')
def get_data(sim_metric, exp=True):
    if sim_metric == 'similarities_Morgan':
        ascend = False
    else:
        ascend = True
    if exp == True:
        val = 'logh50_diff'
    else:
        val = 'pred_diff'
    vals = df.sort_values(['smiles_1', sim_metric], ascending=ascend
                                    ).drop_duplicates('smiles_1')[[
                                        sim_metric, val]]
    return vals


fig = plt.figure(figsize=(7,3),dpi=400)
axs = fig.subplots(nrows=1, ncols=2)
sns.kdeplot(get_data('similarities_Morgan', exp=True).to_numpy()[:,1]
            , color='black', legend=False, ax=axs[0])
sns.kdeplot(get_data('L2_MPN', exp=True).to_numpy()[:,1]
            , color='gray', legend=False, ax=axs[0])
sns.kdeplot(get_data('L2_last', exp=True).to_numpy()[:,1]
            , color='goldenrod', legend=False, ax=axs[0])
#plt.xlim([-2,2])
axs[0].text(0.03,0.93, '(a)', transform=axs[0].transAxes)
axs[0].set_xlabel('Paired difference [log$_{10}$(H$_{50}$)]')
axs[0].set_ylabel('Probability of data')
axs[0].legend(['Morgan Fingerprint', 'MPN Layer', 'Last FFN layer'], fontsize=8)
#plt.tight_layout()
#plt.savefig(r'./features.jpg')
#plt.close()

#fig = plt.figure(figsize=(3.5,2.5),dpi=400)
#axs = fig.subplots(nrows=1, ncols=1)
sns.kdeplot(get_data('similarities_Morgan', exp=False).to_numpy()[:,1]
            , color='black', legend=False, ax=axs[1], linestyle='dashed')
sns.kdeplot(get_data('L2_MPN', exp=False).to_numpy()[:,1]
            , color='gray', legend=False, ax=axs[1], linestyle='dashed')
sns.kdeplot(get_data('L2_last', exp=False).to_numpy()[:,1]
            , color='goldenrod', legend=False, ax=axs[1], linestyle='dashed')
axs[1].text(0.03,0.93, '(b)', transform=axs[1].transAxes)
axs[1].set_xlabel('Paired difference [log$_{10}$(H$_{50}$)]')
axs[1].set_ylabel('Probability of data')
axs[1].legend(['Morgan Fingerprint', 'MPN Layer', 'Last FFN layer'], fontsize=8)
plt.xlim([-2,2])
plt.tight_layout()
plt.savefig(r'./features.jpg')
plt.close()

fig = plt.figure(figsize=(3.5,2.5),dpi=400)
plt.hist(np.abs(df['logh50_diff']) / (1 - df['similarities_Morgan']), bins=10, rwidth=0.5
         , color = 'black')
plt.ylabel('Number of data')
plt.xlabel('SALI index')
plt.savefig(r'./SALI.jpg')
plt.close()

fig = plt.figure(figsize=(3.5,2.5),dpi=400)
Morgan = get_data('pca_MPN', exp=True)
plt.plot(Morgan['pca_MPN'], Morgan['logh50_diff'], 'o')
plt.ylabel('Paired difference')
plt.xlabel('L2 Norm')
plt.savefig(r'./scale_plot.jpg')
plt.close()