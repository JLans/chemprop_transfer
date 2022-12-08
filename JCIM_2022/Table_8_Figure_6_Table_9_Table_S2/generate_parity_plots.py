# -*- coding: utf-8 -*-
import os
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from chemprop_transfer.plotting_tools import set_figure_settings
#from scipy import stats
#from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import MolFromSmiles


def get_files(directory, name):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename == name:
                matches.append(os.path.join(root,filename))
    return matches

Observed_values = []
Ensemble_values = []
Smiles = []
SE_values = []
seeds = ['./best_model/seed_'+str(i+1) for i in range(4)]
for seed in seeds:
    Smiles.append(read_csv(get_files(seed,'test_full.csv')[0]
                                    )['smiles'][0:94].to_numpy())
    Observed_values.append(read_csv(get_files(seed,'test_full.csv')[0]
                                    )['log10h50(exp)'][0:94].to_numpy())
    Ensemble_values.append(read_csv(get_files(seed,'ensemble_preds.csv')[0]
                                    )['log10h50(exp)'][0:94].to_numpy())
    test_value = []
    test_pred_files = get_files(seed,'test_preds.csv')[0::2]
    for file in test_pred_files:
        test_value.append(read_csv(file)['log10h50(exp)'][0:94].to_numpy())
    #t_score = stats.t.ppf(1-0.025, len(test_value)-1)
    SE = np.std(test_value, axis=0, ddof=1) / len(test_value)**0.5 #* t_score * (1+1/len(test_value))**0.5
    SE_values.append(SE)


set_figure_settings('paper')
def plot_data(xdata, ydata, yerror, axis, subfig='', ylim=[], xlim=[]):
    axis.errorbar(xlim, ylim, marker='None'
    , linestyle='-',color='k',zorder=1)
    axis.errorbar(xdata, ydata
    ,yerr=yerror, color='tab:blue'
    , ecolor='tab:blue', elinewidth=1, capsize=2, barsabove=True
    , marker='o', linestyle='None', zorder=2)
    
    #min_value = min(min(ydata), min(xdata))
    #max_value = max(max(ydata), max(xdata))
    axis.text(0.02,0.92, subfig, transform=axis.transAxes)
    axis.set_ylim(ylim)
    axis.set_xlim(xlim)



fig = plt.figure(figsize=(7.2,5),dpi=400)
axes = fig.subplots(nrows=2, ncols=2)
xlim = [0.65, 2.75]
ylim = [0.65, 2.75]


plot_data(Observed_values[0], Ensemble_values[0], SE_values[0], axes[0,0]
          , '(a)', ylim, xlim)
axes[0,0].set_xticks([])

plot_data(Observed_values[1], Ensemble_values[1], SE_values[1], axes[0,1]
          , '(b)', ylim, xlim)
axes[0,1].set_xticks([])
axes[0,1].set_yticks([])

plot_data(Observed_values[2], Ensemble_values[2], SE_values[2], axes[1,0]
          , '(c)', ylim, xlim)

plot_data(Observed_values[3], Ensemble_values[3], SE_values[3], axes[1,1]
          , '(b)', ylim, xlim)
axes[1,1].set_yticks([])

fig.text(0.001, 0.5, r'Predicted [log$_{10}$(H$_{50}$)]', va='center', rotation='vertical')
fig.text(0.5, 0.01, r'Observed [log$_{10}$(H$_{50}$)]', ha='center')
fig.set_tight_layout({'pad':1.5,'w_pad':0.25,'h_pad':0.25})
plt.savefig('./parity_plots.jpg', format='jpg')
plt.close()

for i in range(4):
    index = np.argmax(np.abs(Observed_values[i] - Ensemble_values[i]))
    print(Smiles[i][index])
    mol = MolFromSmiles(Smiles[i][index])
    mol
