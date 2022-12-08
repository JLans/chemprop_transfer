# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 21:49:23 2021

@author: joshua.l.lansford
"""
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from chemprop_transfer.plotting_tools import set_figure_settings

set_figure_settings('paper')
def plot_data(df, xdata, axis, labels=[], colors=[], markers=[], legend=[]
, subfig='', ylim=[], xlim=[]):
    for count, label in enumerate(labels):
        if type(label) == list:
            axis.errorbar(xdata, df[label[0]]
            ,yerr=df[label[1]]
            , ecolor=colors[count], elinewidth=1, capsize=2, barsabove=True
            , marker=markers[count], linestyle='None')
        else:
            axis.errorbar(xdata, df[label], marker=markers[count]
            , linestyle='None')
    axis.legend(legend, loc='best', frameon=False, handlelength=1)
    axis.text(0.01,0.9, subfig, transform=axis.transAxes)
    axis.set_ylim(ylim)
    axis.set_xlim(xlim)

colors = ['b', 'r', 'g']
markers = ['o', '^', 's']

fig = plt.figure(figsize=(3.5,3.2),dpi=400)
axes = fig.subplots(nrows=1, ncols=1)
weighting = read_csv('./error_weighting.csv')
legend = ['Avg. CV error', 'Avg. test error', 'Ensemble error']
labels = [['val_rmse', 'val_rmse_se'], ['test_rmse', 'test_rmse_se'], 'ensemble_rmse']
plot_data(weighting, np.log10(weighting['weights']), axes, labels, colors, markers, legend
          , ylim=[0.20, 0.45], xlim=[-3.1, 3.1])
fig.set_tight_layout('tight')
plt.xlabel(r'log$_{10}$(weighting) for log$_{10}$(H$_{50}$)')
plt.ylabel(r'RMSE [log(H$_{50}$]')
plt.savefig('./weighting.jpg', format='jpg')
plt.close()