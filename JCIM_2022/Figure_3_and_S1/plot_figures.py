# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 21:49:23 2021

@author: joshua.l.lansford
"""
from pandas import read_csv
import matplotlib.pyplot as plt
from chemprop_transfer.plotting_tools import set_figure_settings

set_figure_settings('paper')
def plot_data(df, xdata, axis, labels=[], colors=[], markers=[], legend=[]
, subfig='', ylim=[], xlim=[]):
    for count, label in enumerate(labels):
        if type(label) == list:
            axis.errorbar(xdata, df[label[0]]
            ,yerr=df[label[1]], color=colors[count]
            , ecolor=colors[count], elinewidth=1, capsize=2, barsabove=True
            , marker=markers[count], linestyle='None', label='label')
        else:
            axis.errorbar(xdata, df[label], marker=markers[count]
            , linestyle='None')
    handles, labels = axis.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    axis.legend(handles, legend, loc='best', frameon=False, handlelength=1)
    axis.text(0.03,0.93, subfig, transform=axis.transAxes)
    #axis.text(0.93,0.92, subfig, transform=axis.transAxes)
    axis.set_ylim(ylim)
    axis.set_xlim(xlim)

colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:purple']
markers = ['v', '^', 's', 'o']

fig = plt.figure(figsize=(7.2,5),dpi=400)
axes = fig.subplots(nrows=2, ncols=2)
direct_model = read_csv('./direct_models_bs25.csv')
legend = ['training', 'validation', 'test', 'ensemble (test)']
x_data = direct_model[direct_model['val_frac']==0.05]['epochs']
xlim = [0, 300]
ylim = [0.13, 0.43]
labels = [['train_rmse', 'train_rmse_se'], ['val_rmse', 'val_rmse_se']
            , ['test_rmse', 'test_rmse_se']]#, ['ensemble_rmse', 'ensemble_rmse_se']]
plot_data(direct_model[direct_model['val_frac']==0.05], x_data, axes[0,0], labels
         , colors[0:3], markers[0:3] , legend, '(a)', ylim=ylim, xlim=xlim)
axes[0,0].set_xticks([])

plot_data(direct_model[direct_model['val_frac']==0.15], x_data, axes[0,1], labels
         , colors[0:3], markers[0:3] , legend, '(b)', ylim=ylim, xlim=xlim)
axes[0,1].set_xticks([])
axes[0,1].set_yticks([])

plot_data(direct_model[direct_model['val_frac']==0.20], x_data, axes[1,0], labels
         , colors[0:3], markers[0:3] , legend, '(c)', ylim=ylim, xlim=xlim)

plot_data(direct_model[direct_model['val_frac']==0.30], x_data, axes[1,1], labels
         , colors[0:3], markers[0:3] , legend, '(d)', ylim=ylim, xlim=xlim)
axes[1,1].set_yticks([])

fig.text(0.001, 0.5, r'RMSE [log$_{10}$(H$_{50}$)]', va='center', rotation='vertical')
fig.text(0.5, 0.01, r'Epochs with early stopping', ha='center')
fig.set_tight_layout({'pad':1.5,'w_pad':0.25,'h_pad':0.25})
plt.savefig('./direct_models_bs25.jpg', format='jpg')
plt.close()

labels = [['train_rmse', 'train_rmse_se'], ['val_rmse', 'val_rmse_se']
            , ['test_rmse', 'test_rmse_se'], ['ensemble_rmse', 'ensemble_rmse_se']]
fig = plt.figure(figsize=(7.2,5),dpi=400)
axes = fig.subplots(nrows=2, ncols=2)
direct_model = read_csv('./direct_models_bs5.csv')
x_data = direct_model[direct_model['val_frac']==0.15]['epochs']
xlim = [0, 220]
ylim = [0.08, 0.39]
labels = [['train_rmse', 'train_rmse_se'], ['val_rmse', 'val_rmse_se']
            , ['test_rmse', 'test_rmse_se'], ['ensemble_rmse', 'ensemble_rmse_se']]
plot_data(direct_model[direct_model['val_frac']==0.15], x_data, axes[0,0], labels
         , colors, markers , legend, '(a)', ylim=ylim, xlim=xlim)
axes[0,0].set_xticks([])

plot_data(direct_model[direct_model['val_frac']==0.20], x_data, axes[1,0], labels
         , colors, markers , legend, '(c)', ylim=ylim, xlim=xlim)
axes[0,1].set_xticks([])
axes[0,1].set_yticks([])

direct_model = read_csv('./direct_models_branched.csv')
x_data = direct_model[direct_model['val_frac']==0.15]['epochs']
plot_data(direct_model[direct_model['val_frac']==0.15], x_data, axes[0,1], labels
         , colors, markers , legend, '(b)', ylim=ylim, xlim=xlim)

plot_data(direct_model[direct_model['val_frac']==0.20], x_data, axes[1,1], labels
         , colors, markers , legend, '(d)', ylim=ylim, xlim=xlim)
axes[1,1].set_yticks([])

fig.text(0.001, 0.5, r'RMSE [log$_{10}$(H$_{50}$)]', va='center', rotation='vertical')
fig.text(0.5, 0.01, r'Epochs with early stopping', ha='center')
fig.set_tight_layout({'pad':1.5,'w_pad':0.25,'h_pad':0.25})
plt.savefig('./direct_models_bs5.jpg', format='jpg')
plt.close()
