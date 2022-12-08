# -*- coding: utf-8 -*-
from pandas import read_csv
import matplotlib.pyplot as plt
from chemprop_transfer.plotting_tools import set_figure_settings

data = read_csv('./complexity.csv')
set_figure_settings('paper')

colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:purple']
markers = ['v', '^', 's', 'o']
labels = [['train_rmse', 'train_rmse_se'], ['val_rmse', 'val_rmse_se']
            , ['test_rmse', 'test_rmse_se'], ['ensemble_rmse', 'ensemble_rmse_se']]

def plot_data(df, xdata, labels=[], colors=[], markers=[]):
    for count, label in enumerate(labels):
        plt.errorbar(xdata, df[label[0]]
        ,yerr=df[label[1]], color=colors[count]
        , ecolor=colors[count], elinewidth=1, capsize=2, barsabove=True
        , marker=markers[count], linestyle='None')


plt.figure(figsize=(3.5,2.5),dpi=400)
x_data = data['layer']
plot_data(data, x_data, labels, colors, markers)
plt.xlabel('Message passing inner dimension')
plt.ylabel(r'RMSE [log$_{10}$(H$_{50}$)]')
plt.savefig('./complexity_plot.jpg', format='jpg')
plt.close()
