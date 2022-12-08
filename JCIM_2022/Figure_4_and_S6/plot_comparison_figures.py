# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""
from chemprop_transfer.data import DATASET
import matplotlib.pyplot as plt
from chemprop_transfer.plotting_tools import set_figure_settings
set_figure_settings('paper')
colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:purple']

data_path_Mathieu = r'../../chemprop_transfer/data/log_Mathieu_2020_CHNOFCl.csv'
data_path_GDB = '../../chemprop_transfer/data/Casey_DFT_data.csv'
data_path_PNNL = '../../chemprop_transfer/data/ani_properties_sorted.csv'

Mathieu_data = DATASET(data_path_Mathieu) 
Mathieu_heavy_natoms = Mathieu_data.get_num_heavy_atoms()
Mathieu_H_natoms = Mathieu_data.get_num_H_atoms()


GDB_data = DATASET(data_path_GDB) 
GDB_heavy_natoms = GDB_data.get_num_heavy_atoms()
GDB_H_natoms = GDB_data.get_num_H_atoms()

PNNL_data = DATASET(data_path_PNNL) 
PNNL_heavy_natoms = PNNL_data.get_num_heavy_atoms()
PNNL_H_natoms = PNNL_data.get_num_H_atoms()


fig = plt.figure(figsize=(3.5,5),dpi=400)
axes = fig.subplots(nrows=2, ncols=1)
axes[0].hist([Mathieu_heavy_natoms, GDB_heavy_natoms, PNNL_heavy_natoms], 10, density=True
         , label=['Mathieu dataset', 'CSBB dataset', 'LBRJ dataset'], color=colors[0:3])
axes[0].set_xlabel('Number of heavy atoms')
axes[0].set_ylabel('Probabiliyt density histogram')
axes[0].legend(['Mathieu dataset', 'CSBB dataset', 'LBRJ dataset'], frameon=False)
axes[0].text(0.003,0.93, '(a)', transform=axes[0].transAxes)

axes[1].hist([Mathieu_H_natoms, GDB_H_natoms, PNNL_H_natoms], 10, density=True
         , label=['Mathieu dataset', 'CSBB dataset', 'LBRJ dataset'], color=colors[0:3])
axes[1].set_xlabel('Number of H atoms')
axes[1].set_ylabel('Probabiliyt density histogram')
axes[1].text(0.003,0.93, '(b)', transform=axes[1].transAxes)
axes[1].legend(['Mathieu dataset', 'CSBB dataset', 'LBRJ dataset'], frameon=False)
fig.set_tight_layout({'pad':0.5,'w_pad':0.25,'h_pad':0.5})
plt.savefig('./n_atoms.jpg', format='jpg')
plt.close()

path_Mathieu_DFT = r'./Mathieu_Casey_similar.csv'
path_Mathieu_self = r'./Mathieu_self_similar.csv'
path_Mathieu_PNNL = r'./Mathieu_PNNL_similar.csv'

Mathieu_DFT_data = DATASET(path_Mathieu_DFT, usecols=['similarities'])
Mathieu_self_data = DATASET(path_Mathieu_self, usecols=['similarities'])
Mathieu_PNNL_data = DATASET(path_Mathieu_PNNL, usecols=['similarities'])
Mathieu_DFT_data.load_data()
Mathieu_self_data.load_data()
Mathieu_PNNL_data.load_data()



plt.figure(5, figsize=(3.5,2.75), dpi=400)
plt.hist([Mathieu_self_data.data['similarities']
        , Mathieu_DFT_data.data['similarities']
        ,Mathieu_PNNL_data.data['similarities']], 10, density=True
         , label=['Self-similarity', 'CSBB dataset', 'LBRJ dataset']
         ,color=colors[0:3])
plt.xlabel('Similarity to energetic dataset')
plt.ylabel('Probability density histogram')
plt.xlim([0, 0.6])
plt.legend(['Self-similarity', 'CSBB dataset', 'LBRJ dataset'], frameon=False)
plt.savefig('./similarities.jpg', format='jpg')
plt.close()
