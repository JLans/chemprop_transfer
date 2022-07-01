# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:57:02 2021

@author: lansf
"""
import numpy as np
import os
from chemprop_transfer.data import DATASET
save_dir = r'../chemprop_transfer/data'
file_ends = ['.csv']
data_files = ['Mathieu_2020'+file_end for file_end in file_ends]
data_paths = [os.path.join(save_dir,file) for file in data_files]
out_files = ['./log_Mathieu_2020_CHNOFCl' + file_end for file_end in file_ends]
data = []
for i in range(len(file_ends)):
    data.append(DATASET(data_paths[i]))
    data[i].canonicalize_data()
    data[i].filter_mols(['C', 'H', 'N', 'O', 'F', 'Cl'], min_size = 2)
    data[i].data['log10h50(exp)'] = np.log10(data[i].data['h50(exp)'].astype(float))
    data[i].data = data[i].data[['smiles','log10h50(exp)']]
    data[i].save(out_files[i])