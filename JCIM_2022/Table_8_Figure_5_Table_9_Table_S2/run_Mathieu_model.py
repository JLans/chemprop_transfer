# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:57:02 2021

@author: lansf
"""
import numpy as np
from pandas import read_csv
from chemprop_transfer.h50 import get_h50
from chemprop_transfer.h50opt import get_params
from chemprop_transfer.error_metrics import get_r2
from chemprop_transfer.error_metrics import get_rmse
from chemprop_transfer.data import split_data

data_path_Mathieu = '../../chemprop_transfer/data/Mathieu_2020.csv'
test = 0.307
seeds=[1, 2, 3, 4]
for seed in seeds:
    save_dir = './seed_'+str(seed)
    split_data(data_path_Mathieu, (1-test, 0, test), seed=seed, num_folds=1
               , save_dir=save_dir)
paths = ['./seed_1/', './seed_2/', './seed_3/', './seed_4/']
test_paths = [path + 'test_full.csv' for path in paths]
train_paths = [path + 'fold_0/train_full.csv' for path in paths]

data = []
r2_list = np.zeros(len(paths))
rmse_list = np.zeros(len(paths))
r2_list_train = np.zeros(len(paths))
rmse_list_train = np.zeros(len(paths))
for i in range(len(paths)):
    train_data = read_csv(train_paths[i])
    params = get_params(train_data['smiles'].to_list()
                        , train_data['h50(exp)'].to_list())
    params = dict(kcrit=params['kcrit'].value, eta=params['eta'].value
                  , ZN=params['ZN'].value, ZO=params['ZO'].value
                  , aC2N2O=params['aC2N2O'].value, N2loss=params['N2loss'].value)
    test_data = read_csv(test_paths[i])
    log_ph50 = np.log10([get_h50(smile, params=params) for smile in test_data['smiles']])
    np.savetxt(paths[i]+'test_preds.csv', log_ph50)
    r2_list[i] = get_r2(np.log10(test_data['h50(exp)']), log_ph50)
    rmse_list[i] = get_rmse(np.log10(test_data['h50(exp)']), log_ph50)
    log_ph50_train = np.log10([get_h50(smile, params=params) for smile in train_data['smiles']])
    np.savetxt(paths[i]+'train_preds.csv', log_ph50_train)
    
print(r2_list.mean())
print(r2_list.std(ddof=1)/r2_list.size)
print(rmse_list.mean())
print(rmse_list.std(ddof=1)/rmse_list.size)