# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""
import os
from pandas import read_csv
from pandas import DataFrame
from chemprop_transfer.error_metrics import get_r2
from chemprop_transfer.error_metrics import get_rmse
import numpy as np
if __name__ == '__main__':
    target = 'log10h50(exp)'
    #Do not change below this line
    data = {'directoryCs':['./Mathieu_direct', './combined_10M+DFT', './combined_10M+ani']
            , 'train_r2':[], 'train_r2_se':[]
            , 'train_rmse':[], 'train_rmse_se':[]
            , 'val_r2':[], 'val_r2_se':[]
            , 'val_rmse':[], 'val_rmse_se':[]
            , 'test_r2':[], 'test_r2_se':[]
            , 'test_rmse':[], 'test_rmse_se':[]
            , 'ensemble_r2':[], 'ensemble_rmse':[]}
    fold_list = ['fold_' + str(i) for i in range(5)]
    #Do not change below this line
    for directory in data['directoryCs']:
        test_path = os.path.join(directory, 'test_full.csv')
        ensemble_path = os.path.join(directory, 'ensemble_preds.csv')
        test_data = read_csv(test_path)
        ensemble_predictions = read_csv(ensemble_path)
        ensemble_r2 = get_r2(test_data[target], ensemble_predictions[target])
        ensemble_rmse = get_rmse(test_data[target]
                                , ensemble_predictions[target])
        data['ensemble_r2'].append(ensemble_r2)
        data['ensemble_rmse'].append(ensemble_rmse)
        sub_dict = {
            'train_r2':[], 'train_rmse':[]
           , 'val_r2':[], 'val_rmse':[]
          , 'test_r2':[], 'test_rmse':[]
          }

        for fold in fold_list:
            fold_dir = os.path.join(directory, fold)
            train_path = os.path.join(fold_dir, 'train_full.csv')
            train_data = read_csv(train_path)
            val_path = os.path.join(fold_dir, 'val_full.csv')
            val_data = read_csv(val_path)
            test_preds_path = os.path.join(fold_dir, 'test_preds.csv')
            train_preds_path = os.path.join(fold_dir, 'train_preds.csv')
            val_preds_path = os.path.join(fold_dir, 'val_preds.csv')
            test_preds = read_csv(test_preds_path)
            train_preds = read_csv(train_preds_path)
            val_preds = read_csv(val_preds_path)
            test_r2 = get_r2(test_data[target], test_preds[target])
            test_rmse = get_rmse(test_data[target], test_preds[target])
            train_r2 = get_r2(train_data[target], train_preds[target])
            train_rmse = get_rmse(train_data[target], train_preds[target])
            val_r2 = get_r2(val_data[target], val_preds[target])
            val_rmse = get_rmse(val_data[target], val_preds[target])
            sub_dict['train_r2'].append(train_r2)
            sub_dict['train_rmse'].append(train_rmse)
            sub_dict['val_r2'].append(val_r2)
            sub_dict['val_rmse'].append(val_rmse)
            sub_dict['test_r2'].append(test_r2)
            sub_dict['test_rmse'].append(test_rmse)
        data['train_r2'].append(np.mean(sub_dict['train_r2']))
        data['train_r2_se'].append(np.std(sub_dict['train_r2'], ddof=1)/ len(fold_list)**0.5)
        data['train_rmse'].append(np.mean(sub_dict['train_rmse']))
        data['train_rmse_se'].append(np.std(sub_dict['train_rmse'], ddof=1)/ len(fold_list)**0.5)
        data['val_r2'].append(np.mean(sub_dict['val_r2']))
        data['val_r2_se'].append(np.std(sub_dict['val_r2'], ddof=1)/ len(fold_list)**0.5)
        data['val_rmse'].append(np.mean(sub_dict['val_rmse']))
        data['val_rmse_se'].append(np.std(sub_dict['val_rmse'], ddof=1)/ len(fold_list)**0.5)
        data['test_r2'].append(np.mean(sub_dict['test_r2']))
        data['test_r2_se'].append(np.std(sub_dict['test_r2'], ddof=1)/ len(fold_list)**0.5)
        data['test_rmse'].append(np.mean(sub_dict['test_rmse']))
        data['test_rmse_se'].append(np.std(sub_dict['test_rmse'], ddof=1)/ len(fold_list)**0.5)

df = DataFrame(data)
df.to_csv('./model_results.csv', index=False)
