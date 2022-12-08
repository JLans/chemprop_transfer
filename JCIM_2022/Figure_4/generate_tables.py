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
    data = {'weights':['.001', '.01', '.05', '0.1', '0.5', '0.75', '1', '2.5', '5', '10', '100', '1000']
          , 'val_r2':[], 'val_r2_se':[]
          , 'val_rmse':[], 'val_rmse_se':[]
          , 'test_r2':[], 'test_r2_se':[]
          , 'test_rmse':[], 'test_rmse_se':[]
          , 'ensemble_r2':[], 'ensemble_rmse':[]}

    fold_list = ['fold_' + str(i) for i in range(5)]
    data_dir = './Mathieu_5+energy_10000r'
    #Do not change below this line
    for weight in data['weights']:
        model_path = './Mathieu_5M+10000r_' + weight + 'w'
        test_path = os.path.join(data_dir, 'test_full.csv')
        ensemble_path = os.path.join(model_path, 'ensemble_preds.csv')
        test_data = read_csv(test_path)
        ensemble_predictions = read_csv(ensemble_path)
        ensemble_r2 = get_r2(test_data[target], ensemble_predictions[target])
        ensemble_rmse = get_rmse(test_data[target]
                                , ensemble_predictions[target])
        data['ensemble_r2'].append(ensemble_r2)
        data['ensemble_rmse'].append(ensemble_rmse)
        sub_dict = {
            'val_r2':[], 'val_rmse':[]
          , 'test_r2':[], 'test_rmse':[]
          }

        for fold in fold_list:
            fold_dir = os.path.join(model_path, fold)
            data_folder = os.path.join(data_dir, fold)
            val_path = os.path.join(data_folder, 'val_full.csv')
            val_data = read_csv(val_path)
            test_preds_path = os.path.join(fold_dir, 'test_preds.csv')
            val_preds_path = os.path.join(fold_dir, 'val_preds.csv')
            test_preds = read_csv(test_preds_path)
            val_preds = read_csv(val_preds_path)
            test_r2 = get_r2(test_data[target], test_preds[target])
            test_rmse = get_rmse(test_data[target], test_preds[target])
            val_r2 = get_r2(val_data[target], val_preds[target])
            val_rmse = get_rmse(val_data[target], val_preds[target])
            sub_dict['val_r2'].append(val_r2)
            sub_dict['val_rmse'].append(val_rmse)
            sub_dict['test_r2'].append(val_r2)
            sub_dict['test_rmse'].append(test_rmse)
        data['val_r2'].append(np.mean(sub_dict['val_r2']))
        data['val_r2_se'].append(np.std(sub_dict['val_r2'], ddof=1)/ len(fold_list)**0.5)
        data['val_rmse'].append(np.mean(sub_dict['val_rmse']))
        data['val_rmse_se'].append(np.std(sub_dict['val_rmse'], ddof=1)/ len(fold_list)**0.5)
        data['test_r2'].append(np.mean(sub_dict['test_r2']))
        data['test_r2_se'].append(np.std(sub_dict['test_r2'], ddof=1)/ len(fold_list)**0.5)
        data['test_rmse'].append(np.mean(sub_dict['test_rmse']))
        data['test_rmse_se'].append(np.std(sub_dict['test_rmse'], ddof=1)/ len(fold_list)**0.5)

df = DataFrame(data)
df.to_csv('./error_weighting.csv', index=False)