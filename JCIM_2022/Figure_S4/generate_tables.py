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
    target2 = 'energy'
    final_df = DataFrame(data={'layer':[], 'multiply':[]
                                , 'train_r2':[], 'train_r2_se':[]
                                , 'train_rmse':[], 'train_rmse_se':[]
                                , 'val_r2':[], 'val_r2_se':[]
                                , 'val_rmse':[], 'val_rmse_se':[]
                                , 'test_r2':[], 'test_r2_se':[]
                                , 'test_rmse':[], 'test_rmse_se':[]
                                , 'ensemble_r2':[], 'ensemble_r2_se':[]
                                , 'ensemble_rmse':[], 'ensemble_rmse_se':[],target2:[],'se':[]})
    layer_size = ['5','10','20','50','100','300','600', '1200']
    multiplier = ['3']
    seeds=['1', '2', '3', '4']
    fold_list = ['fold_' + str(i) for i in range(5)]
    #Do not change below this line
    for layer in layer_size:
        for multiply in multiplier:
            seed_dirs = [os.path.join('./multiply_'+multiply,'seed_'+seed) for seed in seeds]
            sub_frame = DataFrame(data={final_df.columns[0]: [int(layer)]
                                , final_df.columns[1]: [int(multiply)]})
            sub_data_list = [[], [], [], [], [], [], [], [], []]
            for seed_dir in seed_dirs:
                epoch_dir = os.path.join(seed_dir, 'layer_'+layer)
                test_path = os.path.join(seed_dir, 'test_full.csv')
                ensemble_path = os.path.join(epoch_dir, 'test_predictions.csv')
                test_data = read_csv(test_path)
                ensemble_predictions = read_csv(ensemble_path)
                ensemble_r2 = get_r2(test_data[target], ensemble_predictions[target])
                ensemble_rmse = get_rmse(test_data[target]
                                        , ensemble_predictions[target])
                sub_data_list[6].append(ensemble_r2)
                sub_data_list[7].append(ensemble_rmse)
                for fold in fold_list:
                    save_dir= os.path.join(epoch_dir, fold)
                    data_folder = os.path.join(seed_dir, fold)
                    val_path = os.path.join(data_folder, 'val_full.csv')
                    train_path = os.path.join(data_folder, 'train_full.csv')
                    val_data = read_csv(val_path)
                    train_data = read_csv(train_path)
                    test_preds_path = os.path.join(save_dir, 'test_preds.csv')
                    val_preds_path = os.path.join(save_dir, 'val_preds.csv')
                    train_preds_path = os.path.join(save_dir, 'train_preds.csv')
                    test_preds = read_csv(test_preds_path)
                    val_preds = read_csv(val_preds_path)
                    train_preds = read_csv(train_preds_path)
                    test_r2 = get_r2(test_data[target], test_preds[target])
                    test_rmse = get_rmse(test_data[target], test_preds[target])
                    val_r2 = get_r2(val_data[target], val_preds[target])
                    val_rmse = get_rmse(val_data[target], val_preds[target])
                    train_r2 = get_r2(train_data[target], train_preds[target])
                    train_rmse = get_rmse(train_data[target], train_preds[target])
                    target2_r2 = get_r2(train_data[target2], train_preds[target2])
                    sub_data_list[0].append(train_r2)
                    sub_data_list[1].append(train_rmse)
                    sub_data_list[2].append(val_r2)
                    sub_data_list[3].append(val_rmse)
                    sub_data_list[4].append(test_r2)
                    sub_data_list[5].append(test_rmse)
                    sub_data_list[8].append(target2_r2)
            for index, sub_list in enumerate(sub_data_list):
                column_avg = final_df.columns[2*index+2]
                column_std = final_df.columns[2*index+3]
                sub_frame[column_avg] =  np.mean(sub_list)
                sub_frame[column_std] =  np.std(sub_list,ddof=1)/len(sub_list)**0.5
            final_df = final_df.append(sub_frame, ignore_index=True)
    final_df.sort_values(by=['multiply', 'layer'], inplace=True)
    print('saving data')
    final_df.to_csv('./complexity.csv', index=False)
