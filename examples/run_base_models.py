# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""

from chemprop_transfer.data import split_data
from chemprop_transfer.data import combine_files
import shutil
from chemprop.args import TrainArgs
from chemprop.train import cross_validate, run_training
from chemprop_transfer.utils import Transfer_Model
import os
from chemprop_transfer.data import DATASET

#extract data
data_path = r'../data/ani_properties_sorted.csv'
data = DATASET(data_path)
data.load_data()
data.data = data.data[data.data['fmax'] < 0.05]
data.data = data.data[['smiles', 'energy']]
data.normalize()
data.canonicalize_data(column='smiles')
data_path_FF = './ani_energy_10000r.csv'
data.save(data_path_FF, rows=10000)

#split data
seed=1
data_path_Mathieu = '../data/log_Mathieu_2020_CHNOFCl.csv'
num_folds=5
temp_dir = './temp'
val = 0.1
test = .29333
directory1 = './Mathieu_5+energy_10000r'
split_data(data_path_Mathieu, (1-val-test, val, test), seed=seed, num_folds=num_folds
               , save_dir=directory1)
split_data(data_path_FF, (1, 0, 0), seed=seed, num_folds=num_folds
               , save_dir=temp_dir)
#combine files    
combined_dir = './combined_5M+energy_10000r'
combine_files(combined_dir, [directory1, temp_dir], multiply=[5,1])
shutil.rmtree(temp_dir)


for weighting in ['.001', '.01', '.05', '0.1', '0.5', '0.75', '1', '5']:
    save_dir = r'./combined_5M+10000r_' + weighting+'w'
    separate_test_path = os.path.join(combined_dir, 'test_full.csv')
    fold_list = ['fold_' + str(i) for i in range(num_folds)]
    base_model = Transfer_Model()
    for fold in fold_list:
        fold_folder = os.path.join(save_dir, fold)
        data_folder = os.path.join(combined_dir, fold)
        separate_val_path = os.path.join(data_folder, 'val_full.csv')
        data_path = os.path.join(data_folder, 'train_full.csv')
        if __name__ == '__main__': # and '__file__' in globals()
            # training arguments
        
            additional_args = [
                '--data_path', data_path,
                '--separate_val_path', separate_val_path,
                '--separate_test_path', separate_test_path,
                '--save_dir', fold_folder,
                '--epochs', '10', #10
                '--batch_size', '25', #25
                '--final_lr', '0.00005', #.00005
                '--init_lr', '0.00001', #.00001
                '--max_lr', '0.001', #0.0005
                #'--ffn_hidden_size', '300','20', '1000', '1000',
                '--loss_weighting', weighting,
                '--hidden_size', '300',
                '--multi_branch_ffn', "(300, 300, 20, (50, 50), (50,50))"
            ]
            train_args = base_model.get_train_args(additional_args)
            args=TrainArgs().parse_args(train_args)
            #train a model on DFT data for pretraining
            mean_score, std_score = cross_validate(args=args
                                               , train_func=run_training)   
