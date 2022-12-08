# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""
from chemprop.args import TrainArgs
from chemprop.train import cross_validate, run_training
from chemprop_transfer.utils import Transfer_Model
import os
save_dir = './ani_energies'
separate_test_path = os.path.join(save_dir, 'test_full.csv')
fold_list = ['fold_' + str(i) for i in range(5)]
base_model = Transfer_Model()
for fold in fold_list:
    fold_file = os.path.join(save_dir,fold)
    separate_val_path = os.path.join(fold_file, 'val_full.csv')
    data_path = os.path.join(fold_file, 'train_full.csv')
    if __name__ == '__main__': # and '__file__' in globals()
        # training arguments
    
        additional_args = [
            '--data_path', data_path,
            '--separate_val_path', separate_val_path,
            '--separate_test_path', separate_test_path,
            '--save_dir', fold_file,
            '--epochs', '10', #10
            '--batch_size', '25', #25
            '--final_lr', '0.00005', #.00005
            '--init_lr', '0.00001', #.00001
            '--max_lr', '0.001', #0.0005
            #'--ffn_hidden_size', '300','20', '1000', '1000',
            #'--loss_weighting', weighting,
            '--hidden_size', '300',
            '--multi_branch_ffn', "(300, 300, 20, (50, 50))",
            '--bias'
        ]
        train_args = base_model.get_train_args(additional_args)
        args=TrainArgs().parse_args(train_args)
        #train a model on DFT data for pretraining
        mean_score, std_score = cross_validate(args=args
                                           , train_func=run_training)