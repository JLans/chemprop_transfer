# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""
import os
from chemprop.args import TrainArgs
from chemprop.train import cross_validate, run_training
from chemprop.train.make_predictions import make_predictions
from chemprop.args import PredictArgs
from chemprop_transfer.data import DATASET

data_path = '../../chemprop_transfer/data/Casey_DFT_train_norm.csv'
data = DATASET(data_path)
data.load_data()
data.data = data.data.sample(frac=1, random_state=1)
data.data['electronic_energy (Hartree)'].iloc[420:] = ''
train_path = './Casey_DFT_train_norm_energy420r.csv'
data.save(train_path)

target_columns = ['crystal_density (g/cc)', 'heat_of_formation (kcal/mol)'
                  , 'dipole_moment (Debye)', 'HOMO_LUMO_gap (eV)']
data_path = '../../chemprop_transfer/data/Casey_DFT_val_norm.csv'
data = DATASET(data_path)
data.load_data()
data.data = data.data.sample(frac=1, random_state=1)[['smiles', 'electronic_energy (Hartree)']]
data.data[target_columns] = ''
separate_val_path = './Casey_DFT_val_energy.csv'
data.save(separate_val_path)
separate_test_path = '../../chemprop_transfer/data/Casey_DFT_test_norm.csv'

save_dirs = ['./casey_CD', './casey_delH', './casey_dipole', './casey_HLg']
save_dirs = [save_dir + '_big' for save_dir in save_dirs]
if __name__ == '__main__':
    for count, save_dir in enumerate(save_dirs):
         # and '__file__' in globals()
        # training arguments
        train_args = [
            '--data_path', train_path,
           '--separate_val_path', separate_val_path,
            '--separate_test_path', separate_test_path,
            #'--split_sizes', '0.8','0.1','0.1',
            '--num_workers', '0',
            '--save_dir', save_dir,
            '--save_preds',
            '--show_individual_scores',
            '--metric', 'r2',
            '--extra_metrics', 'mse','rmse',
            '--dataset_type', 'regression',
            '--epochs', '50',
            '--batch_size', '25',
            '--final_lr', '0.0001',
            '--init_lr', '0.0001',
            '--max_lr', '0.001',
            '--hidden_size', '2048',
            '--depth', '4',
            '--bias',
            '--aggregation', 'sum',
            '--target_columns', 'electronic_energy (Hartree)', target_columns[count],
            #'--ffn_hidden_size', '1000','20', '1000', '1000',
            '--loss_weighting', '1.0', '0.001',
            '--multi_branch_ffn', "(2048, 2048, 256, 128, 64, (64, 32, 16), (64, 32, 16))"
            ]
        args=TrainArgs().parse_args(train_args)
        #train a model on DFT data for pretraining
        mean_score, std_score = cross_validate(args=args
                                               , train_func=run_training)
    
        predict_args = ['--checkpoint_dir', save_dir
                , '--test_path', separate_val_path
                , '--preds_path', os.path.join(save_dir,'val_preds.csv')
                ,'--num_workers', '0'
                ]
        prediction_args = PredictArgs().parse_args(predict_args)
        make_predictions(args=prediction_args)
    
        predict_args = ['--checkpoint_dir', save_dir
                , '--test_path', train_path
                , '--preds_path', os.path.join(save_dir,'train_preds.csv')
                ,'--num_workers', '0'
                ]
        prediction_args = PredictArgs().parse_args(predict_args)
        make_predictions(args=prediction_args)


