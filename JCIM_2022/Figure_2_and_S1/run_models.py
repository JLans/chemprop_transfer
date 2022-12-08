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
from chemprop_transfer.data import split_data

#splitting data
data_path_Mathieu = '../../chemprop_transfer/data/log_Mathieu_2020_CHNOFCl.csv'
num_folds=5
vals = ['0.05', '0.10', '0.15', '0.20', '0.25', '0.30']
val_dirs = ['./val_' + frac for frac in vals]
test = 0.307
seeds=[1, 2, 3, 4]
for i, val in enumerate(vals):
    for seed in seeds:
        save_dir = os.path.join(val_dirs[i], 'seed_'+str(seed))
        split_data(data_path_Mathieu, (1-float(val)-test, float(val), test), seed=seed, num_folds=num_folds
                   , save_dir=save_dir)

if __name__ == '__main__':
    epochs = ['5', '10','20','30','40','50', '100', '150', '200', '250']
    seeds=['1', '2', '3', '4']
    fold_list = ['fold_' + str(i) for i in range(5)]
    #Do not change below this line
    for epoch in epochs:
        for val_dir in val_dirs:
            seed_dirs = [os.path.join(val_dir,'seed_'+seed) for seed in seeds]
            for seed_dir in seed_dirs:
                epoch_dir = os.path.join(seed_dir, 'epochs_'+epoch)
                separate_test_path = os.path.join(seed_dir, 'test_full.csv')
                for fold in fold_list:
                    save_dir= os.path.join(epoch_dir, fold)
                    data_folder = os.path.join(seed_dir, fold)
                    separate_val_path = os.path.join(data_folder, 'val_full.csv')
                    train_path = os.path.join(data_folder, 'train_full.csv')
                    # training arguments
                    train_args = [
                        '--data_path', train_path,
                        '--separate_val_path', separate_val_path,
                        '--separate_test_path', separate_test_path,
                        '--num_workers', '0',
                        '--save_dir', save_dir,
                        '--save_preds',
                        '--show_individual_scores',
                        '--metric', 'mse',
                        '--extra_metrics', 'r2','rmse',
                        '--dataset_type', 'regression',
                        '--epochs', epoch,
                        '--batch_size', '5', #batch size 25 for testing validation set size
                        '--final_lr', '0.00005',
                        '--init_lr', '0.00001',
                        '--max_lr', '0.001',
                        '--hidden_size', '300',
                        '--depth', '4',
                        '--bias',
                        #'--ffn_hidden_size', '300', #used for simple models
                        #'--ffn_num_layers', '2',
                        '--multi_branch_ffn','(300,300,20,(50,50))', #used for branched models
                        '--aggregation', 'sum',
                        ]
                    args=TrainArgs().parse_args(train_args)
                    #train a model on DFT data for pretraining
                    mean_score, std_score = cross_validate(args=args
                                                           , train_func=run_training)
                    #making predictions for validation set
                    predict_args = ['--checkpoint_dir', save_dir
                        , '--test_path', separate_val_path
                        , '--preds_path', os.path.join(save_dir,'val_preds.csv')
                        ,'--num_workers', '0'
                        ]
                    prediction_args = PredictArgs().parse_args(predict_args)
                    make_predictions(args=prediction_args)
                    
                    #making predictions for test set
                    predict_args = ['--checkpoint_dir', save_dir
                            , '--test_path', train_path
                            , '--preds_path', os.path.join(save_dir,'train_preds.csv')
                            ,'--num_workers', '0'
                            ]
                    prediction_args = PredictArgs().parse_args(predict_args)
                    make_predictions(args=prediction_args)
                
                #making ensemble prediction on test set
                predict_args = ['--checkpoint_dir', epoch_dir
                                , '--test_path', separate_test_path
                                , '--preds_path', os.path.join(epoch_dir,'ensemble_preds.csv')
                                ,'--num_workers', '0'
                                ]
                prediction_args = PredictArgs().parse_args(predict_args)
                make_predictions(args=prediction_args)
                
                