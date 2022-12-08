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
save_dir = './casey_multi_target'
if __name__ == '__main__':
    separate_val_path = '../../chemprop_transfer/data/Casey_DFT_val_norm.csv'
    train_path = '../../chemprop_transfer/data/Casey_DFT_train_norm.csv'
    separate_test_path = '../../chemprop_transfer/data/Casey_DFT_test_norm.csv'
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
        '--epochs', '10',
        '--batch_size', '25',
        '--final_lr', '0.0001',
        '--init_lr', '0.0001',
        '--max_lr', '0.001',
        '--hidden_size', '2048',
        '--depth', '4',
        '--bias',
        '--aggregation', 'sum',
        #'--ffn_hidden_size', '1000','20', '1000', '1000',
        #'--loss_weighting','1', '1.5', '2', '1.2', '1',
        '--multi_branch_ffn', "(2048, 2048, 256, 128, 64, (64, 32, 16)\
    , (64, 32, 16), (64, 32, 16), (64, 32, 16), (64, 32, 16))"
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
            , '--preds_path', os.path.join(save_dirs,'train_preds.csv')
            ,'--num_workers', '0'
            ]
    prediction_args = PredictArgs().parse_args(predict_args)
    make_predictions(args=prediction_args)


