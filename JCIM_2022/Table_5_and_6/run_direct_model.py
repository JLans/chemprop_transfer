# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""

from chemprop_transfer.data import split_data
from chemprop.args import TrainArgs
from chemprop.train import cross_validate, run_training
from chemprop_transfer.utils import Transfer_Model
import os
from chemprop_transfer.data import DATASET
from chemprop.train.make_predictions import make_predictions
from chemprop.args import PredictArgs

#extract data
seed=1
data = DATASET('../../chemprop_transfer/data/log_Mathieu_2020_CHNOFCl.csv')
data.load_data()
data.filter_mols(['C','O','N','H'])
data_path_Mathieu = './log_Mathieu_2020_CHNO.csv'
data.save(data_path_Mathieu)
num_folds=5

val = 0.1
test = .29333
directoryM = './Mathieu_direct'
split_data(data_path_Mathieu, (1-val-test, val, test), seed=seed, num_folds=num_folds
               , save_dir=directoryM)

if __name__ == '__main__':
    separate_test_path = os.path.join(directoryM, 'test_full.csv')
    fold_list = ['fold_' + str(i) for i in range(num_folds)]
    base_model = Transfer_Model()
    for fold in fold_list:
        fold_folder = os.path.join(directoryM, fold)
        separate_val_path = os.path.join(fold_folder, 'val_full.csv')
        train_path = os.path.join(fold_folder, 'train_full.csv')
        # training arguments
        
        additional_args = [
            '--data_path', train_path,
            '--separate_val_path', separate_val_path,
            '--separate_test_path', separate_test_path,
            '--save_dir', fold_folder,
            #'--epochs', '10', #10
            '--epochs', '100', #10
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
        
        predict_args = ['--checkpoint_dir', fold_folder
                        , '--test_path', train_path
                        , '--preds_path', os.path.join(fold_folder,'train_preds.csv')
                        ,'--num_workers', '0'
                        ]
        prediction_args = PredictArgs().parse_args(predict_args)
        make_predictions(args=prediction_args)
        
        predict_args = ['--checkpoint_dir', fold_folder
                        , '--test_path', separate_val_path
                        , '--preds_path', os.path.join(fold_folder,'val_preds.csv')
                        ,'--num_workers', '0'
                        ]
        prediction_args = PredictArgs().parse_args(predict_args)
        make_predictions(args=prediction_args)
    
    predict_args = ['--checkpoint_dir', directoryM
                        , '--test_path', separate_test_path
                        , '--preds_path', os.path.join(directoryM,'ensemble_preds.csv')
                        ,'--num_workers', '0'
                        ]
    prediction_args = PredictArgs().parse_args(predict_args)
    make_predictions(args=prediction_args)

