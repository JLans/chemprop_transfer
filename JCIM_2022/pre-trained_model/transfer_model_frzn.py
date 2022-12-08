# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""
from chemprop.train import cross_validate, run_training
from chemprop_transfer.utils import Transfer_Model
from chemprop.args import TrainArgs
from chemprop.train.make_predictions import make_predictions
from chemprop.args import PredictArgs
import os
base_dir = './ani_energies'
base_model = Transfer_Model()
Mathieu_dir = './Mathieu_frzn_MPN'
separate_test_path = os.path.join(Mathieu_dir, 'test_full.csv')
fold_list = ['fold_' + str(i) for i in range(5)]
for index, fold in enumerate(fold_list):
    fold_folder = os.path.join(Mathieu_dir,fold)
    separate_val_path = os.path.join(fold_folder, 'val_full.csv')
    data_path = os.path.join(fold_folder, 'train_full.csv')
    checkpoint_dir = os.path.join(base_dir, fold)
    if __name__ == '__main__': # and '__file__' in globals()
        # training arguments
        additional_args = [
            '--data_path', data_path,
            '--separate_val_path', separate_val_path,
            '--separate_test_path', separate_test_path,
            '--checkpoint_dir', checkpoint_dir,
            '--frzn_ffn_layers', '0',
            '--save_dir', fold_folder,
            '--epochs', '100', #10
            '--batch_size', '5', #25
            '--final_lr', '0.00005', #.00005
            '--init_lr', '0.00001', #.00001
            '--max_lr', '0.001', #0.0005
            #'--ffn_hidden_size', '300','20', '1000', '1000',
            #'--loss_weighting', weighting,
            '--hidden_size', '300',
            '--multi_branch_ffn', "(300, 300, 20, (50, 50))",
            '--bias'
        ]

        #train a model on DFT data for pretraining
        train_args = base_model.get_train_args(additional_args)
        args=TrainArgs().parse_args(train_args)
        
        del args.split_sizes
        mean_score, std_score = cross_validate(args=args, train_func=run_training)
        
        predict_args = ['--checkpoint_dir', fold_folder
                        , '--test_path', separate_test_path
                        , '--preds_path', os.path.join(Mathieu_dir,'test_predictions_'+str(index)+'.csv')
                        ,'--num_workers', '0'
                        #,'--no_features_scaling'
                        ]
        prediction_args = PredictArgs().parse_args(predict_args)
        make_predictions(args=prediction_args)
    
    predict_args = ['--checkpoint_dir', Mathieu_dir
                    , '--test_path', separate_test_path
                    , '--preds_path', os.path.join(Mathieu_dir,'test_predictions.csv')
                    ,'--num_workers', '0'
                    #,'--no_features_scaling'
                    ]
    prediction_args = PredictArgs().parse_args(predict_args)
    make_predictions(args=prediction_args)
        
        
        
    
     