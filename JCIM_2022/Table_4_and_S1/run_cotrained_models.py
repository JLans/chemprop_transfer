# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""
import os
from chemprop_transfer.data import split_data
from chemprop_transfer.data import combine_files
import shutil
from chemprop_transfer.data import DATASET
from chemprop.args import TrainArgs
from chemprop.train import cross_validate, run_training
from chemprop.train.make_predictions import make_predictions
from chemprop.args import PredictArgs

data_path = '../../chemprop_transfer/data/Casey_DFT_data.csv'
data_energy = './Casey_DFT_energy.csv'
data_delH = './Casey_DFT_delH.csv'
data = DATASET(data_path)
data.skiprows = 7
data.load_data()
data.save(data_energy, columns=['smiles', 'electronic_energy (Hartree)'])
data.save(data_delH, columns=['smiles', 'heat_of_formation (kcal/mol)'])

num_folds=3
val=0.4988
test = 0.4988
seeds=[1, 2, 3]
scaffold_dir = './casey_scaffold'
for seed in seeds:
    save_dir = os.path.join(scaffold_dir, 'seed_'+str(seed))
    split_data(data_energy, (1, 0, 0), seed=seed, num_folds=num_folds
               , save_dir=save_dir+'_energy')
    
    split_data(data_delH, (1-float(val)-test, float(val), test), seed=seed, num_folds=num_folds
               , save_dir=save_dir+'_delH', scaffold=True, balanced=True)
    combine_files(save_dir, [save_dir+'_delH', save_dir+'_energy'], multiply=[1,1])
    shutil.rmtree(save_dir+'_delH')
    shutil.rmtree(save_dir+'_energy')

random_dir = './casey_random'    
for seed in seeds:
    save_dir = os.path.join(random_dir, 'seed_'+str(seed))
    split_data(data_energy, (1, 0, 0), seed=seed, num_folds=num_folds
               , save_dir=save_dir+'_energy')
    
    split_data(data_delH, (1-float(val)-test, float(val), test), seed=seed, num_folds=num_folds
               , save_dir=save_dir+'_delH')
    combine_files(save_dir, [save_dir+'_delH', save_dir+'_energy'], multiply=[1,1])
    shutil.rmtree(save_dir+'_delH')
    shutil.rmtree(save_dir+'_energy')
    
if __name__ == '__main__':
    epochs=['10','50','100']
    fold_list = ['fold_' + str(i) for i in range(num_folds)]
    #Do not change below this line
    for directory in [scaffold_dir, random_dir]:
        for epoch in epochs:
            seed_dirs = [os.path.join(directory,'seed_'+str(seed)) for seed in seeds]
            for seed_dir in seed_dirs:
                epoch_dir = os.path.join(seed_dir, 'epochs_'+epoch)
                separate_test_path = os.path.join(seed_dir, 'test_full.csv')
                for fold in fold_list:
                    save_dir= os.path.join(epoch_dir, fold)
                    data_folder = os.path.join(seed_dir, fold)
                    separate_val_path = os.path.join(data_folder, 'val_full.csv')
                    train_path = os.path.join(data_folder, 'train_full.csv')
                     # and '__file__' in globals()
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
                        '--batch_size', '12',
                        '--final_lr', '0.0001',
                        '--init_lr', '0.0000001',
                        '--max_lr', '0.001',
                        '--hidden_size', '300',
                        '--depth', '4',
                        '--bias',
                        '--ffn_hidden_size', '300',
                        '--ffn_num_layers', '2',
                        '--aggregation', 'sum',
                        '--loss_weighting', '1.0', '0.005'
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
           
                predict_args = ['--checkpoint_dir', epoch_dir
                                , '--test_path', separate_test_path
                                , '--preds_path', os.path.join(epoch_dir,'ensemble_preds.csv')
                                ,'--num_workers', '0'
                                ]
                prediction_args = PredictArgs().parse_args(predict_args)
                make_predictions(args=prediction_args)
