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
from chemprop_transfer.data import split_data

#generating datasets
data_path = r'../../chemprop_transfer/data/ani_properties_sorted.csv'
data = DATASET(data_path)
data.load_data()
data.data = data.data[data.data['fmax'] < 0.05]
data.data = data.data[['smiles', 'energy']]
data.normalize()
data.canonicalize_data(column='smiles')
rows = [300, 900, 9000, 30000, 90000]
row_names = ['r_'+ str(row) for row in rows]
data_base = './ani_energy_'
for i, row_name in enumerate(row_names):
    data.save(data_base+row_name+'.csv', rows=rows[i])

#splitting data into training, validation, and test sets.
num_folds=5
val=0.15
test = 0.307
seeds=[1, 2, 3]
for row_name in row_names:
    for seed in seeds:
        save_dir = os.path.join(row_name, 'seed_'+str(seed))
        split_data(data_base+row_name+'.csv', (1-val-test, val, test), seed=seed, num_folds=num_folds
                   , save_dir=save_dir)


if __name__ == '__main__':
    epochs = ['1', '2', '3', '5', '10']
    fold_list = ['fold_' + str(i) for i in range(5)]
    #Do not change below this line
    for epoch in epochs:
        for row_name in row_names:
            seed_dirs = [os.path.join(row_name,'seed_'+str(seed)) for seed in seeds]
            for seed_dir in seed_dirs:
                epoch_dir = os.path.join(seed_dir, 'epochs_'+epoch)
                separate_test_path = os.path.join(seed_dir, 'test_full.csv')
                for fold in fold_list:
                    save_dir= os.path.join(epoch_dir, fold)
                    data_folder = os.path.join(seed_dir, fold)
                    separate_val_path = os.path.join(data_folder, 'val_full.csv')
                    train_path = os.path.join(data_folder, 'train_full.csv')
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
                        '--batch_size', '5',
                        '--final_lr', '0.00005',
                        '--init_lr', '0.00001',
                        '--max_lr', '0.001',
                        '--hidden_size', '300',
                        '--depth', '4',
                        '--bias',
                        #'--ffn_hidden_size', '300',
                        #'--ffn_num_layers', '2',
                        '--aggregation', 'sum',
                        '--multi_branch_ffn', "(300, 300, 20, (50, 50))"
                        ]
                    args=TrainArgs().parse_args(train_args)
                    mean_score, std_score = cross_validate(args=args
                                                           , train_func=run_training)

                    predict_args = ['--checkpoint_dir', save_dir
                            , '--test_path', separate_val_path
                            , '--preds_path', os.path.join(save_dir,'val_preds.csv')
                            ,'--num_workers', '0']
                    prediction_args = PredictArgs().parse_args(predict_args)
                    make_predictions(args=prediction_args)

                    predict_args = ['--checkpoint_dir', save_dir
                            , '--test_path', train_path
                            , '--preds_path', os.path.join(save_dir,'train_preds.csv')
                            ,'--num_workers', '0']
                    prediction_args = PredictArgs().parse_args(predict_args)
                    make_predictions(args=prediction_args)
           
                predict_args = ['--checkpoint_dir', epoch_dir
                                , '--test_path', separate_test_path
                                , '--preds_path', os.path.join(epoch_dir,'ensemble_preds.csv')
                                ,'--num_workers', '0']
                prediction_args = PredictArgs().parse_args(predict_args)
                make_predictions(args=prediction_args)
