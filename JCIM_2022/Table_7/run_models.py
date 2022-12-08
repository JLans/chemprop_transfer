# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:57:02 2021

@author: lansf
"""
import os
from chemprop_transfer.data import split_data
from chemprop_transfer.data import combine_files
import shutil
from chemprop_transfer.data import DATASET
from chemprop.args import TrainArgs
from chemprop.train import cross_validate, run_training
from chemprop_transfer.utils import Transfer_Model
from chemprop.train.make_predictions import make_predictions
from chemprop.args import PredictArgs
data_path = data_path = r'../../chemprop_transfer/data/ani_properties_sorted.csv'
data = DATASET(data_path)
data.load_data()
data.data = data.data[data.data['fmax'] < 0.05]
data.data = data.data.loc[:, data.data.columns != 'fmax'][0:5000]
data.normalize()
data_path_FF = './ani_all.csv'
data.save('./ani_all.csv')

directories = ['direct_model', './ani_all', './ani_energy', './ani_energy+fcs', './ani_fcs'
               , './ani_energy+vib', './ani_vib', './ani_energy+sym'
               , './ani_sym', './ani_energy+MOI', './ani_MOI']
target_columns = [[], ['energy','FC_1', 'FC_2', 'FC_3', 'SYM'
                   , 'Hvib75', 'Hvib600', 'TSvib75', 'TSvib600'
                   , 'MOI1', 'MOI2', 'MOI3']
                  , ['energy'], ['energy','FC_1', 'FC_2', 'FC_3']
                   , ['FC_1', 'FC_2', 'FC_3']
                   , ['energy', 'Hvib75', 'Hvib600', 'TSvib75', 'TSvib600']
                   , ['Hvib75', 'Hvib600', 'TSvib75', 'TSvib600']
                   , ['energy', 'SYM'], ['SYM']
                   , ['energy', 'MOI1', 'MOI2', 'MOI3']
                   , ['MOI1', 'MOI2', 'MOI3']]
seed=1
data = DATASET('../../chemprop_transfer/data/log_Mathieu_2020_CHNOFCl.csv')
data.load_data()
data.filter_mols(['C','O','N','H'])
data_path_Mathieu = './log_Mathieu_2020_CHNO.csv'
data.save(data_path_Mathieu)
num_folds=5 #5


seed=1
M_dir = './Mathieu_data'
temp_dir = './temp'
val = 0.1
test = .29333
split_data(data_path_Mathieu, (1-val-test, val, test), seed=seed, num_folds=num_folds
           , save_dir=M_dir)
split_data(data_path_FF, (1, 0, 0), seed=seed, num_folds=num_folds
           , save_dir=temp_dir)

combined_directory = './combined_10M+all'
combine_files(combined_directory, [M_dir, temp_dir], multiply=[10,1])
shutil.rmtree(temp_dir)

if __name__ == '__main__':
    for count, directory in enumerate(directories):
        loss_weighting = ['1']
        ffn = [300, 300, 20, (50, 50)]
        for item in target_columns[count]:
            loss_weighting.append(str(1/len(target_columns[count])))
            ffn.append((50,50))
        separate_test_path = os.path.join(combined_directory, 'test_full.csv')
        fold_list = ['fold_' + str(i) for i in range(num_folds)]
        base_model = Transfer_Model()
        for fold in fold_list:
            fold_out = os.path.join(directory,fold)
            fold_in = os.path.join(combined_directory,fold)
            separate_val_path = os.path.join(fold_in, 'val_full.csv')
            train_path = os.path.join(fold_in, 'train_full.csv')
            # training arguments
            additional_args = [
                '--data_path', train_path,
                '--separate_val_path', separate_val_path,
                '--separate_test_path', separate_test_path,
                '--save_dir', fold_out,
                '--epochs', '10', #10
                '--batch_size', '25', #25
                '--final_lr', '0.00005', #.00005
                '--init_lr', '0.00001', #.00001
                '--max_lr', '0.001', #0.0005
                '--hidden_size', '300', '--bias']
            additional_args.append('--loss_weighting')
            for item in loss_weighting:
                additional_args.append(item)
            additional_args.append('--multi_branch_ffn')
            additional_args.append(str(tuple(ffn)))
            additional_args.append('--target_columns')
            additional_args.append('log10h50(exp)')
            for item in target_columns[count]:
                additional_args.append(item)
            train_args = base_model.get_train_args(additional_args)
            args=TrainArgs().parse_args(train_args)
            #train a model on DFT data for pretraining
            mean_score, std_score = cross_validate(args=args
                                                   , train_func=run_training)
            
            predict_args = ['--checkpoint_dir', fold_out
                            , '--test_path', separate_val_path
                            , '--preds_path', os.path.join(fold_out, 'val_preds.csv')
                            ,'--num_workers', '0'
                            ]
            prediction_args = PredictArgs().parse_args(predict_args)
            make_predictions(args=prediction_args)
            
            predict_args = ['--checkpoint_dir', fold_out
                            , '--test_path', train_path
                            , '--preds_path', os.path.join(fold_out, 'train_preds.csv')
                            ,'--num_workers', '0'
                            ]
            prediction_args = PredictArgs().parse_args(predict_args)
            make_predictions(args=prediction_args)
            
        predict_args = ['--checkpoint_dir', directory
                        , '--test_path', separate_test_path
                        , '--preds_path', os.path.join(directory, 'ensemble_preds.csv')
                        ,'--num_workers', '0'
                        ]
        prediction_args = PredictArgs().parse_args(predict_args)
        make_predictions(args=prediction_args)

