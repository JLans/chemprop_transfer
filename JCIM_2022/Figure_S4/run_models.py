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
from chemprop_transfer.data import combine_files
import shutil
from chemprop_transfer.data import DATASET
#generate dataset
data_path = r'../../chemprop_transfer/data/ani_properties_sorted.csv'
data = DATASET(data_path)
data.load_data()
data.data = data.data[data.data['fmax'] < 0.05]
data.data = data.data[['smiles', 'energy']]
data.normalize()
data.canonicalize_data(column='smiles')
data_ani = './ani_energy_9000r.csv'
data.save(data_ani, rows=9000)


#split data
data_Mathieu = '../../chemprop_transfer/data/log_Mathieu_2020_CHNOFCl.csv'
num_folds=5
multiplier = ['3']
val=0.15
test = 0.307
seeds=[1, 2, 3, 4]
for multiply in multiplier:
    for seed in seeds:
        save_dir = os.path.join('./multiply_'+multiply, 'seed_'+str(seed))
        split_data(data_ani, (1, 0, 0), seed=seed, num_folds=num_folds
                   , save_dir=save_dir+'ani')
        
        split_data(data_Mathieu, (1-float(val)-test, float(val), test), seed=seed, num_folds=num_folds
                   , save_dir=save_dir+'Mathieu')
        
        combine_files(save_dir, [save_dir+'Mathieu', save_dir+'ani'], multiply=[int(multiply),1])
        shutil.rmtree(save_dir+'Mathieu')
        shutil.rmtree(save_dir+'ani')

#run models
if __name__ == '__main__':
    layer_size=['5','10','20','50','100','300','600', '1200']
    fold_list = ['fold_' + str(i) for i in range(num_folds)]
    #Do not change below this line
    for layer in layer_size:
        for multiply in multiplier:
            seed_dirs = [os.path.join('./multiply_'+multiply,'seed_'+str(seed)) for seed in seeds]
            for seed_dir in seed_dirs:
                epoch_dir = os.path.join(seed_dir, 'layer_'+layer)
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
                        '--epochs', '30',
                        '--batch_size', '25',
                        '--final_lr', '0.00005',
                        '--init_lr', '0.00001',
                        '--max_lr', '0.001',
                        '--hidden_size', layer,
                        '--depth', '4',
                        '--bias',
                        #'--ffn_hidden_size', '300',
                        #'--ffn_num_layers', '2',
                        '--aggregation', 'sum',
                        '--no_features_scaling',
                        '--multi_branch_ffn', "((50, 50, 50), (50, 50, 50))"
                        ]
                    args=TrainArgs().parse_args(train_args)
                    #train a model on DFT data for pretraining
                    mean_score, std_score = cross_validate(args=args
                                                           , train_func=run_training)

                    predict_args = ['--checkpoint_dir', save_dir
                            , '--test_path', separate_val_path
                            , '--preds_path', os.path.join(save_dir,'val_preds.csv')
                            ,'--num_workers', '0'
                            ,'--no_features_scaling'
                            ]
                    prediction_args = PredictArgs().parse_args(predict_args)
                    make_predictions(args=prediction_args)

                    predict_args = ['--checkpoint_dir', save_dir
                            , '--test_path', train_path
                            , '--preds_path', os.path.join(save_dir,'train_preds.csv')
                            ,'--num_workers', '0'
                            ,'--no_features_scaling'
                            ]
                    prediction_args = PredictArgs().parse_args(predict_args)
                    make_predictions(args=prediction_args)
           
                predict_args = ['--checkpoint_dir', epoch_dir
                                , '--test_path', separate_test_path
                                , '--preds_path', os.path.join(epoch_dir,'test_predictions.csv')
                                ,'--num_workers', '0',
                                '--no_features_scaling']
                prediction_args = PredictArgs().parse_args(predict_args)
                make_predictions(args=prediction_args)
