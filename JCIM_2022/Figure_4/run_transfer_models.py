# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""
from chemprop.train import cross_validate, run_training
from chemprop_transfer.utils import Transfer_Model
from chemprop.train.make_predictions import make_predictions
from chemprop.args import PredictArgs
import os
data_dir = r'./Mathieu_5+energy_10000r'
if __name__ == '__main__':
    for weighting in ['.001', '.01', '.05', '0.1', '0.5', '0.75', '1', '2.5', '5', '10', '100', '1000']:
        folder = './Mathieu_5M+10000r_'+weighting+'w'
        combined_dir = './combined_5M+10000r_'+weighting+'w'
        separate_test_path = os.path.join(data_dir, 'test_full.csv')
        fold_list = ['fold_' + str(i) for i in range(5)]
        for fold in fold_list:
            data_folder = os.path.join(data_dir, fold)
            fold_folder = os.path.join(folder,fold)
            separate_val_path = os.path.join(data_folder, 'val_full.csv')
            data_path = os.path.join(data_folder, 'train_full.csv')
            
            # training arguments
            additional_args = [
                '--data_path', data_path,
                '--separate_val_path', separate_val_path,
                '--separate_test_path', separate_test_path,
                '--save_dir', fold_folder,
                '--epochs', '0'
            ]
    
            #train a model on DFT data for pretraining
            BaseModel_path = os.path.join(combined_dir,fold+'/fold_0/model_0/model.pt')
            BaseModel = Transfer_Model(BaseModel_path)
            
            transfer_model, args = BaseModel.get_transfer_model(frzn_ffn_layers='all'
                                                              ,args=additional_args)
            
            
            #args.target_stds[0] = 0.1
            mean_score, std_score = cross_validate(args=args, train_func=run_training
                                                   ,model_list = [transfer_model])
            
            predict_args = ['--checkpoint_dir', fold_folder
                            , '--test_path', separate_val_path
                            , '--preds_path', os.path.join(fold_folder,'val_preds.csv')
                            ,'--num_workers', '0'
                            ]
            prediction_args = PredictArgs().parse_args(predict_args)
            make_predictions(args=prediction_args)
            
        predict_args = ['--checkpoint_dir', folder
                        , '--test_path', separate_test_path
                        , '--preds_path', os.path.join(folder,'ensemble_preds.csv')
                        ,'--num_workers', '0'
                        ]
        prediction_args = PredictArgs().parse_args(predict_args)
        make_predictions(args=prediction_args)

        
        
        
    
     