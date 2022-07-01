#import torch
from chemprop_transfer.data import DATASET
from chemprop.args import TrainArgs
from chemprop.models import MoleculeModel
from chemprop.utils import load_checkpoint
from chemprop.train.make_predictions import make_predictions
from chemprop.args import PredictArgs
import numpy as np
import torch
import os
#from copy import deepcopy

def get_default_train_args():
    train_args = [
            '--save_preds',
            '--show_individual_scores',
            '--metric', 'mse',
            '--extra_metrics', 'r2','rmse',
            '--dataset_type', 'regression',
            '--depth', '4',
            '--epochs', '25', #200
            '--ffn_num_layers', '5', 
            '--ffn_hidden_size', '300','20', '1000', '1000',
            '--batch_size', '5',
            '--seed', '1',
            '--final_lr', '0.0001', #.00005 '0.00001'
            '--init_lr', '0.0000001', #.00001 '0.000001'
            '--max_lr', '0.001', #0.0001 '0.00005'
            '--num_workers', '0',
            '--aggregation', 'sum'
        ]
    return train_args

def dict_to_list(args_dict):
        #dictionary expected be formatted as follows {key: [values]}
        args_list = []
        for key  in args_dict.keys():
            args_list.append(key)
            for value in args_dict[key]:
                args_list.append(str(value))
        return args_list

def get_predictions(path, n_models, individual_preds=False, use_predictions_file=False):
    if use_predictions_file == True:
        pred_path = os.path.join(path,'test_predictions.csv')
        pred = DATASET(pred_path)
        pred.load_data()
        predictions = pred.data.to_numpy()[:,1].astype(float)
    else:
        separate_test_path = os.path.join(path, 'test_full.csv')
        
        fold_list = ['fold_' + str(i) for i in range(n_models)]
        checkpoint_paths = [os.path.join(path,fold+'/fold_0/model_0/model.pt')
                            for fold in fold_list]
        
        predict_args = dict_to_list({'--checkpoint_paths': checkpoint_paths[0:n_models]
                        , '--test_path': [separate_test_path]
                        , '--preds_path': [os.path.join(path,'test_predictions.csv')]
                        ,'--no_features_scaling': []
                        ,'--num_workers': ['0']})
        prediction_args = PredictArgs().parse_args(predict_args)
        predictions = np.array(make_predictions(args=prediction_args)).flatten()
        
        if individual_preds == True:
            for index, fold in enumerate(fold_list[0:n_models]):
                predict_args = ['--checkpoint_path', checkpoint_paths[index]
                                , '--test_path', separate_test_path
                                , '--preds_path', os.path.join(path,'test_predictions_'+str(index)+'.csv')
                                ,'--no_features_scaling'
                                ,'--num_workers', '0']
                prediction_args = PredictArgs().parse_args(predict_args)
                make_predictions(args=prediction_args)
            
    return predictions

class Transfer_Model:
    def __init__(self, path=None):
        """ 
        Parameters
        ----------
        path : str
            path to chemprop model checkpoint
        """
        if path !=None:
            state = torch.load(path, map_location=lambda storage, loc: storage)
            self.base_args = TrainArgs()
            self.base_args.from_dict(vars(state['args']), skip_unsettable=True)
            self.base_model = load_checkpoint(path)
        else:
            self.base_args = None
            self.base_model = None

    def get_existing_keys(self, args_list):
        key_indices = []
        keys = []
        for index, key in enumerate(args_list):
            if key[0:2] == '--':
                key_indices.append(index)
                keys.append(key)          
        return key_indices, keys
    
    def list_to_dict(self, args_list):
        args_dict = {}
        for index, value in enumerate(args_list):
            if value[0:2] == '--':
                args_dict.update({value:[]})
                last_key = value
            else:
                args_dict[last_key].append(value)
        return args_dict
    
    def dict_to_list(self, args_dict):
        #dictionary expected be formatted as follows {key: [values]}
        args_list = []
        for key  in args_dict.keys():
            args_list.append(key)
            if type(args_dict[key]) not in (list, tuple):
                args_dict[key] = [args_dict[key]]
            for value in args_dict[key]:
                args_list.append(str(value))
        return args_list
    
    def get_train_args(self, args):
        if type(args) == list:
            args = self.list_to_dict(args)
        args_all = dict([('--'+key,value) for key, value 
                         in TrainArgs()._get_class_dict().items()])
        args_all.update(self.list_to_dict(get_default_train_args()))
        if self.base_args is not None:
            for key in list(args_all.keys()):
                if (key[2:] in self.base_args.__dict__.keys() and
                   key[2:] not in ['loss_weighting']):
                    args_all.update({key: self.base_args.__dict__[key[2:]]})
            if args_all['--multi_branch_ffn'] is not None:
                del args_all['--ffn_hidden_size']
                del args_all['--ffn_num_layers']
                del args_all['--loss_weighting']
        args_all.update(args)
        for key in list(args_all.keys()):
            if str(args_all[key]) in ('None', 'False'):
                del args_all[key]
            elif str(args_all[key]) == 'True':
                args_all[key] = []

        return self.dict_to_list(args_all)
            
    def get_transfer_model(self, base_model_targets = [0]
                           , frzn_mpn_layers=3, frzn_ffn_layers='all'
                           ,args={}):
        if self.base_args.multi_branch_ffn is not None:
            transfer_model, model_args = self.get_branched_model(
                                              base_model_targets
                                            , frzn_mpn_layers
                                            , frzn_ffn_layers
                                            , args)
        else:
            transfer_model, model_args = self.get_basic_model(
                                              base_model_targets
                                            , frzn_mpn_layers
                                            , frzn_ffn_layers
                                            , args)
        
        return transfer_model, model_args
    
    def get_branched_model(self, base_model_targets, frzn_mpn_layers
                           , frzn_ffn_layers, args):
        
        if type(args) == list:
            args = self.list_to_dict(args)
        if '--multi_branch_ffn' not in args.keys():
            multi_branch_ffn = []
            shared_layers = self.base_args.ffn_num_layers[0]
            for i in range(shared_layers):
                multi_branch_ffn.append(self.base_args.ffn_hidden_size[i])
            for count, branch in enumerate(self.base_args.ffn_hidden_size[shared_layers:]):
                if count in base_model_targets:
                    multi_branch_ffn.append(branch)
            args.update({'--multi_branch_ffn':[str(tuple(multi_branch_ffn))]})
        args = self.get_train_args(args)
        model_args=TrainArgs().parse_args(args)
        transfer_model = MoleculeModel(args=model_args)
        
        base_model = self.base_model
        if frzn_ffn_layers == 'all':
            frzn_ffn_layers = self.base_args.ffn_num_layers
        elif frzn_ffn_layers == 'hidden':
            frzn_ffn_layers = list(self.base_args.ffn_num_layers)
            for i in range(len(frzn_ffn_layers)-1):
                frzn_ffn_layers[i+1] += -1
        
        for transfer_param, base_param in zip(transfer_model.encoder.parameters()
                                        , base_model.encoder.parameters()):
            transfer_param.data = base_param.cpu().detach()
            transfer_param.requires_grad = False
                    
        # Freeze weights and bias for given number of layers
        count = 0
        if base_model.shared_layers is not None:
            for transfer_param, base_param in zip(
                    transfer_model.shared_layers.parameters()
                    , base_model.shared_layers.parameters()
                                                ):
                if count < 2 * frzn_ffn_layers[0]:
                    transfer_param.data = base_param.cpu().detach()
                    transfer_param.requires_grad=False
                else:
                    break
                count +=1
        
        for t_branch, b_branch in enumerate(base_model_targets):
            count = 0
            for transfer_param, base_param in zip(
                    transfer_model.ffn[t_branch].parameters()
                    , base_model.ffn[b_branch].parameters()
                                                ):
                if count < 2 * frzn_ffn_layers[b_branch + 1]:
                    transfer_param.data = base_param.cpu().detach()
                    transfer_param.requires_grad=False
                else:
                    break
                count +=1
        
        return transfer_model, model_args
    
    def get_basic_model(self, base_model_targets, frzn_mpn_layers
                           , frzn_ffn_layers, args):
        
        args = self.get_train_args(args)
        model_args=TrainArgs().parse_args(args)
        transfer_model = MoleculeModel(args=model_args)
        
        base_hidden_layers = self.base_args.ffn_num_layers -1
        if frzn_ffn_layers == 'all':
            frzn_ffn_layers = self.base_args.ffn_num_layers
        elif frzn_ffn_layers == 'hidden':
            frzn_ffn_layers = base_hidden_layers
        base_model = self.base_model
        for transfer_param, base_param in zip(transfer_model.encoder.parameters()
                                        , base_model.encoder.parameters()):
            transfer_param.data = base_param.cpu().detach()
            transfer_param.requires_grad = False
                    
        frzn_hidden_layers = min(frzn_ffn_layers,base_hidden_layers)
        # Freeze weights and bias for given number of layers
        count = 0
        for transfer_param, base_param in zip(
                transfer_model.ffn.parameters()
                , base_model.ffn.parameters()
                                            ):
            if count < 2 * frzn_hidden_layers:
                transfer_param.data = base_param.cpu().detach()
                transfer_param.requires_grad=False
            elif frzn_ffn_layers == frzn_hidden_layers + 1:
                for i in base_model_targets:
                    transfer_param.data[i] = base_param.cpu().detach()[i]
                    #transfer_param.requires_grad=False
            else:
                break
            count +=1
        
        return transfer_model, model_args