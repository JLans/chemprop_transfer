# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""
import os
import pandas as pd
import numpy as np
import chemprop
from sklearn.ensemble import GradientBoostingRegressor
from chemprop_transfer.error_metrics import get_r2
from chemprop_transfer.error_metrics import get_rmse

M_dirs = [r'./combined_10M+ani']
def get_features(data_file, model):
    input_data = pd.read_csv(data_file)
    smiles = [[smile] for smile in input_data['smiles']]
    features_MPN = model.fingerprint(smiles, fingerprint_type='MPN'
                                     )#.cpu().detach().numpy()
    features_shared = model.shared_layers(features_MPN)
    features_FFN = model.ffn[0][0:-1](features_shared
                                      ).cpu().detach().numpy()
    features_MPN = features_MPN.cpu().detach().numpy()
    features_shared = features_shared.cpu().detach().numpy()
    return (features_MPN, features_shared, features_FFN)

def get_h50(data_file):
    input_data = pd.read_csv(data_file)
    h50 = input_data['h50(exp)'].to_numpy()
    return h50

df_error = {'data_set':[], 'feature':[], 'r2':[], 'rmse':[]}
feature_list = ['MPN', 'shared', 'ffn']
data_list = ['train', 'val', 'test']
if __name__ == '__main__': # and '__file__' in globals()
    folder = r'./Mathieu_10+ani'
    test_path = os.path.join(folder, 'test_full.csv')
    fold_list = ['fold_' + str(i) for i in range(5)]
    for fold in fold_list:
        fold_folder = os.path.join(folder,fold)
        val_path = os.path.join(fold_folder, 'val_full.csv')
        data_path = os.path.join(fold_folder, 'train_full.csv')
        model = chemprop.utils.load_checkpoint(
            os.path.join(fold_folder,'fold_0/model_0/model.pt'))
        features1 = get_features(data_path, model)
        h501 = get_h50(data_path)
        features2 = get_features(val_path, model)
        h502 = get_h50(val_path)
        features = []
        for i in range(3):
            features.append(np.concatenate((features1[i], features2[i])))
        h50 = np.concatenate((h501, h502))
        GBDT_models = []
        for feature in features:
            GBDT = GradientBoostingRegressor(random_state=1
                , learning_rate=0.004, n_estimators=3000, min_samples_split=7
                , max_depth=2, max_features='sqrt')
            GBDT.fit(feature, h50)
            GBDT_models.append(GBDT)
        for count, input_file in enumerate([data_path, val_path, test_path]):
            features = get_features(input_file, model)
            h50 = get_h50(input_file)
            for count2, feature in enumerate(features):
                pred = GBDT_models[count2].predict(feature)
                df_error['r2'].append(get_r2(h50,pred))
                df_error['rmse'].append(get_rmse(h50,pred))
                df_error['data_set'].append(data_list[count])
                df_error['feature'].append(feature_list[count2])
df_error = pd.DataFrame(df_error)
group_by_columns = ['data_set', 'feature']
print(df_error.groupby(group_by_columns).mean())
print(df_error.groupby(group_by_columns).std(ddof=1)/len(fold_list)**0.5)
                
                    
                