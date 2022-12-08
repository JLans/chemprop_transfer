# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""
import os
from chemprop_transfer.data import DATASET
from chemprop_transfer.utils import get_predictions
import matplotlib.pyplot as plt
from chemprop_transfer.error_metrics import get_r2, get_rmse
import numpy as np
from chemprop.utils import load_checkpoint
from scipy import stats
for save_dir in ['./Mathieu', './Mathieu_frzn_MPN']:
    n_models = 5
    t_score = stats.t.ppf(1-0.025, n_models-1)
    predictions = get_predictions(save_dir, n_models, use_predictions_file=True) 
    target_name = 'h50(exp)' #'log(h50 cm)'
    Mathieu_path = os.path.join(save_dir, 'test_full.csv')
    model_path = os.path.join(save_dir,'fold_0/fold_0/model_0/model.pt')
    model = load_checkpoint(model_path)
    
    hobs = DATASET(Mathieu_path)
    hobs.load_data()
    
    pred_list = []
    np_list = []
    
    
    for i in range(n_models):
        pred_path = os.path.join(save_dir,'test_predictions_'+str(i)+'.csv')
        h50_dataset = DATASET(pred_path) 
        pred_list.append(h50_dataset)
        pred_list[-1].load_data()
        np_list.append(pred_list[-1].data[target_name])
        
    np_list = np.array(np_list)    
    
    
    
    
    """"
    plt.figure()
    plt.plot([np.min(hobs.data[target_name]),
              np.max(hobs.data[target_name])],
             [np.min(hobs.data[target_name]),
              np.max(hobs.data[target_name])],'k-')
    plt.errorbar(hobs.data[target_name], predictions, linestyle=''
                ,yerr=t_score*np_list.std(axis=0,ddof=1), barsabove=True, capthick=1, capsize=2, ecolor='black')
    plt.plot(hobs.data[target_name], predictions,'o',color='royalblue')
    plt.xlabel('Observed log(H$_{50\%}$)')
    plt.ylabel('Predicted log(H$_{50\%}$)')
    #plt.xlabel('Solid phase heat of formation (kcal/mol)')
    #plt.ylabel('Predicted heat of formation (kcal/mol)')
    plt.show()
    """
    r2 = get_r2(hobs.data[target_name], predictions)
    rmse = get_rmse(hobs.data[target_name], predictions)
    print ('save_dir: ' + save_dir)
    print(r2)
    print(rmse)
    
    r2_list = np.zeros(n_models)
    rmse_list = np.zeros(n_models)
    for i in range(n_models):
        r2_list[i] = get_r2(hobs.data[target_name], np_list[i])
        rmse_list[i] = get_rmse(hobs.data[target_name], np_list[i])
    print(r2_list.mean())
    print(r2_list.std(ddof=1)/n_models**0.5)
    print(rmse_list.mean())
    print(rmse_list.std(ddof=1)/n_models**0.5)