# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""
from chemprop_transfer.data import split_data
seed=1
num_folds=5
temp_dir = './temp'
val = 0.1
test = .29333   
split_data('./log_Mathieu_2020.csv' , (1-val-test, val, test), seed=seed, num_folds=num_folds
           , save_dir='./Mathieu')
split_data('./ani_energy_5000r.csv', (1-val-test, val, test), seed=seed, num_folds=num_folds
           , save_dir='./ani_energies')
