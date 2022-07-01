# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:45:56 2021

@author: joshua.l.lansford
"""
import multiprocessing as mp
from chemprop_transfer.data import MP_functions
import torchani
from chemprop_transfer.data import DATASET
from chemprop_transfer.property_generator import PROPERTY_GENERATOR


model = torchani.models.ANI1ccx(periodic_table_index=True).double()
PG = PROPERTY_GENERATOR(model)

if __name__ == '__main__':
    out_file = './Mathieu_energies.csv'
    out_columns = ['smiles', 'energy', 'fmax', 'SYM', 'MOI1', 'MOI2', 'MOI3'
                     , 'Hvib75', 'Hvib150', 'Hvib300', 'Hvib600', 'Hvib1200'
                     , 'TSvib75', 'TSvib150', 'TSvib300', 'TSvib600', 'TSvib1200'
                     , 'FC_1', 'FC_2', 'FC_3']
    in_column = 'smiles'
    #data_path = './sorted_molecules.csv'
    data_path = r'../../data/Mathieu_2020.csv'
    data = DATASET(data_path, chunksize=10)
    data.load_data()
    num_cpus = mp.cpu_count() - 2
    mp_func = MP_functions(2)
    mp_func.apply_function(PG.get_properties,data.data, in_column, out_file
                       ,out_columns, verbose=True)
