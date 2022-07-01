"""filters on only CHNO molecules"""
import multiprocessing as mp
from chemprop_transfer.data import DATASET
from chemprop_transfer.data import MP_functions


if __name__ == '__main__':
    num_cpus = mp.cpu_count() - 2
    mp_func = MP_functions(2)
    mp_func.set_filter_atoms(['C', 'H', 'N', 'O'])
    
    data_path = r'../data/Mathieu_2020.csv'
    data = DATASET(data_path, chunksize=155)
    data.load_data()
    names = data.get_column_names()
    mp_func.apply_function(mp_func.filter_and_canonicalize, data.data
                           ,'../data/Mathieu_2020_CHNO.csv', column_names=names)
    
    