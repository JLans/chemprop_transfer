"""filters on only CHNO molecules"""
import multiprocessing as mp
from chemprop_transfer.data import DATASET
from chemprop_transfer.data import MP_functions


if __name__ == '__main__':
    num_cpus = mp.cpu_count() - 2
    mp_func = MP_functions(num_cpus)
    
    comparison_data_path = r'../data/Mathieu_2020_CHNO.csv'
    data_path = '../../large_data.csv'
    data = DATASET(data_path, chunksize=5000)
    data.load_data()
    mp_func.get_similar_mols(comparison_data_path, data.data
                             , '../../similar_mols.csv', 50)
    
    