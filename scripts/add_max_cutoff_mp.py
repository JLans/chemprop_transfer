import multiprocessing as mp
from chemprop_transfer.data import DATASET
from chemprop_transfer.data import MP_functions


if __name__ == '__main__':
    num_cpus = mp.cpu_count() - 2
    mp_func = MP_functions(num_cpus)
    
    comparison_data_path = r'./log_Mathieu_2020.csv'
    data_path = './ox_bal_big_db_chnof.csv'
    data = DATASET(data_path, chunksize=400000)
    data.load_data()
    mp_func.add_cutoff(comparison_data_path
                           , data.data, './log_Mathieu_2020_maxsim.csv'
                           , criteria=['max'], sample_rate=0.001)