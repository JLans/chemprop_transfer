"""filters on only CHNO molecules"""
from chemprop_transfer.data import DATASET


"""
data_path = r'../old_data/similar_mols.csv'
out_file = r'../sorted_molecules.csv'
data = DATASET(data_path)
data.load_data()
data.order_dataset(out_file, new_molecules='new_mol'
                      , target_molecules=['match_mol', 'max_mol']
                      , sim_names = ['match_sim', 'max_sim']
                      , sim_func='sum')
"""

data_path = r'../old_data/similar_molecules_max_cutoff.csv'
out_file = r'../sorted_molecules.csv'
data = DATASET(data_path)
data.load_data()
data.order_dataset(out_file, new_molecules='smiles'
                      , target_molecules=['Mathieu']
                      , sim_names = ['similarity'])

    
    