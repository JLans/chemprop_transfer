"""Computes the similarity of molecular scaffolds between two datasets."""
import multiprocessing as mp
from chemprop_transfer.data import DATASET
from rdkit.Chem import MolFromSmiles
from rdkit import DataStructs
from rdkit.Chem import AllChem
from time import time
from chemprop_transfer.data import MP_functions

data_path_2 = r'../../chemprop_transfer/data/log_Mathieu_2020_CHNOFCl.csv'
data_2 = DATASET(data_path_2,usecols=[0])
data_2.load_data()
smiles_2 = data_2.data['smiles'].to_list()
radius=3
mol_2_list = [MolFromSmiles(smile_2) for smile_2 in smiles_2]
fp_2_list =  [AllChem.GetMorganFingerprint(mol_2, radius) for mol_2 in mol_2_list]

def get_all_similarities(smile_1):
    mol_1 = MolFromSmiles(smile_1)
    fp_1 = AllChem.GetMorganFingerprint(mol_1, radius)
    similarities = []
    for i, fp_2 in enumerate(fp_2_list):
        if smile_1 != smiles_2[i]:
            similarity = DataStructs.TanimotoSimilarity(fp_1, fp_2)
            similarities.append([smile_1, smiles_2[i], similarity, True])
    return similarities

in_files = [data_path_2
            ,'../../chemprop_transfer/data/Casey_DFT_data.csv'
            ,'../../chemprop_transfer/data/ani_properties_sorted.csv']
out_files = ['./Mathieu_self_similar.csv'
             ,'./Mathieu_Casey_similar.csv'
             ,'./Mathieu_PNNL_similar.csv']

if __name__ == '__main__':
    for count in range(3):
        data_path_1 = in_files[count]
        if count == 0:
            data_chunk = DATASET(data_path_1, chunksize=200)
            frac = None
        else:
            data_chunk = DATASET(data_path_1, chunksize=5000)
            frac=0.03
        if count == 1:
            data_chunk.skiprows = 7
        data_chunk.load_data()
        num_cpus = mp.cpu_count() - 2
        print('num cpus ' + str(num_cpus))
        mp_func = MP_functions(num_cpus)
        in_columns='smiles'
        matrix=[None]
        mp_func.apply_function(get_all_similarities, data_chunk.data
                               , in_columns='smiles', out_file=out_files[count]
                               ,out_columns=['smiles_1', 'smiles_2', 'similarities']
                               , verbose=True, frac=frac)