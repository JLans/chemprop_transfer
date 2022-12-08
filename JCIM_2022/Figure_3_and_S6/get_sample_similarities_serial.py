"""Computes the similarity of molecular scaffolds between two datasets."""
import numpy as np
from chemprop_transfer.data import DATASET
from rdkit.Chem import MolFromSmiles
from rdkit import DataStructs
from rdkit.Chem import AllChem
from time import time

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
            similarities.append([smile_1, smiles_2[i], similarity])
    return similarities

if __name__ == '__main__':
    data_path_1 = r'../../chemprop_transfer/data/log_Mathieu_2020_CHNOFCl.csv'
    t0 = time()
    data_1 = DATASET(data_path_1)#, chunksize=6000)
    data_1.load_data()
    similarity_list = []
    smiles_1 = data_1.data['smiles'].to_numpy()
    args = data_1.data['similarities'].argsort()[::-1][0:30000].to_list()
    sample_smiles_1 = np.random.choice(smiles_1[args], size=12000, replace=False)
    for smile_1 in sample_smiles_1:
        if smile_1 not in smiles_2:
            similarity_list += get_all_similarities(smile_1)
        
    similarity_list = np.array(similarity_list)
    similarity_list
    t1 = time()
    print(t1 - t0)
    np.savetxt('../Mathieu_self_similar.csv', similarity_list, delimiter=","
               , header="smiles_1,smiles_2,similarities", comments='', fmt='%s')
