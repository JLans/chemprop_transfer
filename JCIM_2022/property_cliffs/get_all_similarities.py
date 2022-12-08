"""Computes the similarity of molecular scaffolds between two datasets."""
import multiprocessing as mp
from chemprop_transfer.data import DATASET
from rdkit.Chem import MolFromSmiles
from rdkit import DataStructs
from rdkit.Chem import AllChem
from chemprop_transfer.data import MP_functions
from chemprop_transfer.error_metrics import get_rmse
import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

data_path_2 = r'../../chemprop_transfer/data/log_Mathieu_2020_CHNOFCl.csv'
data_2 = DATASET(data_path_2)
data_2.load_data()
smiles_2 = data_2.data['smiles'].to_list()
smiles = [[smile] for smile in smiles_2]
logh50_2 = data_2.data['log10h50(exp)']

radius=2
bits = 2048
mol_2_list = [MolFromSmiles(smile_2) for smile_2 in smiles_2]
fp_2_list =  [AllChem.GetHashedMorganFingerprint(mol_2, radius, nBits=bits) for mol_2 in mol_2_list]

in_files = [data_path_2]
out_files = ['./Mathieu_self_similar.csv']

mean_preds = np.loadtxt('./preds.csv', delimiter=',', dtype=float)
mean_params = np.loadtxt('./params.csv', delimiter=',', dtype=float)
mean_last = np.loadtxt('./last.csv', delimiter=',', dtype=float)

pca_model_params = decomposition.PCA(n_components=50)
pca_params = pca_model_params.fit_transform(mean_params)
x_std = StandardScaler().fit_transform(mean_params)
pca_params_norm = pca_model_params.fit_transform(x_std)
pca_model_last = decomposition.PCA(n_components=50)
pca_last = pca_model_last.fit_transform(mean_last)
x_std = StandardScaler().fit_transform(mean_last)
pca_last_norm = pca_model_last.fit_transform(x_std)
EVR_params = pca_model_params.explained_variance_ratio_
EVR_last = pca_model_last.explained_variance_ratio_

def get_all_similarities(data):
    smile_1 = data[0]
    index_1 = smiles_2.index(smile_1)
    preds_1 = mean_preds[index_1]
    params_1 = mean_params[index_1]
    last_1 = mean_last[index_1]
    pca_params_1 = pca_params[index_1]
    pca_last_1 = pca_last[index_1]
    pca_params_norm_1 = pca_params_norm[index_1]
    pca_last_norm_1 = pca_last_norm[index_1]
    logh50_1 = data[1]
    mol_1 = MolFromSmiles(smile_1)
    fp_1 = AllChem.GetHashedMorganFingerprint(mol_1, radius, nBits=bits)
    similarities = []
    for i, fp_2 in enumerate(fp_2_list):
        if smile_1 != smiles_2[i]:
            preds_2 = mean_preds[i]
            params_2 = mean_params[i]
            last_2 = mean_last[i]
            pca_params_2 = pca_params[i]
            pca_last_2 = pca_last[i]
            pca_params_norm_2 = pca_params_norm[i]
            pca_last_norm_2 = pca_last_norm[i]
            similarity = DataStructs.TanimotoSimilarity(fp_1, fp_2)
            similarities.append([smile_1, smiles_2[i], similarity
                                 , get_rmse(params_1, params_2)
                                 , get_rmse(last_1, last_2)
                                 , get_rmse(pca_params_1, pca_params_2)
                                 , get_rmse(pca_last_1, pca_last_2)
                                 , get_rmse(pca_params_norm_1, pca_params_norm_2)
                                 , get_rmse(pca_last_norm_1, pca_last_norm_2)
                                 , get_rmse(pca_params_norm_1*EVR_params
                                            , pca_params_norm_2*EVR_params)
                                 , get_rmse(pca_last_norm_1*EVR_last
                                            , pca_last_norm_2*EVR_last)
                                 , logh50_1
                                 , logh50_2[i], logh50_2[i] - logh50_1
                                 , preds_1, preds_2, preds_2 - preds_1, True])
    return similarities

if __name__ == '__main__':
    data_path_1 = in_files[0]
    data_chunk = DATASET(data_path_1, chunksize=60)
    data_chunk.load_data()
    num_cpus = mp.cpu_count() - 2
    print('num cpus ' + str(num_cpus))
    mp_func = MP_functions(num_cpus)
    in_columns=['smiles', 'log10h50(exp)']
    matrix=[None]
    mp_func.apply_function(get_all_similarities, data_chunk.data
                           , in_columns=in_columns, out_file=out_files[0]
                           ,out_columns=['smiles_1', 'smiles_2', 'similarities_Morgan'
                                         , 'L2_MPN', 'L2_last', 'pca_MPN', 'pca_last'
                                         , 'pca_norm_MPN', 'pca_norm_last'
                                         , 'pca_norm_MPN_EVR', 'pca_norm_last_EVR'
                                         , 'logh50_1', 'logh50_2', 'logh50_diff'
                                         , 'pred_1', 'pred_2', 'pred_diff']
                           , verbose=True)
    

