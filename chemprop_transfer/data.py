# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:01:57 2021

@author: lansf
"""
import gc
import csv
import multiprocessing as mp
import os
import math
from chemprop.utils import makedirs
import pandas as pd
from chemprop.data.utils import get_data as get_chemprop_dataset
from chemprop.data.utils import get_header
from chemprop.utils import save_smiles_splits
from chemprop.data import MoleculeDataset
from chemprop.data.scaffold import scaffold_split
from random import Random
import numpy as np
from rdkit.Chem import AddHs
from rdkit.Chem import MolToSmiles
from rdkit.Chem import MolFromSmiles
from rdkit import DataStructs
from rdkit.Chem import AllChem
from typing import List
from tqdm import tqdm
from time import time
from .molecule_objects import Simple_Molecule

def get_similarity(smile_1, smile_2):
        mol1 = MolFromSmiles(smile_1)
        mol2 = MolFromSmiles(smile_2)
        radius = 3
        fp_1 = AllChem.GetMorganFingerprint(mol1, radius)
        fp_2 = AllChem.GetMorganFingerprint(mol2, radius)
        similarity = DataStructs.TanimotoSimilarity(fp_1, fp_2)
        return similarity

def split_data(data_path, split_tuple, num_folds=1, seed=0, save_dir=None
               , return_data=False, data_weights_path: str = None
               , features_path: List[str]=None
               , max_data_size=None, scaffold=False, balanced=True
               , test_file=None):
    if save_dir is None:
        save_dir = data_path
        
    data = get_chemprop_dataset(
        path=data_path,
        features_path=features_path,
        data_weights_path=data_weights_path,
        args=None,
        smiles_columns=None,
        logger=None,
        skip_none_targets=True,
        max_data_size = max_data_size
    )
    
    seed_list = [seed + i for i in range(num_folds)]
    if scaffold == False:
        random = Random(seed_list[0])
        indices = list(range(len(data)))
        random.shuffle(indices)
        train_size = int(split_tuple[0] * len(data))
        train_val_size = int((split_tuple[0] + split_tuple[1]) * len(data))
    
        if test_file is None:
            test = [data[i] for i in indices[train_val_size:]]
            test_path = data_path
        else:
            test_data = get_chemprop_dataset(
            path=test_file,
            features_path=features_path,
            data_weights_path=data_weights_path,
            args=None,
            smiles_columns=None,
            logger=None,
            skip_none_targets=True
            )
            test_path = test_file
            test = [t for t in test_data]
        data = MoleculeDataset([data[i] for i in indices[:train_val_size]])
    else:
        data, _, test_data = scaffold_split(data,
                                          (split_tuple[0] 
                                          + split_tuple[1]
                                          , 0, split_tuple[2])
                                          ,balanced=balanced,
                                          seed=seed_list[0])
        test_path = data_path                                  
        test = [t for t in test_data]
    
    
    save_smiles_splits(
            data_path=test_path,
            save_dir=save_dir,
            task_names=None,
            features_path=features_path,
            train_data=None,
            val_data=None,
            test_data=MoleculeDataset(test),
            smiles_columns=None
        )
    for i in range(num_folds):
        fold_dir = os.path.join(save_dir,'fold_'+str(i))
        makedirs(fold_dir)
        if scaffold == False:
            random = Random(seed_list[i])
            indices = list(range(len(data)))
            random.shuffle(indices)
            train = [data[i] for i in indices[:train_size]]
            val = [data[i] for i in indices[train_size:]]
        else:
            train_data, val_data, _ = scaffold_split(data,
                                          (split_tuple[0] / 
                                          (split_tuple[0] + split_tuple[1])
                                          , split_tuple[1] / 
                                          (split_tuple[0] + split_tuple[1]), 0)
                                          ,balanced=balanced,
                                          seed=seed_list[i])
            train = [t for t in train_data]
            val = [t for t in val_data]
        save_smiles_splits(
                data_path=data_path,
                save_dir=fold_dir,
                task_names=None,
                features_path=features_path,
                train_data=MoleculeDataset(train),
                val_data=MoleculeDataset(val),
                test_data=None,
                smiles_columns=None
            )

def combine_csv_files(list_of_files=[], out_file=r'./target_scaling.csv'):
    file_pd = pd.DataFrame()
    for file in list_of_files:
        file_pd_2 = pd.read_csv(file)
        file_pd = file_pd.append(file_pd_2)
    file_pd.to_csv(out_file, index=False)

def combine_files(combined_directory, dir_names=['dir1', 'dir2'], multiply=[10,1]):
    makedirs(combined_directory)
    fold_dirs = []
    for root, dirs, files in os.walk(dir_names[0]):
        for dir_name in dirs:
            if 'fold' in dir_name:
                fold_dirs.append(dir_name)
    if 'train_features.csv' in files:
        test_names = ['test_full.csv', 'test_smiles.csv', 'test_features.csv']
        train_val_names = ['train_full.csv', 'train_smiles.csv', 'train_features.csv'
                           ,'val_full.csv', 'val_smiles.csv', 'val_features.csv']
    else:
        test_names = ['test_full.csv', 'test_smiles.csv']
        train_val_names = ['train_full.csv', 'train_smiles.csv', 'val_full.csv'
                           , 'val_smiles.csv']
    
    for name in test_names:
        file_pd = pd.DataFrame()
        for i, dir_name in enumerate(dir_names):
            file_pd_2 = pd.read_csv(os.path.join(dir_name, name))
            #for _ in range(multiply[i]):
            #    file_pd = file_pd.append(file_pd_2)
            file_pd = file_pd.append(file_pd_2)
        file_pd.to_csv(os.path.join(combined_directory, name)
                             , index=False)
    
    for fold in fold_dirs:
        combined_sub = os.path.join(combined_directory, fold)
        makedirs(combined_sub)
        for name in train_val_names:
            file_pd = pd.DataFrame()
            for i, dir_name in enumerate(dir_names):
                sub = os.path.join(dir_name, fold)
                file_2_pd = pd.read_csv(os.path.join(sub, name))
                if 'train' in name:
                    for _ in range(multiply[i]):
                        file_pd = file_pd.append(file_2_pd)
                else:
                    file_pd = file_pd.append(file_2_pd)
            file_pd.to_csv(os.path.join(combined_sub, name)
                                 , index=False)

def normalize(original_path, new_path=None, return_data=False):
    df = pd.read_csv(original_path)
    # copy the dataframe
    df_norm = df.copy()
    # centering and scaling by std
    mean_list = []
    std_list = []
    for column in df_norm.columns[1:]:
        mean_list.append(df_norm[column].mean())
        std_list.append(df_norm[column].std())
        df_norm[column] = (df_norm[column] - mean_list[-1])/std_list[-1]
    if new_path is not None:
        df_norm.to_csv(new_path, index=False)
    if return_data == True:
        return df_norm, mean_list, std_list
    else:
        return None
    
def get_statistics(data):
    df_norm = data.copy()
    # centering and scaling by std
    mean_list = []
    std_list = []
    for column in df_norm.columns[1:]:
        mean_list.append(df_norm[column].mean())
        std_list.append(df_norm[column].std())
    return mean_list, std_list
    
def max_morgan_similarity(smiles_1: List[str], smiles_2: List[str]
                        , radius: int, sample_rate: list):
        """
        Determines the maximum smiliarity of smile in smiles_1 with those in smiles_2.
    
        :param smiles_1: A list of smiles strings.
        :param smiles_2: A list of smiles strings.
        :param radius: The radius of the morgan fingerprints.
        :param sample_rate: Rate at which to sample pairs of molecules for Morgan similarity (to reduce time).
        """
        
        # Compute similarities    
        # Sample to improve speed
        if sample_rate[0] < 1.0:
            sample_smiles_1 = np.random.choice(smiles_1
                , size=math.ceil(sample_rate[0]*len(smiles_1)), replace=False)
        else:
            sample_smiles_1 = smiles_1
        if sample_rate[1] < 1.0:
            sample_smiles_2 = np.random.choice(smiles_2
                , size=math.ceil(sample_rate[1]*len(smiles_2)), replace=False)
        else:
            sample_smiles_2 = smiles_2
        
        num_smiles = len(sample_smiles_1)
        max_similarities = np.zeros(num_smiles)
        similar_smiles = np.array(['smile2']*num_smiles)
        
        for index, smile_1 in enumerate(tqdm(sample_smiles_1, total=num_smiles)):
            mol_1 = MolFromSmiles(smile_1)
            fp_1 = AllChem.GetMorganFingerprint(mol_1, radius)
            for smile_2 in sample_smiles_2:
                mol_2 = MolFromSmiles(smile_2)
                fp_2 =  AllChem.GetMorganFingerprint(mol_2, radius)
                similarity = DataStructs.TanimotoSimilarity(fp_1, fp_2)
                if similarity > max_similarities[index]:
                    max_similarities[index] = similarity
                    similar_smiles[index] = smile_2
        return similar_smiles, max_similarities
            
def canonicalize(smiles, exclude=[None]):
    new_smiles = []
    invalid = []
    for index, smile in enumerate(smiles):
        if smile not in exclude:
            try:
                mol = MolFromSmiles(smile)
                new_smiles.append(MolToSmiles(mol))
            except:
                new_smiles.append('invalid')
                invalid.append(index)
        else:
            new_smiles.append(smile)
    return new_smiles, invalid

def get_duplicates(smiles):
    indices = []
    duplicates=[]
    for i in range(len(smiles)):
        if smiles[i] in smiles[0:i] or smiles[i] in smiles[i+1:]:
            indices.append(i),
            duplicates.append(smiles[i])
    return indices, duplicates

class MP_functions:
    def __init__(self, num_cpus):
        self.num_cpus = num_cpus
        
    def apply_function(self, function, file_reader, in_columns, out_file
                       ,out_columns, verbose=True):
        t0 = time()
        with open(out_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(out_columns)
        for index, data_chunk in enumerate(file_reader):
            pool = mp.Pool(self.num_cpus)
            matrix = pool.map(function, data_chunk[in_columns].values.tolist())
            pool.close()
            pool.join()
            with open(out_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for row in matrix:
                    if row[-1] == True:
                        writer.writerow(row[0:-1])
            if verbose == True:
                t1 = time()
                print('index: ' + str(index))
                print(t1 - t0)
                    
    def add_cutoff(self, data_path, file_reader, out_file
                       , criteria=['max'], sample_rate=1):
        if type(criteria) == str:
            criteria = [criteria]
        self.set_comparison_data(data_path)
        comparison_smiles = self.comparison_data.data['smiles'].to_list()
        radius=3
        mols = [MolFromSmiles(smile) for smile in comparison_smiles]
        self.fp_2_list =  np.array([AllChem.GetMorganFingerprint(mol, radius) for mol in mols])
        max_similarities = np.zeros(self.fp_2_list.size)
        sum_sim = np.zeros(self.fp_2_list.size)
        sum_variance = np.zeros(self.fp_2_list.size)
        t0 = time()
        num_data=0
        for index, data_chunk in enumerate(file_reader):
            if sample_rate < 1:
                data_chunk = np.random.choice(data_chunk['smiles'].values
                        , size=math.ceil(sample_rate*data_chunk['smiles'].values.size)
                        , replace=False).tolist()
            else:
                data_chunk = data_chunk['smiles'].values.tolist()
            pool = mp.Pool(self.num_cpus)
            matrix = pool.map(self.get_similarities, data_chunk)
            pool.close()
            pool.join()
            if 'max' in criteria:
                cutoff_max = np.array(matrix, dtype=float).max(axis=0)
                max_similarities = np.max((max_similarities,cutoff_max),axis=0)
            if 'average' in criteria:
                num_data += len(data_chunk)
                sum_sim += np.array(matrix, dtype=float).sum(axis=0)
                sum_variance += ( np.array(matrix, dtype=float).var(axis=0)
                                 * len(data_chunk) )             
            t1 = time()
            print('index: ' + str(index))
            print(t1 - t0)
        if 'max' in criteria:
            self.comparison_data.data['max_sim_cutoff'] = max_similarities
        if 'average' in criteria:
            self.comparison_data.data['avg_sim_cutoff'] = sum_sim/num_data
            self.comparison_data.data['std'] = (sum_variance/num_data)**0.5
        self.comparison_data.data.to_csv(out_file, sep=',', index=False)
        
    def get_similar_mols(self, data_path, file_reader, out_file, num_mol, rate
                         ,fast=True):
        self.set_comparison_data(data_path)
        comparison_smiles = self.comparison_data.data['smiles'].values
        radius=3
        mols = [MolFromSmiles(smile) for smile in comparison_smiles]
        self.fp_2_list =  np.array([AllChem.GetMorganFingerprint(mol, radius)
                                    for mol in mols])
        top_new_molecules = np.empty((2*num_mol, self.fp_2_list.size)
                                      , dtype=object)
        top_similarities = np.zeros((2*num_mol,self.fp_2_list.size)) - 1000
        
        t0 = time()
        for index, data_chunk in enumerate(file_reader):
            if rate < 1:
                sample_smiles = np.random.choice(data_chunk['smiles'].values
                        , size=math.ceil(rate*data_chunk['smiles'].values.size)
                        , replace=False)
                del data_chunk
            else:
                sample_smiles = data_chunk['smiles'].values
                del data_chunk
            pool = mp.Pool(self.num_cpus)
            similarities = np.array(pool.map(self.get_similarities
                                             , sample_smiles))
            pool.close()
            pool.join()
            order_list = similarities.argsort(axis=0)[::-1][0:num_mol]
            for i in range(self.fp_2_list.size):
                top_new_molecules[num_mol:,i] = sample_smiles[order_list[:,i]]
                top_similarities[num_mol:,i] = similarities[:,i][order_list[:,i]]
            order_list = top_similarities.argsort(axis=0)[::-1]
            for i in range(self.fp_2_list.size):
                top_new_molecules[:,i] = top_new_molecules[:,i][order_list[:,i]]
                top_similarities[:,i] = top_similarities[:,i][order_list[:,i]]
            t1 = time()
            print('index: ' + str(index))
            print(t1 - t0)
        del sample_smiles
        del similarities
        gc.collect()
        
        if fast == True:
            target_smiles_matrix = np.zeros_like(top_new_molecules,dtype=int)
            for i in range(self.fp_2_list.size):
                target_smiles_matrix[:,i] = i
            target_smiles_matrix = target_smiles_matrix.flatten()
            top_new_molecules = top_new_molecules.flatten()
            top_similarities = top_similarities.flatten()
            
            top_new_molecules, indices = np.unique(top_new_molecules
                                                   , return_index=True)
            top_similarities = top_similarities[indices]
            target_smiles_matrix = target_smiles_matrix[indices]
            target_smiles_matrix = comparison_smiles[target_smiles_matrix]
            
            with open(out_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['new_mol','match_mol', 'similarity'])
                for new_mol, match_mol, sim in zip(top_new_molecules
                                                   , target_smiles_matrix
                                                   , top_similarities):
                    writer.writerow([new_mol, match_mol, sim])
        else:
            file_2_write = np.empty((len(set(top_new_molecules[0:num_mol,:].flatten())), 5)
                                    ,dtype=object)
            iteration = 0
            for i in range(num_mol):
                for ii in range(self.fp_2_list.size):
                    if top_new_molecules[i,ii] not in file_2_write[:,0]:
                        file_2_write[iteration, 0] = top_new_molecules[i,ii]
                        file_2_write[iteration, 1] = comparison_smiles[ii]
                        file_2_write[iteration, 2] = top_similarities[i,ii]
                        file_2_write[iteration, 3] = comparison_smiles[ii]
                        file_2_write[iteration, 4] = top_similarities[i,ii]
                        iteration += 1
                    else:
                        index = np.where(file_2_write[:,0]==top_new_molecules[i,ii])
                        if top_similarities[i,ii] > file_2_write[index,4]:
                            file_2_write[index, 3] = comparison_smiles[ii]
                            file_2_write[index, 4] = top_similarities[i,ii]
                        
            
            
            np.savetxt(out_file, file_2_write, delimiter=","
                       , header="new_mol,match_mol,match_sim,max_mol,max_sim"
                       , comments='', fmt='%s')
        
    def set_filter_atoms(self,filter_atoms):
        self.filter_atoms = filter_atoms
        
    def set_comparison_data(self, data_path):
        data = DATASET(data_path)
        data.load_data()
        self.comparison_data = data
        
    def get_similarities(self, smile):
        fp_2_list = self.fp_2_list
        mol = MolFromSmiles(smile)
        radius = 3
        fp_1 = AllChem.GetMorganFingerprint(mol, radius)
        similarities = np.zeros(fp_2_list.size)
        for i in range(similarities.size):
            similarities[i] = DataStructs.TanimotoSimilarity(fp_1
                        , fp_2_list[i])
        return similarities
        
    def filter_and_canonicalize(self, smile_list):
        if type(smile_list) != list:
            smile_list = [smile_list]
        smile = smile_list[0] 
        mol = MolFromSmiles(smile)
        atom_types = set([mol.GetAtomWithIdx(i).GetSymbol() for i in 
                          range(mol.GetNumAtoms())])
        canon_smile = MolToSmiles(mol)
        if atom_types.issubset(self.filter_atoms):
            in_subset = True
        else:
            in_subset = False
        smile_list[0] = canon_smile
        smile_list += [in_subset]
        return tuple(smile_list)

class DATASET:
    """Class for loading an manipulating a dataset"""
    def __init__(self, file_path, sep=',', header='infer', usecols=None
                 , names=None, nrows=None, chunksize=None):
        """ 
        Parameters
        ----------
        GAS_PDOS : vasp_dos.VASP_DOS
            VASP_DOS object of gas phase calculation of adsorbate.
        """
        self.file_path = file_path
        self.chemprop_dataset = None
        self.data = None
        self.filename, self.filetype = os.path.splitext(file_path)
        self.sep = sep
        self.header = header
        self.usecols=usecols
        self.names=names
        self.nrows=nrows
        self.chunksize=chunksize
        
    def load_data(self):
        self.data = pd.read_csv(self.file_path, sep=self.sep, header=self.header
                                , usecols=self.usecols, names=self.names
                                , nrows=self.nrows, chunksize=self.chunksize)
        self.normalized = False
        
    def get_column_names(self):
        data = pd.read_csv(self.file_path, sep=self.sep, nrows=1)
        return data.columns.to_list()
        
    def load_new_data(self, filepath, sep=',', header='infer'):
            new_data = pd.read_csv(filepath, sep=sep, header=header)
            self.new_data = new_data
    def get_statistics(self):
        self.mean_list, self.std_list = get_statistics(self.data)
        
    def denormalize(self,file_path, sep=',', header='infer'):
        if self.data is None:
            self.load_data()
        mean_list, std_list = get_statistics(self.data)
        new_data = pd.read_csv(file_path, sep=sep, header=header)
        for index, column in enumerate(new_data.columns[1:]):
            new_data[column] = new_data[column] * std_list[index] + mean_list[index]
        filename, filetype = os.path.splitext(file_path)
        new_path = filename + '_denormalized' + filetype
        self.new_path = new_path
        new_data.to_csv(new_path,index=False)
        
    def save_data(self, data, alteration, data_type='pandas'):
        new_path = self.filename + '_' + alteration + self.filetype
        if data_type == 'numpy':
            np.savetxt(new_path, data, delimiter=self.sep)
        elif data_type == 'pandas':
            data.to_csv(new_path, sep=self.sep, index=False)
            
    def save(self, path, columns=None, rows=None):
        if rows is None:
            self.data.to_csv(path, columns=columns, sep=self.sep, index=False)
        else:
            new_data = self.data[0:rows]
            new_data.to_csv(path, columns=columns, sep=self.sep, index=False)
        
    def normalize(self, new_path=None):
        if self.data is None:
            self.data, self.mean_list, self.std_list = normalize(self.file_path
                                                             , new_path
                                                             , True)
        else:
            df_norm = self.data.copy()
            # centering and scaling by std
            mean_list = []
            std_list = []
            for column in df_norm.columns[1:]:
                mean_list.append(df_norm[column].mean())
                std_list.append(df_norm[column].std())
                df_norm[column] = (df_norm[column] - mean_list[-1])/std_list[-1]
            if new_path is not None:
                df_norm.to_csv(new_path, index=False)
        self.normalized = True
        self.data = df_norm
        self.mean_list = mean_list
        self.std_list = std_list
        
    def get_chemprop_dataset(self):
        self.chemprop_dataset = get_chemprop_dataset(self.file_path)
        
    def canonicalize_data(self,column=None):
        if column is None:
            if self.chemprop_dataset is None:
                self.get_chemprop_dataset()
            smiles = [[MolToSmiles(mol[0])] for mol in self.chemprop_dataset.mols()]
            targets = self.chemprop_dataset.targets()
            headers = get_header(self.file_path)
            self.data = pd.DataFrame(data=np.concatenate((smiles,targets),axis=1),    # values
                     columns=headers)
        else:
            smiles = [MolToSmiles(MolFromSmiles(smile)) for smile in self.data[column]]
            self.data[column] = smiles
        
    def remove(self, data, column='smiles'):
        in_category = []
        out_category = []
        for count, molecule in enumerate(self.data[column]):
            if molecule in data:
                in_category.append(count)
            else:
                out_category.append(count)
        self.removed_data = self.data.iloc[in_category]
        self.data = self.data.iloc[out_category]
        
    def get_similar_smiles(self, smiles_1, column, radius, sample_rate, min_similarity):
        smiles_2 = self.data[column].to_list()
        
        similar_smiles, similarities = max_morgan_similarity(smiles_1, smiles_2
                                                            , radius, sample_rate)
        indices = similarities >= min_similarity
        return pd.DataFrame(
                {'smiles_1': np.array(smiles_1)[indices],
                 'smiles_2': similar_smiles[indices],
                 'similarities': similarities[indices]
                        })
        
    def filter_mols(self, keep_atoms, min_size=2):
        molecules = self.get_simple_molecules('smiles')
        in_category = []
        out_category = []
        for count, molecule in enumerate(molecules):
            if ( set(molecule.get_atoms()).issubset(keep_atoms)
                and molecule.get_num_atoms().sum() >= min_size 
                and molecule.mol_str != 'C' ):
                in_category.append(count)
            else:
                out_category.append(count)
        self.data = self.data.iloc[in_category]        
        
    def get_num_heavy_atoms(self):
        if self.chemprop_dataset is None:
            self.get_chemprop_dataset()
        natoms = []
        for mol in self.chemprop_dataset.mols():
            natoms.append(mol[0].GetNumHeavyAtoms())
        return np.array(natoms)
    
    def get_num_heavy_bonds(self):
        if self.chemprop_dataset is None:
            self.get_chemprop_dataset()
        nbonds = []
        for mol in self.chemprop_dataset.mols():
            nbonds.append(mol[0].GetNumBonds())
        return np.array(nbonds)
    
    def get_num_H_atoms(self):
        if self.chemprop_dataset is None:
            self.get_chemprop_dataset()
        natoms = []
        for mol in self.chemprop_dataset.mols():
            mol_with_H = AddHs(mol[0])
            natoms.append(mol_with_H.GetNumAtoms() - mol_with_H.GetNumHeavyAtoms())
        return np.array(natoms)
    
    def get_num_H_bonds(self):
        if self.chemprop_dataset is None:
            self.get_chemprop_dataset()
        natoms = []
        for mol in self.chemprop_dataset.mols():
            mol_with_H = AddHs(mol[0])
            natoms.append(mol_with_H.GetNumBonds() - mol[0].GetNumBonds())
        return np.array(natoms)
    
    def get_simple_molecules(self, column_name):
        if self.data is None:
            self.load_data()
        
        molecule_list = [Simple_Molecule(molecule) for molecule in self.data[column_name]]
        return molecule_list
    
    @staticmethod
    def get_max_sim(fp_out_group, fp_in_group):
        max_sim = np.zeros(fp_out_group.size)
        for i, fp_out in enumerate(fp_out_group):
            similarities = np.zeros(fp_in_group.size)
            for j, fp_in in enumerate(fp_in_group):
                    similarities[j] = DataStructs.TanimotoSimilarity(fp_out, fp_in)
            max_sim[i] = np.max(similarities)
        return max_sim
    
    @staticmethod
    def get_max_sim_efficient(fp_out_group, prev_fp, prev_max):
        max_sim = np.zeros(fp_out_group.size)
        for i, fp_out in enumerate(fp_out_group):
            max_sim[i] = np.max([DataStructs.TanimotoSimilarity(fp_out, prev_fp), prev_max[i]])
        return max_sim
    
    @staticmethod
    def get_next_index(similarities, indices, in_group, self_sim):
        diff = similarities[in_group == False] - self_sim
        argmax = diff.argmax()
        next_index = indices[in_group == False][argmax]
        return next_index, diff[argmax], self_sim[argmax], argmax
    
    def order_dataset(self, out_file, new_molecules='new_mol'
                      , target_molecules=['match_mol', 'max_mol']
                      , sim_names = ['match_sim', 'max_sim']
                      , sim_func='sum'):
        radius=3
        if len(sim_names) >1 and sim_func == 'sum':
            similarities = self.data[sim_names].sum(axis=1).values
        elif len(sim_names) >1 and sim_func == 'average':
            similarities = self.data[sim_names].mean(axis=1).values
        else:
            similarities = self.data[sim_names].values.flatten()
        mol_1_list = [MolFromSmiles(smile) for smile in self.data[new_molecules]]
        fp_1_list =  np.array([AllChem.GetMorganFingerprint(mol_1, radius)
        for mol_1 in mol_1_list])
        
        in_group = np.array([False]*fp_1_list.size)
        #group_order = np.ones(fp_1_list.size,dtype=int)*(fp_1_list.size)
        #self_similarities = np.zeros(fp_1_list.size,dtype=float)
        #similarity_diff = np.zeros(fp_1_list.size,dtype=float)
        indices = np.arange(fp_1_list.size)
        
        first_index = similarities.argmax()
        in_group[first_index] = True
        #group_order[first_index] = 1
        #similarity_diff[first_index] = similarities[first_index]
        
        prev_index = first_index
        prev_max = self.get_max_sim(fp_1_list[in_group == False]
        , fp_1_list[in_group])
        t0 = time()
        
        
        smiles = self.data[new_molecules].values
        targets = self.data[target_molecules].values
        sim_vals = self.data[sim_names].values
        with open(out_file, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = [new_molecules, 'ext_sim', 'self_sim', 'difference'
                             ,'group_order']
            headers.extend(target_molecules)
            headers.extend(sim_names)
            writer.writerow(headers)
            row = [smiles[first_index], similarities[first_index], 0
                            , similarities[first_index], 1]
            row.extend(targets[first_index])
            row.extend(sim_vals[first_index])
            
            writer.writerow(row)
        
            for i in range(fp_1_list.size-1):
                print(i)
                max_sim = self.get_max_sim_efficient(fp_1_list[in_group == False]
                , fp_1_list[prev_index], prev_max)
                next_index, diff, self_sim, argmax = self.get_next_index(
                        similarities, indices, in_group, max_sim)
                in_group[next_index] = True
                #group_order[next_index] = i+2
                #similarity_diff[next_index] = diff
                #self_similarities[next_index] = self_sim
                prev_index = next_index
                prev_max = np.delete(max_sim, argmax)
                
                row = [smiles[next_index], similarities[next_index], self_sim
                       , diff, i+2]
                row.extend(targets[next_index])
                row.extend(sim_vals[next_index])
                writer.writerow(row)
                
                t1 = time()
                print(t1 - t0)
        
        """
        sort_order = np.argsort(group_order[in_group])
        
        similarity_df = pd.DataFrame({
        new_molecules
        : self.data[new_molecules].values[in_group][sort_order].astype(str),
        'ext_sim': similarities[in_group][sort_order],
        'self_sim': self_similarities[in_group][sort_order],
        'difference': similarity_diff[in_group][sort_order],
        'group_order': group_order[in_group][sort_order]
                })
        for target in target_molecules:
            similarity_df[target] = self.data[target].values[in_group][sort_order]
            
        for name in sim_names:
            similarity_df[name] = self.data[name].values[in_group][sort_order]
        
        similarity_df.to_csv(out_file, sep=',', index=False)
        """
