# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:58:37 2022

@author: joshua.l.lansford
"""
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from chemprop_transfer.plotting_tools import set_figure_settings
from sklearn.decomposition import PCA
set_figure_settings('paper')

def getXbonds(mol, patt):
    """Return set of bonds in 'mol' matching a given explosophore 'patt'

    Arguments:
    mol  -- molecule as RDKit object
    patt -- explosophore as RDKit molecular substructure
    Return:
    set of substructures
    """
    if type(patt) == list:
        list_of_tuples = []
        for p in patt:
            list_of_tuples += mol.GetSubstructMatches(p)
    else:
        list_of_tuples = mol.GetSubstructMatches(patt)
    #bonds = set([tuple(sorted(t[:2])) for t in list_of_tuples])
    #return len(bonds)
    return len(list_of_tuples)

def dicts_to_list(dicts):
    expl_list = []
    for dic in dicts:
        if len(dic) == 0:
            dic['none'] = 1
        expl_list += list(dic.keys())
    return expl_list

NNITRO = "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"
n_NNITRO = "[!$([NX3](=O)=O)&!$([NX3+](=O)[O-])]"
Explosophores = (
    # Nitroaromatic
    ("nitroaromatic", "%sc" % NNITRO),
    # Nitroalkane: C-NO2
    ("nitroalkane", "%s[C]" % NNITRO),
    # Nitramine N-NO2
    ("nitramine", "%s[#7]" % NNITRO),
    # Nitric ester O-NO2
    ("nitric ester", "%s[#8X2]" % NNITRO),
    #N2loss (azide)
    ( "other N2 loss", ["[#6,#8]~[#7X2]~[#7X2]~[#6,#8]"
                        ,"[#6]~[#7X3]([!n])~[#7X2]~[#6,#8]"
                        ,"[#6]~[#7X3]([!n])~[#7X3]([!n])~[#6]"
                        ,"[#6,#8]~[#7X2]~[#7X1,#7X2&H1,#7X3&H2]"
                        ,"[#6,#8]~[#7X2]~[#7X2]~[#7X1,#7X2&H1,#7X3&H2]"]),
    # trinitro aromatic
    ("triazole", "[n]~[n]~[n]"),
    ("aromatic nitrogen", "[c]~[$([n](~[!#7])),$([nX2])]~[c]"),
    # furazan
    ("furazan", "[n]~[o]~[n]"),
    # Fluorine atom
    #("F", "[#9]"),
    # Chlorine atom
    #("Cl", "[#17]")
)

# initialize global variables (ExploCodes, Bde, Pattern, Formation_enthalpies)
Pattern = dict()
for explocode, smart in Explosophores:
    if type(smart) == list:
        Pattern[explocode] = [Chem.MolFromSmarts(s) for s in smart]
    else:
        Pattern[explocode] = Chem.MolFromSmarts(smart)

def get_explosophores(mol):
    """Return occurrences of all explosophores found in molecule

    Arguments:
    mol  -- molecule as RDKit object
    """
    expl_dict = dict()
    for expl_code in Pattern.keys():
        nbonds = getXbonds(mol, Pattern[expl_code])
        if nbonds > 0:
            expl_dict[expl_code] = nbonds
    if len(expl_dict) == 0:
        expl_dict['none'] = True
    return expl_dict

def get_expl_dicts(df):
    expl_dicts = []
    for smile in df['smiles']:
        mol = Chem.MolFromSmiles(smile)
        expl_dicts.append(get_explosophores(mol))
    return expl_dicts

expl_groups = list(Pattern.keys()) + ['none']
group_names = ['nitro-\naromatic', 'nitro-\nalkane', 'nitramine'
               , 'nitric\nester', 'other\nN$_{2}$ loss', '123-\ntriazole'
               , 'aromatic\nnitrogen', 'furazan', 'none']


df_train = pd.read_csv(r'../Table_5_and_6\Direct\val_0.10_v2\seed_1\fold_0\train_full.csv')
expl_dicts = get_expl_dicts(df_train)
expl_list = dicts_to_list(expl_dicts)
expl_counts_train = []
for group in expl_groups:
    expl_counts_train.append(len([item for item in expl_list if item == group]))
expl_counts_train = np.array(expl_counts_train) / len(expl_dicts)
df_val = pd.read_csv(r'../Table_5_and_6\Direct\val_0.10_v2\seed_1\fold_0\val_full.csv')
expl_dicts = get_expl_dicts(df_val)
expl_list = dicts_to_list(expl_dicts)
expl_counts_val = []
for group in expl_groups:
    expl_counts_val.append(len([item for item in expl_list if item == group]))
expl_counts_val = np.array(expl_counts_val) / len(expl_dicts)
df_test = pd.read_csv(r'../Table_5_and_6\Direct\val_0.10_v2\seed_1\test_full.csv')
expl_dicts = get_expl_dicts(df_test)
expl_list = dicts_to_list(expl_dicts)
expl_counts_test = []
for group in expl_groups:
    expl_counts_test.append(len([item for item in expl_list if item == group]))
expl_counts_test = np.array(expl_counts_test) / len(expl_dicts)
plt.figure(figsize=(7.2,3.5), dpi=400)
plt.bar(np.arange(9) - 0.2, expl_counts_train, width=0.2, align='center')
plt.bar(np.arange(9), expl_counts_val, width=0.2, align='center')
plt.bar(np.arange(9) + 0.2, expl_counts_test, width=0.2, align='center')
plt.legend(['train', 'validation', 'test'])
plt.xticks(np.arange(9), group_names)
plt.ylabel('Fraction of molecules')
plt.xlabel('Chemical family')
plt.tight_layout()
plt.savefig('Mathieu_splits.jpg')
plt.close()

df_Mathieu = pd.read_csv('../../chemprop_transfer/data/Mathieu_2020.csv')
expl_dicts = get_expl_dicts(df_Mathieu)
expl_list = dicts_to_list(expl_dicts)
expl_counts_Mathieu = []
for group in expl_groups:
    expl_counts_Mathieu.append(len([item for item in expl_list if item == group]))
expl_counts_Mathieu = np.array(expl_counts_Mathieu) / len(expl_dicts)
df_CSBB = pd.read_csv('../../chemprop_transfer/data/Casey_DFT_data.csv', skiprows=7)
expl_dicts = get_expl_dicts(df_CSBB)
expl_list = dicts_to_list(expl_dicts)
expl_counts_CSBB = []
for group in expl_groups:
    expl_counts_CSBB.append(len([item for item in expl_list if item == group]))
expl_counts_CSBB = np.array(expl_counts_CSBB) / len(expl_dicts)
df_LBRJ = pd.read_csv('../../chemprop_transfer/data/ani_properties_filtered_and_normalized.csv'
                 ,nrows=5000)
expl_dicts = get_expl_dicts(df_LBRJ)
expl_list = dicts_to_list(expl_dicts)
expl_counts_LBRJ = []
for group in expl_groups:
    expl_counts_LBRJ.append(len([item for item in expl_list if item == group]))
expl_counts_LBRJ = np.array(expl_counts_LBRJ) / len(expl_dicts)
plt.figure(figsize=(7.2,3.5), dpi=400)
plt.bar(np.arange(9) - 0.2, expl_counts_Mathieu, width=0.2
        , align='center', color='tab:red')
plt.bar(np.arange(9), expl_counts_CSBB, width=0.2
        , align='center', color='tab:orange')
plt.bar(np.arange(9) + 0.2, expl_counts_LBRJ, width=0.2
        , align='center', color='tab:blue')
plt.legend(['Mathieu dataset', 'CSBB dataset', 'LBRJ dataset'])
plt.xticks(np.arange(9), group_names)
plt.ylabel('Fraction of molecules')
plt.xlabel('Chemical family')
plt.tight_layout()
plt.savefig('data_set_compare.jpg')
plt.close()

df_Mathieu = pd.read_csv('../../chemprop_transfer/data/Mathieu_2020.csv')
expl_dicts = get_expl_dicts(df_Mathieu)
data_matrix = np.zeros((len(expl_dicts), len(Pattern.keys())))
for count, key in enumerate(Pattern.keys()):
    for count2, dic in enumerate(expl_dicts):
        if key in dic.keys():
            data_matrix[count2][count] = 1
            
corr_matrix = np.corrcoef(data_matrix.T)
np.savetxt('./correlation.csv', corr_matrix, delimiter=",")

mols = []
for count, dic in enumerate(expl_dicts):
    mols.append(Chem.MolFromSmiles(df_Mathieu['smiles'].iloc[count]))
    if 'nitroalkane' in dic.keys():
        print(count)
        print(dic)
        