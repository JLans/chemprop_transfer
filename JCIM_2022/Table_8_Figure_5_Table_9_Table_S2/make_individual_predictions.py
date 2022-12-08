# -*- coding: utf-8 -*-
import os
from pandas import read_csv
import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import MolToSmiles
#from rdkit.Chem import Draw
#from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.useBWAtomPalette()
from IPython.display import SVG
import chemprop

def get_files(directory, name):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename == name:
                matches.append(os.path.join(root,filename))
    return matches

train_files = get_files('./best_model','train_full.csv')
model_files = get_files('./best_model','model.pt')

def canonacalize(smiles):
    if type(smiles) == str:
        new_smiles = MolToSmiles(MolFromSmiles(smiles))
    else:
        new_smiles = [MolToSmiles(MolFromSmiles(smile)) for smile in smiles]
    return new_smiles

example_data = read_csv('./Table_9_SMILES.csv')
example_smiles = canonacalize(example_data['smiles'].to_list())
predictions_list = []
for count, smile in enumerate(example_smiles):
    print(count)
    predictions_list.append([])
    for count_2 , file in enumerate(train_files):
        training_data = read_csv(file)
        smiles = training_data['smiles'][
            np.isnan(training_data['log10h50(exp)'])==False]
        smiles = canonacalize(smiles.to_list())
        if smile not in smiles:
            arguments = [
             '--individual_ensemble_predictions',
             '--num_workers', '0',
             '--test_path', '/dev/null',
             '--preds_path', '/dev/null',
             '--checkpoint_path', model_files[count_2]]
            args = chemprop.args.PredictArgs().parse_args(arguments)
            preds = chemprop.train.make_predictions(args=args, smiles=[[smile]])
            predictions_list[count].append(10**preds[0][0])

#SE = np.std(test_value, axis=0, ddof=1)
means = []
STDs = []
for predictions in predictions_list:
    means.append(np.mean(predictions))
    STDs.append(np.std(predictions,ddof=1) / len(predictions)**0.5)
    
idx=13
d2d = rdMolDraw2D.MolDraw2DSVG(800,600)
d2d.drawOptions().updateAtomPalette({k: (0, 0, 0) for k in DrawingOptions.elemDict.keys()})
d2d.DrawMolecule(MolFromSmiles(example_smiles[idx]))
#d2d.DrawMolecule(MolFromSmiles(smile))
d2d.FinishDrawing()
SVG(d2d.GetDrawingText())