"""Computes the similarity of molecular scaffolds between two datasets."""
import chemprop
import numpy as np
from chemprop_transfer.data import DATASET
import os
from torch import tensor

def get_files(directory, name):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename == name:
                matches.append(os.path.join(root,filename))
    return matches

model_files = get_files('../Table_8_Figure_5_Table_9_Table_S2/seed_1','model.pt')

data_path_2 = r'../../chemprop_transfer/data/log_Mathieu_2020_CHNOFCl.csv'
data_2 = DATASET(data_path_2)
data_2.load_data()
smiles_2 = data_2.data['smiles'].to_list()
smiles = [[smile] for smile in smiles_2]

if __name__ == '__main__':
    predictions_list = []
    parameters_list = []
    last_FFN_list = []
    for count, smile in enumerate(model_files):
        print(count)
        arguments = ['--no_features_scaling',
         '--num_workers', '0',
         '--test_path', '/dev/null',
         '--preds_path', '/dev/null',
         '--checkpoint_path', model_files[count]]
        args = chemprop.args.PredictArgs().parse_args(arguments)
        preds = chemprop.train.make_predictions(args=args
                                                , smiles=smiles)
        predictions_list.append([pred[0] for pred in preds])
        model_objects = chemprop.train.load_model(args=args)
        parameters = model_objects[2][0].fingerprint(batch=smiles,
                          fingerprint_type = 'MPN').detach().cpu().numpy()
        parameters_list.append(parameters)
        last_FFN = model_objects[2][0].ffn[0][0:-1](tensor(parameters).to('cuda')).detach().cpu().numpy()
        last_FFN_list.append(last_FFN)
    mean_preds = np.mean(predictions_list,axis=0)
    mean_params = np.mean(parameters_list,axis=0)
    mean_last = np.mean(last_FFN_list, axis=0)
    np.savetxt('./preds.csv', mean_preds, delimiter=',')
    np.savetxt('./params.csv', mean_params, delimiter=',')
    np.savetxt('./last.csv', mean_last, delimiter=',')