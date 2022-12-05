# The Chemprop Transfer-Learning Toolkit (CTL)

CTL is a toolkit for transfer-learning with the Chemprop message-passing neural network model builder via the package chemprop_transfer.

## Table of Contents
[Background and description](#bckgrd_dscrpts)

[Install package](#install_package)

[Examples](#examples)

* [Filtering](#filtering)

* [Sort molecules](#sort)

* [Compute ANI properties](#ani)

* [Co-train a base model](#run_base_model)

* [Transfer model parameters](#transfer_model)

[Credits](#credits)

[License](#license)

## <a name="bckgrd_dscrpts"/></a>Background and description
Many chemical datasets, particularly experimental datsets, are small. Often limited to hundreds or even just dozens of datapoints. To enable property prediction using Chemprop with these types of datasets, an emerging methadology is to apply transfer-learning techniques. In the journal publication this software accompanies, take an approach of co-training the experimental property with computationally computed properties for 10 to 100 times as many molecules. 

This package facilitates the building of a working transfer-model. Relevant tools include software for filtering large datasets of molecules via similarity to the desired chemical dataset, sorting the filtered molecules to build a maximize inter-dataset similarity to the desired chemical dataset while limiting intra-similarity with itself, and computing properties via the ANI-c1xx force field for co-training. A multi-processing class is also provided for running software on multiple cores. In addition, a class for transfer the parameters from one model to another and freezing individual layers of a model before re-training are also included.

## <a name="install_package"/></a>Install package
A setup.py file is provided for installation from source.
```
cd chemprop_transfer
pip install .
```

## <a name="examples"/></a>Examples
See *examples* folder.

### <a name="filtering"/></a>Filtering
Import multiprocessing function class a DATASET class.
Create an instance of the multiprocessing class.
```python
import multiprocessing as mp
from chemprop_transfer.data import DATASET
from chemprop_transfer.data import MP_functions

num_cpus = mp.cpu_count() - 2
mp_func = MP_functions(num_cpus)
```

Remove all molecules from the desired dataset that contain elements outside the relevant sccope and canonacalize the smiles strings.

```python
if __name__ == '__main__':
    mp_func.set_filter_atoms(['C', 'H', 'N', 'O'])
    
    data_path = '../data/Mathieu_2020.csv'
    data = DATASET(data_path, chunksize=155)
    data.load_data()
    names = data.get_column_names()
    
    mp_func.apply_function(mp_func.filter_and_canonicalize, data.data
                           , in_columns=names
                           , out_file='../data/Mathieu_2020_CHNO.csv'
                           , out_columns=names)
```

Filter through "large_data.csv" to identify molecules find the 1000 molecules most similar to each molecule in "Mathieu_2020_CHNO.csv".
"large_data.csv" comes from PNNL as described in the main text of the associated paper. It can be any .csv file with SMILES strings in a column labeled "smiles".

```python
if __name__ == '__main__':
    comparison_data_path = '../data/Mathieu_2020_CHNO.csv'
    data_path = '../data/large_data.csv'
    data = DATASET(data_path, chunksize=5000)
    data.load_data()
    
    mp_func.get_similar_mols(comparison_data_path, data.data
                             , '../../similar_molecules.csv', 1000
                             , rate=1
                             , fast=False)
```

### <a name="sort"/></a>Sort molecules
Sort the molecules so that a diverse co-training dataset that maximizes that minimizes intra-group similarity can be made.

```python
from chemprop_transfer.data import DATASET
data_path = r'../data/similar_molecules.csv'
out_file = r'../data/sorted_molecules.csv'
data = DATASET(data_path)
data.load_data()
data.order_dataset(out_file, new_molecules='new_mol'
                      , target_molecules=['match_mol', 'max_mol']
                      , sim_names = ['match_sim', 'max_sim']
                      , sim_func='sum')
```

### <a name="ani"/></a>Compute ANI properties
Calculate properties with ANI to be used for co-training.
A set list of properties can be calculated and is provided to out_columns.

```python
import multiprocessing as mp
from chemprop_transfer.data import MP_functions
import torchani
from chemprop_transfer.data import DATASET
from chemprop_transfer.property_generator import PROPERTY_GENERATOR
model = torchani.models.ANI1ccx(periodic_table_index=True).double()
PG = PROPERTY_GENERATOR(model)
if __name__ == '__main__':
    out_file = './Mathieu_energies.csv'
    out_columns = ['smiles', 'energy', 'fmax', 'SYM', 'MOI1', 'MOI2', 'MOI3'
                     , 'Hvib75', 'Hvib150', 'Hvib300', 'Hvib600', 'Hvib1200'
                     , 'TSvib75', 'TSvib150', 'TSvib300', 'TSvib600', 'TSvib1200'
                     , 'FC_1', 'FC_2', 'FC_3']
    in_column = 'smiles'
    data_path = '../data/sorted_molecules.csv'
    data = DATASET(data_path, chunksize=10000)
    data.load_data()
    num_cpus = mp.cpu_count() - 2
    mp_func = MP_functions(num_cpus)
    mp_func.apply_function(PG.get_properties,data.data, in_column, out_file
                       ,out_columns, verbose=True)
```

### <a name="run_base_model"/></a>Co-train a base model
Load required modules

```python
from chemprop_transfer.data import split_data
from chemprop_transfer.data import combine_files
import shutil
from chemprop.args import TrainArgs
from chemprop.train import cross_validate, run_training
from chemprop_transfer.utils import Transfer_Model
import os
from chemprop_transfer.data import DATASET
```

Extract, select, normalize, and canonicalize data.

```python
data_path = r'../data/ani_properties_sorted.csv'
data = DATASET(data_path)
data.load_data()
data.data = data.data[data.data['fmax'] < 0.05]
data.data = data.data[['smiles', 'energy']]
data.normalize()
data.canonicalize_data(column='smiles')
data_path_FF = './ani_energy_10000r.csv'
data.save(data_path_FF, rows=10000)
```

Split data into training, validation, and test sets. Combine co-triaining and primary data.

```python
seed=1
data_path_Mathieu = '../data/log_Mathieu_2020_CHNOFCl.csv'
num_folds=5
temp_dir = './temp'
val = 0.1
test = .29333
directory1 = './Mathieu_5+energy_10000r'
split_data(data_path_Mathieu, (1-val-test, val, test), seed=seed, num_folds=num_folds
               , save_dir=directory1)
split_data(data_path_FF, (1, 0, 0), seed=seed, num_folds=num_folds
               , save_dir=temp_dir)
#combine files    
combined_dir = './combined_5M+energy_10000r'
combine_files(combined_dir, [directory1, temp_dir], multiply=[5,1])
shutil.rmtree(temp_dir)
```

Run the base model.

```python
for weighting in ['.001', '.01', '.05', '0.1', '0.5', '0.75', '1', '5']:
    save_dir = r'./combined_5M+10000r_' + weighting+'w'
    separate_test_path = os.path.join(combined_dir, 'test_full.csv')
    fold_list = ['fold_' + str(i) for i in range(num_folds)]
    base_model = Transfer_Model()
    for fold in fold_list:
        fold_folder = os.path.join(save_dir, fold)
        data_folder = os.path.join(combined_dir, fold)
        separate_val_path = os.path.join(data_folder, 'val_full.csv')
        data_path = os.path.join(data_folder, 'train_full.csv')
        if __name__ == '__main__': # and '__file__' in globals()
            # training arguments
        
            additional_args = [
                '--data_path', data_path,
                '--separate_val_path', separate_val_path,
                '--separate_test_path', separate_test_path,
                '--save_dir', fold_folder,
                '--epochs', '10', #10
                '--batch_size', '25', #25
                '--final_lr', '0.00005', #.00005
                '--init_lr', '0.00001', #.00001
                '--max_lr', '0.001', #0.0005
                #'--ffn_hidden_size', '300','20', '1000', '1000',
                '--loss_weighting', weighting,
                '--hidden_size', '300',
                '--multi_branch_ffn', "(300, 300, 20, (50, 50), (50,50))"
            ]
            train_args = base_model.get_train_args(additional_args)
            args=TrainArgs().parse_args(train_args)
            #train a model on DFT data for pretraining
            mean_score, std_score = cross_validate(args=args
                                               , train_func=run_training)
```

### <a name="transfer_model"/></a>Transfer model parameters
Transfer model parameters to a new model. Any number of model layers can be frozen.
If the models are branched, any number of branches can be transferred.

```python
from chemprop.train import cross_validate, run_training
from chemprop_transfer.utils import Transfer_Model
from chemprop.train.make_predictions import make_predictions
from chemprop.args import PredictArgs
import os
data_dir = r'./Mathieu_5+energy_10000r'
if __name__ == '__main__':
    for weighting in ['.001', '.01', '.05', '0.1', '0.5']:
        folder = './Mathieu_5M+10000r_'+weighting+'w'
        combined_dir = './combined_5M+10000r_'+weighting+'w'
        separate_test_path = os.path.join(data_dir, 'test_full.csv')
        fold_list = ['fold_' + str(i) for i in range(5)]
        for fold in fold_list:
            data_folder = os.path.join(data_dir, fold)
            fold_folder = os.path.join(folder,fold)
            separate_val_path = os.path.join(data_folder, 'val_full.csv')
            data_path = os.path.join(data_folder, 'train_full.csv')
            
            # training arguments
            additional_args = [
                '--data_path', data_path,
                '--separate_val_path', separate_val_path,
                '--separate_test_path', separate_test_path,
                '--save_dir', fold_folder,
                '--epochs', '0'
            ]
    
            #train a model on DFT data for pretraining
            BaseModel_path = os.path.join(combined_dir,fold+'/fold_0/model_0/model.pt')
            BaseModel = Transfer_Model(BaseModel_path)
            
            transfer_model, args = BaseModel.get_transfer_model(frzn_ffn_layers='all'
                                                              ,args=additional_args)
            
            
            #args.target_stds[0] = 0.1
            mean_score, std_score = cross_validate(args=args, train_func=run_training
                                                   ,model_list = [transfer_model])
            
            predict_args = ['--checkpoint_dir', fold_folder
                            , '--test_path', separate_val_path
                            , '--preds_path', os.path.join(fold_folder,'val_preds.csv')
                            ,'--num_workers', '0'
                            ]
            prediction_args = PredictArgs().parse_args(predict_args)
            make_predictions(args=prediction_args)
            
        predict_args = ['--checkpoint_dir', folder
                        , '--test_path', separate_test_path
                        , '--preds_path', os.path.join(folder,'ensemble_preds.csv')
                        ,'--num_workers', '0'
                        ]
        prediction_args = PredictArgs().parse_args(predict_args)
        make_predictions(args=prediction_args)
```

## <a name="credits"/></a>Credits
See publication for details.

Contributors:

Joshua L. Lansford <br />
Brian C. Barnes

## <a name="license"/></a>License
This project is licensed under the [MIT](https://opensource.org/licenses/MIT) license.
