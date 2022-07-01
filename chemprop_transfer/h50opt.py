#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Estimate impact sensitivity of energetic compounds from structural formulas
This script implements the model in: J. Mol. Graph. Model. 62 (2015) 81-86
usage: python h50opt.py <input.file>
The input file is a simple text file where each line corresponds to a compound
from the training set with the following content:
<SMILES for compound>\t<experimental h50 value in cm>\t<compound name>

The script requires lmfit which may be installed as follows:
$ pip install lmfit --user
The --user option allows any user to get it working in its own directory
"""

from math import log10, sqrt
from lmfit import minimize, Parameters
from rdkit import Chem
from . import h50

LOWESTRMSE = 9.E99
def func(params, mols, expes):
    """
    Return list of residuals for current model when applied to molecules
    This function is requested by the lmfit library to optimize the parameters
    
    Parameters
    ----------
    params: lmfit.Parameters
        dictionary-like structure storing current model parameters
    mols: list of rdkit.Chem.rdchem.Mol objects
        molecules for which the model is to be applied
    expes: list of float
        list of corresponding experimental h50 values (any unit acceptable)
        
    Returns
    ----------
    residus: list of float
        Values of the residuals (i.e. differences between estimated and actual h50 values) 
    """
    global LOWESTRMSE
    for code in params:
        if code in h50.Bde:
            h50.Bde[code] = params[code]
    calcs = [h50.get_h50_mol(params, mol)[0] for mol in mols]
    residus = [log10(c)-log10(expe) for c, expe in zip(calcs, expes)]
    mse = sum(r**2 for r in residus)/len(residus)
    rmse = sqrt(mse)
    if rmse < LOWESTRMSE:
        print('%f' %rmse + ''.join(' %s=%.0f' %(k, params[k]) for k in params if params[k].vary))
        LOWESTRMSE = rmse
    return residus

def get_params(smiles, h50):

    # initial parameters
    params = Parameters()
    params.add('eta', 1., vary=True, min=0, max=1000)
    params.add('kcrit', 1., vary=True, min=0, max=500)
    params.add('ZN', 1., vary=True, min=0.6, max=2)
    params.add('ZO', 1., vary=True, min=0.6, max=2)
    params.add('aC2N2O', 200., vary=True, min=0, max=1000)
    params.add('N2loss', 200., vary=True, min=0, max=1000)

    # read input file => mols, expes, names
    data = smiles
    mols = [Chem.MolFromSmiles(d) for d in data]
    expes = h50
    
    print('----- starting parameters -----')
    for k in params:
        try:
            unc = float(params[k].stderr)
            unc = '%5f' %unc
        except TypeError:
            unc = '  N/A'
        print('%s = %8.3f  +/- %s' %(k.ljust(16), params[k].value, unc))
    # optimize the parameters
    print('----- starting optimization -----')
    fitout = minimize(func, params, args=(mols, expes))
    print('----- restarting optimization -----')
    fitout = minimize(func, fitout.params, args=(mols, expes))
    print('----- finished optimization -----')

    return fitout.params