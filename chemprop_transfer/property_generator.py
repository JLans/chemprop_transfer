# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:45:56 2021

@author: joshua.l.lansford
"""
import torchani
from rdkit import Chem
import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.geometry.analysis import Analysis
import torch
from pmutt.statmech.vib import HarmonicVib
from pmutt.statmech.rot import get_geometry_from_atoms
from pmutt.constants import R
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

class PROPERTY_GENERATOR:
    def __init__(self, model):
        self.model = model
        self.ddofs = {'linear': 5, 'nonlinear': 6}
        
    def get_symm_num(self, pointgroup):
        if pointgroup[0:2] in ('Ci', 'Cs', 'Kh', 'C*'):
            return 1
        elif pointgroup[0:2] == 'D*':
            return 2
        elif pointgroup[0] == 'T':
            return 12
        elif pointgroup[0] == 'O':
            return 24
        elif pointgroup[0] == 'I':
            return 60
        elif pointgroup[0] == 'C':
            return int(pointgroup[1])
        elif pointgroup[0] == 'D':
            return 2 * int(pointgroup[1])
        elif pointgroup[0] == 'S':
            return int(pointgroup[1]) / 2
        else:
            return 0

    def get_force_constants(self, num, unique_bonds, mol, disp=0.01):
        
        def get_force_constant(mol,atom1, atom2, disp=0.01):
            dist = (disp / 2 * mol.get_distance(0,1,vector=True)
                   /mol.get_distance(0,1,vector=False))
            mol_pos = mol.copy()
            mol_pos.set_calculator(self.model.ase())
            mol_neg = mol.copy()
            mol_neg.set_calculator(self.model.ase())
            mol_pos[atom1].position += dist
            mol_pos[atom2].position -= dist
            mol_neg[atom2].position += dist
            mol_neg[atom1].position -= dist
            f_i = mol.get_potential_energy()
            f_pos = mol_pos.get_potential_energy()
            f_neg = mol_neg.get_potential_energy()
            force_constant = (f_pos - 2 * f_i + f_neg)/disp**2
            return force_constant
        
        bonds = []
        for index, val in enumerate(unique_bonds[0]):
            for mol_index in val:
                bonds.append([index,mol_index])
        force_constants = np.zeros(max(num,len(bonds)))
        for index, mol_pair in enumerate(bonds):
            force_constants[index] = get_force_constant(mol, mol_pair[0], mol_pair[1]
                                                    , disp=disp)
        return np.sort(force_constants)[0:num]
    

    def get_properties(self,smile):
        row = []
        m = Chem.MolFromSmiles(smile)
        m2=Chem.AddHs(m)
        symbols = [atom.GetSymbol() for atom in m2.GetAtoms()]
        # run ETKDG 10 times
        numConfs = 3
        params = Chem.AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        params.useRandomCoords = True
        Chem.AllChem.EmbedMultipleConfs(m2, numConfs=numConfs , params=params)
        if len(m2.GetConformers()) > 0:    
            MMFF_energies = Chem.AllChem.MMFFOptimizeMoleculeConfs(m2, maxIters=2000)
            best_conformer_index = int(np.array(MMFF_energies)[:,1].argmin())
            best_conformer = m2.GetConformers()[best_conformer_index]
            mol = Atoms(symbols,best_conformer.GetPositions())
            mol.set_calculator(self.model.ase())
            opt = BFGS(mol,logfile='../temp.out')
            opt.run(fmax=0.001, steps=150)
            energy = mol.get_potential_energy()
            geometry = get_geometry_from_atoms(mol)
            mol_anl = Analysis(mol)
            force_constants = self.get_force_constants(3, mol_anl.unique_bonds, mol)
            if geometry != 'monotonic':
                species = torch.tensor(mol.get_atomic_numbers(), dtype=torch.long).unsqueeze(0)
                coordinates = torch.from_numpy(mol.get_positions()).unsqueeze(0).requires_grad_(True)
                masses = torchani.utils.get_atomic_masses(species)
                energies = self.model((species, coordinates)).energies
                hessian = torchani.utils.hessian(coordinates, energies=energies)
                freq, modes, fconstants, rmasses = torchani.utils.vibrational_analysis(masses, hessian, mode_type='MDU')
                pg = PointGroupAnalyzer(AseAtomsAdaptor.get_molecule(mol))
                #print(pg.sch_symbol)
                #print(self.get_symm_num(pg.sch_symbol))
                StatMech = HarmonicVib(freq[self.ddofs[geometry]:])
                temp = [75, 150, 300, 600, 1200]
                Hvib = [StatMech.get_HoRT(T)*R('eV/K')*T for T in temp]
                TSvib = [StatMech.get_SoR(T)*R('eV/K')*T for T in temp]
                max_force = np.max(np.mean(mol.get_forces()**2,axis=1)**0.5)
                row += ([smile] + [energy] + [max_force]
                    + [self.get_symm_num(pg.sch_symbol)]
                    + mol.get_moments_of_inertia().tolist() #amu/A^2
                    + Hvib
                    + TSvib
                    + force_constants.tolist())
        if len(row) == 20:
            row += [True]
        else:
            row += [False]
        return row
