import numpy as np
from rdkit.Chem import MolFromInchi
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AddHs
class Simple_Molecule:
    """Class for loading an manipulating a dataset"""
    def __init__(self, mol_str, str_type='smiles'):
        """ 
        Parameters
        ----------
        mol_string : str
            smiles string.
        """
        if str_type.lower() == 'smiles':
            self.str_type = 'smiles'
            self.molecule = AddHs(MolFromSmiles(mol_str))
        elif str_type.lower() == 'inchi':
            self.str_type = 'inchi'
            self.molecule = AddHs(MolFromInchi(mol_str))
        self.mol_str = mol_str
        self._atoms = None
        self._atomic_numbers = None
        self._bonds = None
        self._num_atoms = None
        self._num_bonds = None
        self._get_molecule()
    
    def _get_molecule(self):
        atoms = []
        #bonds = []
        for atom in self.molecule.GetAtoms():
            atoms.append([atom.GetSymbol(), atom.GetAtomicNum()])
        atoms = np.array(atoms)
        atoms, num_atoms = np.unique(atoms, return_counts=True, axis=0)
        atomic_numbers = atoms[:,1].astype(int)
        index_sort = np.argsort(atomic_numbers)
        self._atoms = atoms[:,0].astype('<U2')[index_sort]
        self._atomic_numbers = atoms[:,1].astype(int)[index_sort]
        self._num_atoms = num_atoms[index_sort]
        
    def get_atoms(self):
        return self._atoms
    
    def get_atomic_numbers(self):
        return self._atomic_numbers
    
    def get_num_atoms(self, atom_type='all'):
        if atom_type == 'all':
            return self._num_atoms
        elif atom_type == 'sum':
            return self._num_atoms.sum()
        elif atom_type == 'heavy':
            total = 0
            for atom in self.get_atoms():
                if atom != 'H':
                    total += self._num_atoms[list(self._atoms).index(atom)]
            return total
        else:
            if atom_type in self.get_atoms():
                return self._num_atoms[list(self._atoms).index(atom_type)]
            else:
                return 0
            
            