This file contains a description of data files in the chemprop_transfer/data directory

Data that already exists in the public domain.
1) Mathieu_2020.csv
     Description
       This file contains data taken from the supporting information of the article published by Mathieu located at 
       https://doi.org/10.1002/prep.201900377 (PropellantsExplos.Pyrotech.2020,45,966–973).
     Columns
       smiles: SMILES strings for molecules subjected to impact sensitivity tests
       h50(exp): Experimental impact sensitivity measurments collated by Mathieu.
       h50(calc): Calclulated impact sensitivities by Mathieu using his physics-based model

2) log_Mathieu_2020_CHNOFCl.csv
     Description
       This file contains the log transformation of h50(exp) data in Mathieu_2020.csv
     Columns
       smiles: SMILES strings for molecules subjected to impact sensitivity tests, canonicalized using RDKit
       log10h50(exp): Base 10 log tranformation of h50(exp) data found in Mathieu_2020.csv

3) Casey_DFT_data.csv
     Description
       This file contains data taken from the supporting information of the article published by Casey et al.
       located at https://pubs.acs.org/doi/10.1021/acs.jcim.0c00259 (J. Chem. Inf. Model. 2020, 60, 10, 4457–4473)
     Columns
       smiles: SMILES string for molecule taken from the GDB database filtered on oxygen balance
       electronic_energy (Hartree): Electronic energy associated of the molecule calculated using DFT
       HOMO_LUMO_gap (eV): Energy difference between the highest occupied and lowest unoccupied molecular orbitals
       dipole_moment (Debye): Dipole moment of the molecule calculated by DFT
       crystal_density (g/cc): Predicted crystal density
       heat_of_formation (kcal/mol): Calculated heat of formation data
       set_designation: Indication of whether the data is in the training, validation, or test set.

4) Casey_DFT_train_norm.csv
     Description
       Data taken from Casey_DFT_data.csv where set_designation = 'train' and data is normalized by the column's
       mean and standard deviation.
     Columns
       contain's all columns foudn in Casey_DFT_data.csv, excluding set_designation

5) Casey_DFT_val_norm.csv
     Description
       Data taken from Casey_DFT_data.csv where set_designation = 'valid' and data is normalized by the column's
       mean and standard deviation.
     Columns
       contain's all columns foudn in Casey_DFT_data.csv, excluding set_designation


6) Casey_DFT_test_norm.csv
     Description
       Data taken from Casey_DFT_data.csv where set_designation = 'test' and data is normalized by the column's
       mean and standard deviation.
     Columns
       contain's all columns found in Casey_DFT_data.csv, excluding set_designation
       
   
Newly generated data
1) similar molecules.csv
     Description
       This file contains molecules selected from a dataset of 172 million molecules described in a publication
       by McGrady et al at Pacific Northwest National Labs (PNNL) located at https://doi.org/10.48550/arXiv.2201.12398
       (AI for Chemical Space Gap Filling and Novel Compound Generation. arXiv preprint arXiv:2201.12398 (2022).)
       Molecules were selected based on their similarity to the molecules in Mathieu_2020.csv file as described in
       the article published by Lansford, Barnes, Rices, and Jensen at (journal TBA).
     Columns
       new_mol: SMILES strings selected from the PNNL dataset
       match_mol: SMILE string from Mathieu_2020.csv for which the molecule from PNNL was selected
       match_sim: Tanimoto similarity calculated using RDKIT between the "new" and "match_mol" SMILEs
       max_mol: SMILE string from Mathieu_2020.csv for which the molecule from PNNL is most similar to
       max_sim: Tanimoto similarity calculated using RDKIT between the "new" and "max_mol" SMILEs

2) sorted_molecules.csv
     Description
       This file contains all molecules from similar molecules.csv sorted in order of "additional similarity"
       to the data in Mathieu_2020.csv as described in the Methods of the article published by Lansford, Barnes, 
       Rices, and Jensen at (journal TBA).
     Columns
       new_mol: SMILES strings selected from the PNNL dataset
       ext_sim: Sum of "match_sim" and "max_sim" columns
       self_sim: Maximum similarity between "new_mol" and all other molecules already selected
       difference: Difference calculated by subtracting "self_sim" from "ext_sim"
       group_order: The order in which the molecule should be selected for generating an optimally diverse dataset
                    similar to Mathieu_2020.csv
       match_mol: SMILE string from Mathieu_2020.csv for which the molecule from PNNL was selected
       match_sim: Tanimoto similarity calculated using RDKIT between the "new" and "match_mol" SMILEs
       max_mol: SMILE string from Mathieu_2020.csv for which the molecule from PNNL is most similar to
       max_sim: Tanimoto similarity calculated using RDKIT between the "new" and "max_mol" SMILEs

3) ani_properties_sorted.csv
     Description
       This file contains all molecules from sorted_molecules.csv with varous molecules properties calculated
       using the ANI-1ccx force field as described in the Methods of the article published by Lansford, Barnes, 
       Rices, and Jensen at (journal TBA).
     Columns
       SMILES: SMILES strings selected from the PNNL dataset and labeled as "new_mol" in sorted_molecules.csv
       energy: Electronic energy of the molecule
       fmax: Maximum force between two atoms of the molecule
       SYM: Symmetry number of the molecule
       MOI1: First moment of inertia of the molecule
       MOI2: Second moment of inertia of the molecule
       MOI3: Third moment of inertia of the molecule
       Hvib75: Vibrational enthalpy calculated at 75K
       Hvib150: Vibrational enthalpy calculated at 150K
       Hvib300: Vibrational enthalpy calculated at 300K
       Hvib600: Vibrational enthalpy calculated at 600K
       Hvib1200: Vibrational enthalpy calculated at 1200K
       TSvib75: Temperature multilied by vibrational entopy at 75K
       TSvib150: Temperature multilied by vibrational entopy at 150K
       TSvib300: Temperature multilied by vibrational entopy at 300K
       TSvib600: Temperature multilied by vibrational entopy at 600K
       TSvib1200: Temperature multilied by vibrational entopy at 1200K
       FC_1: Lowest force constant computed for all bonded atoms
       FC_2: Second lowest force constant computed for all bonded atoms
       FC_3: Third lowest force constant computed for all bonded atoms

4) ani_properties_filtered_and_normalized.csv
     Description
       This file contains all molecules from ani_properties_sorted.csv where "fmax" is less than 0.5 eV/A and
       each property is normalized by the mean and standard deviation as described in the Methods of the article
       published by Lansford, Barnes, Rices, and Jensen at (journal TBA).
     Columns
       Contains all columns ani_properties_sorted.csv, excluding "fmax".