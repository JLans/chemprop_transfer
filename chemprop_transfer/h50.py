#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Estimate impact sensitivity of energetic compounds from structural formulas
This script implements a variant of: J. Mol. Graph. Model. 62 (2015) 81-86
"""

import math
from collections import Counter, defaultdict

from rdkit import Chem

# Global variables are used to store data relative to explosophore groups.
# Relevant ones are:
# ExploCodes = list of strings used as identifiers for the Explosophores
#              e.g. 'x17', 'x16' denotes nitro groups with codes 17, 16
#              as defined in Table S1 in the above mentioned paper.
# Bde[explocode] = activation energy (kJ/mol) associated with the explosophore
# Pattern[explocode] = explosophore described as a RDKit-molecular substructure
# Formation_enthalpies = formation enthalpies (kJ/mol) of potential products

# SMARTS for N atom in nitro groups (encompassing two alternative descriptions)
NNITRO = "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"
TETRAZOLE = Chem.MolFromSmarts("c1[nX2][nX2][nX2][nH1]1")

# 'Explosphores' (tuple of tuples) allows one to easily specify explosophores.
# each tuple is (explocode, activation_energy_in_kJpermol, SMARTS_pattern)
Explosophores = (
    # Nitroalkane: nitro group on sp3 carbon
    ("x10", 174, "%s[CX4]" % NNITRO),
    # Nitroalkane: nitro group on sp3 carbon with one geminal nitro
    ("x11", 106, '[$(%s([CX4](%s)))][CX4]'  % (NNITRO, NNITRO),),
    # Nitroalkane: nitro group on sp3 carbon with two geminal nitros
    ("x12", 254, '[$(%s([CX4](%s)(%s)))][CX4]'  % (NNITRO, NNITRO, NNITRO),),
    # Nitro group on non-aromatic sp2 carbon
    ("x09", 216, "%s[CX3]" % NNITRO),
    # Aromatic nitro group (generic)
    ("x08", 221, "%sc" % NNITRO),
    # Aromatic nitro with one ortho -NH2
    ("x03", 235, "%scc[NX3H2]" % NNITRO),
    # Aromatic nitro with one ortho -OH
    ("x04", 238, "%scc[OX2H1]" % NNITRO),
    # Aromatic nitro with two -NH2 in ortho (like in TATB)
    ("x01", 245, "%sc(c%s)(c%s)" % (NNITRO, "[NX3H2]", "[NX3H2]"),),
    # Aromatic nitro with -NH2 and -OH in ortho positions
    ("x02", 251, "%s[$(c(c%s)c%s)]" % (NNITRO, "[OX2H1]", "[NX3H2]"),),
    # Aromatic nitro with two nitro groups in ortho
    ("x05", 254, "%s[$(c(c%s)c%s)]" % (NNITRO, NNITRO, NNITRO),),
    # Nitric ester
    ("x17", 104, "%s[OX2]" % NNITRO),
    # =N-NO2 group
    ("x15", 141, "%s[NX2H0;R0]" % NNITRO),
    # Nitramine >N-NO2
    ("x16", 120, "%s[#7;X3]" % NNITRO),
    # Nitramine -NH-NO2
    ("x13", 138, "%s[NX3H1]" % NNITRO),
    # Nitramine next to carbonyl: -N(NO2)-C=O
    ("x14", 149, "%s[NX3](C=O)" % NNITRO),
    # Nitro group in NTO-like environment
    ("x07", 244, "[$(%s[#6][#7;H1])][#6]" % NNITRO),
    # Nitro group on 3-Amino-4-nitrofurazan ring
    ("x06", 67, "%s[#6;X3]1[#6;X3]([NX3!H0])[#7;X2][#8;X2][#7;X2]1" % NNITRO,),
    # Aromatic azido group -N3
    ("cN3", 0, "c-[NX2]~[NX2]~[NX1]"),
    # -NF2 group
    ("NF2", 0, "[CX4][NX3](F)F"),
    # All aC2N2O rings
    ("aC2N2O", 208, "[$([oX2]1nccn1),$([oX2]1[#7;X3+]ccn1),$([oX2]1ncnc1),$([oX2]1cnnc1)]"),
    # Nitronium -N2+
    ("nitronium", 0, "N#[#7;X2+]cc[O-]"),
    # explosophore associated with the possibility for N2 departure
    ( "N2loss",    120, "[nX2][nX2][nX3]"),
    # Aromatic nitro with C-H in alpha position
    ("no2CH", 221, "c(%s)c[Ch]" % NNITRO),
)

# initialize global variables (ExploCodes, Bde, Pattern, Formation_enthalpies)
ExploCodes = [expl[0] for expl in Explosophores]
Bde, Pattern = dict(), dict()
for explocode, energy, smart in Explosophores:
    Bde[explocode] = energy
    Pattern[explocode] = Chem.MolFromSmarts(smart)

# Formation enthalpies and algorithm to assess decomposition products

Formation_enthalpies = dict(HF=-273.30, HCl=-92.31, H2O=-241.826,
    CO2=-393.51, CO=-110.53, Cs=0, N2=0, O2=0, F2=0, Cl2=0, H2=0, S8=0,
    CF4=-933., CCl4=-95.8, Br2=0, HBr=-36.29, NaCl=-411.12, Na=0, PO3=-450)

def getDecompoProducts(mf):
    """Return decomposition products given an empirical formula 'mf'

    Arguments:
    mf   -- empirical molecular formula provided as dictionary
    Return:
    p    -- dictionary of decomposition products:
            p['H2O'] == 2 means that decomposition of the molecule
                          yields two water molecules
    """
    allowed_elements = ['C','H','N','O','F','Cl','S','Br','Na']
    if set(mf) - set(allowed_elements):
        return None
    p = defaultdict(int)
    # first Na -> NaCl
    p['NaCl'] = min(mf['Na'],mf['Cl'])
    mf['Na'] -= p['NaCl']
    mf['Cl'] -= p['NaCl']
    # first HF
    p['HF'] = min(mf['H'],mf['F'])
    mf['H'] -= p['HF']
    mf['F'] -= p['HF']
    # HCl
    p['HCl'] = min(mf['H'],mf['Cl'])
    mf['H'] -= p['HCl']
    mf['Cl'] -= p['HCl']
    # CF4
    p['CF4'] = min(mf['C'],0.25*mf['F'])
    mf['C'] -= p['CF4']
    mf['F'] -= 4*p['CF4']
    # CCl4
    p['CCl4'] = min(mf['C'],0.25*mf['Cl'])
    mf['C'] -= p['CCl4']
    mf['Cl'] -= 4*p['CCl4']
    # H2O
    p['H2O'] = min(mf['O'],0.5*mf['H'])
    mf['O'] -= p['H2O']
    mf['H'] -= 2*p['H2O']
    # CO2
    p['CO2'] = min(mf['C'],0.5*mf['O'])
    mf['O'] -= 2*p['CO2']
    mf['C'] -= p['CO2']
    # CO
    p['CO'] = min(mf['O'],mf['C'])
    mf['C'] -= p['CO']
    mf['O'] -= p['CO']
    # HBr arbitrarily addedcompo
    p['HBr'] = min(mf['H'],mf['Br'])
    mf['H'] -= p['HBr']
    mf['Br'] -= p['HBr']
    # mf updates now unnecessary
    p['Cs'] = mf['C']
    p['H2'] = 0.5*mf['H']
    p['N2'] = 0.5*mf['N']
    p['O2'] = 0.5*mf['O']
    # ajout arbitraire
    p['F2'] = 0.5*mf['F']
    p['Cl2'] = 0.5*mf['Cl']
    p['Br2'] = 0.5*mf['Br']
    p['S8'] = 1./8*mf['S']
    p['Na'] = mf['Na']
    return p


def getXbonds(mol, patt):
    """Return set of bonds in 'mol' matching a given explosophore 'patt'

    Arguments:
    mol  -- molecule as RDKit object
    patt -- explosophore as RDKit molecular substructure
    Return:
    set of bonds, where eavery bond is represented as (i1,i2) where the tuple
    components are the indexes i1<i2 of the bonded atoms in 'mol' object
    """
    list_of_tuples = mol.GetSubstructMatches(patt)
    if not list_of_tuples:
        return set()
    bonds = [tuple(sorted(t[:2])) for t in list_of_tuples]
    return set(bonds)


def get_explosophores(mol):
    """Return occurrences of all explosophores found in molecule

    Arguments:
    mol  -- molecule as RDKit object
    Return:
    counter -- counter[explocode] = number of occurrences of explosophore
    """
    dic = dict()
    for expl_code in ExploCodes:
        for bond in getXbonds(mol, Pattern[expl_code]):
            dic[bond] = expl_code
    exploCounter = Counter(dic.values())
    # account for other possibility for N2 release from tetrazole
    exploCounter['N2loss'] += len(getXbonds(mol, TETRAZOLE))
    return exploCounter


def Z(params, explocode):
    """Return prefactor associated with a given explosphore

    Arguments:
    params    -- parameters of the model
    explocode -- code of the explosophore
    Return:
    prefactor -- numerical value
    """
    if explocode in ("x13", "x14", "x15", "x16"):
        return params["ZN"]
    if explocode in ("x17",):
        return params["ZO"]
    return 1


def get_h50(smile, hof=0, return_expls=False, params=None):
    """Return tuple:
       h50 = impact sensitivity as ERL Type 12 drop-weight impact height (cm)
       expls = dict of explosophores: expls[ex] = number of ex on molecule mol
    Arguments:
    smile       -- smile string
    hof       -- heat of formation of the material (kJ/mol) ignored by default
    Return:
    h50 -- estimated ERL Type 12 drop-weight impact height
    """
    if params is None:
        params = dict(kcrit=0.359, eta=29.992, ZN=1.44, ZO=1.37, aC2N2O=385.834, N2loss=277.009)
    mol = Chem.MolFromSmiles(smile)
    molH = Chem.AddHs(mol)        # molecule with explicit hydrogen atoms
    natom = molH.GetNumAtoms()    # total number of atoms in molecule
    mf = Counter(atom.GetSymbol() for atom in molH.GetAtoms())
    decprods = getDecompoProducts(mf)           # decomposition products
    # set Ec to energy content (kJ/mol) obtained as the difference between the
    # formation enthalpy of the material (hof) and the sum of corresponding 
    # values for the decomposition products stored in decprops dictionary:
    # e.g. decprods['H2O'] = number of moles of H2O in decomposition products
    #      Formation_enthalpies['H2O'] = formation enthalpy of water (kJ/mol)
    Ec = hof - sum(decprods[k]*Formation_enthalpies[k] for k in decprods)
    expls = get_explosophores(mol)  # store explosophores into dict-like structure
    if len(expls) == 0:
        return 100000, []  # No explosophore => inert material
    # nEc is the inverse effective temperature 1/kTe = (3/2)*NA/(eta*Ec) in mol/kJ
    # This is the quantity to be multiplied by bond dissociation energies when 
    # computing a rate constant 
    nEc = 1.5 * natom / (Ec * params["eta"])
    # The loop below updates the dict of bond dissociation energies 
    # using current parameters
    for w in expls:
        if w in params:
            Bde[w] = params[w]
    # propagation rate depends on a sum over explosophores (w)
    kpr = sum(expls[w] * Z(params, w) * math.exp(-nEc * Bde[w]) for w in expls)
    kpr /= natom
    # Power law expression specific to the current model to estimate h50
    # from the 'kpr' rate constant for the propagation of the decomposition
    h50 = (params["kcrit"] / kpr) ** 4 if kpr > 1.E-20 else 1.E8
    if return_expls == True:
        return h50, expls
    else:
        return h50

def get_h50_mol(params, mol, hof=0):
    """Return tuple:
       h50 = impact sensitivity as ERL Type 12 drop-weight impact height (cm)
       expls = dict of explosophores: expls[ex] = number of ex on molecule mol
    Arguments:
    params    -- parameters of the model
    mol       -- energetic molecule as RDkit object
    hof       -- heat of formation of the material (kJ/mol) ignored by default
    Return:
    h50 -- estimated ERL Type 12 drop-weight impact height
    """
    molH = Chem.AddHs(mol)        # molecule with explicit hydrogen atoms
    natom = molH.GetNumAtoms()    # total number of atoms in molecule
    mf = Counter(atom.GetSymbol() for atom in molH.GetAtoms())
    decprods = getDecompoProducts(mf)           # decomposition products
    # set Ec to energy content (kJ/mol) obtained as the difference between the
    # formation enthalpy of the material (hof) and the sum of corresponding 
    # values for the decomposition products stored in decprops dictionary:
    # e.g. decprods['H2O'] = number of moles of H2O in decomposition products
    #      Formation_enthalpies['H2O'] = formation enthalpy of water (kJ/mol)
    Ec = hof - sum(decprods[k]*Formation_enthalpies[k] for k in decprods)
    expls = get_explosophores(mol)  # store explosophores into dict-like structure
    if len(expls) == 0:
        return 100000, []  # No explosophore => inert material
    # nEc is the inverse effective temperature 1/kTe = (3/2)*NA/(eta*Ec) in mol/kJ
    # This is the quantity to be multiplied by bond dissociation energies when 
    # computing a rate constant 
    nEc = 1.5 * natom / (Ec * params["eta"])
    # The loop below updates the dict of bond dissociation energies 
    # using current parameters
    for w in expls:
        if w in params:
            Bde[w] = params[w]
    # propagation rate depends on a sum over explosophores (w)
    kpr = sum(expls[w] * Z(params, w) * math.exp(-nEc * Bde[w]) for w in expls)
    kpr /= natom
    # Power law expression specific to the current model to estimate h50
    # from the 'kpr' rate constant for the propagation of the decomposition
    h50 = (params["kcrit"] / kpr) ** 4 if kpr > 1.E-20 else 1.E8
    return h50, expls