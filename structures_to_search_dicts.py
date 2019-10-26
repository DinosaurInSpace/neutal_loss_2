"""
Contains structures to search against in SMARTS format
5x substrucutes from Sebastian Wolf et al BMC Bioinform.

Future to test:
140x neutral losses from Ma, Yan, et al Anal. Chem.
"""

#SMARTS
"""
cde = 
{
'Carboxylic_acid_or_conj_base': '[CX3](=O)[OX1H0-,OX2H1]'
'Hydroxyl_bonded_to_alkyl': '[OX2H][CX4]',
'Phenol': '[OX2H][cX3]:[c]'
}
"""

target_structures = {
    'Hydroxyl_c': '[CX3](=O)[OX1H0-,OX2H1]',
    'Hydroxyl_d': '[OX2H][CX4]',
    'Hydroxyl_e': '[OX2H][cX3]:[c]',
    'Cyano': '[CX2]#[NX1]',
    'Amino': '[NX3;H2,H1;!$(NC=O)]',
    'Aldehyde': '[CX3H1](=O)[#6]',
    'Carboxylic': '[CX3](=O)[OX1H0-,OX2H1]',
}

#Formula
target_loss_formula = {
    'Hydroxyl_c': 'H2Oc',
    'Hydroxyl_d': 'H2Od',
    'Hydroxyl_e': 'H2Oe',
    'Cyano': 'CN',
    'Amino': 'NH2',
    'Aldehyde': 'COH',
    'Carboxylic': 'CO2H',
}

#Exact mass Da, float
target_nl_mass = {
    'Hydroxyl_c': 18.0106,
    'Hydroxyl_d': 18.0106,
    'Hydroxyl_e': 18.0106,
    'Cyano': 27.0109,
    'Amino': 17.0266,
    'Aldehyde': 30.0106,
    'Carboxylic': 46.0055,
}

#Short encoding for column labels etc
target_short_codes = {
    'Hydroxyl_c': '001',
    'Hydroxyl_d': '001',
    'Hydroxyl_e': '001',
    'Cyano': '002',
    'Amino': '003',
    'Aldehyde': '004',
    'Carboxylic': '005',
}