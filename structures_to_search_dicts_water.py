"""
Contains structures to search against in SMARTS format
5x substrucutes from Sebastian Wolf et al BMC Bioinform.

Future to test:
140x neutral losses from Ma, Yan, et al Anal. Chem.

Modified below for water only, see "structures_search_dicts.py" for all 5x.
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
}

#Formula
target_loss_formula = {
    'Hydroxyl_c': 'H2Oc',
    'Hydroxyl_d': 'H2Od',
    'Hydroxyl_e': 'H2Oe',
}

#Exact mass Da, float
target_nl_mass = {
    'Hydroxyl_c': 18.0106,
    'Hydroxyl_d': 18.0106,
    'Hydroxyl_e': 18.0106,
}

#Short encoding for column labels etc
target_short_codes = {
    'Hydroxyl_c': '001',
    'Hydroxyl_d': '001',
    'Hydroxyl_e': '001',
}