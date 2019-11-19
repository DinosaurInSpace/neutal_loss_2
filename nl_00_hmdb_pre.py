import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import rdkit.DataStructs
from mordred import Calculator, descriptors
import time
import random

"""
nl_00_hmdb_pre.py:

Purpose:
This script is designed to perform pre-processing on hmdb for fingerprints,
molecular weight, and other features to reduce the overhead from recalculating 
as datasets are rerun.

See this notebook for further calculations:
'nl_00_hmdb_preprocess_test'

Steps include:
1) Perform all calculations that are not specific to NL's under investigation.
2) Export pickle

Previous script in series is:
""

Next script in series is:
"nl_01_preprocess.py"
"""


class MordClass(object):
    def __init__(self, input_path):
        self.input_path = input_path


    def load_hmdb_df(self):
        # input file from: HMDB_rdkit_output_analysis.ipynb
        hmdb_df = pd.read_pickle(self.input_path)
        hmdb_df = hmdb_df.rename(columns={'id': 'hmdb_ids'})
        return hmdb_df


    def mordred(self, mols):
        # See /mordred_features/mordred_test.ipynb
        calc = Calculator(descriptors)
        m_df = calc.pandas(mols)
        print(type(m_df))
        m_df = m_df.astype('float64').copy(deep=True)
        m_df.to_pickle('temp_mord_int.pickle')
        return m_df


    def mord_norm(self, df):
        # Use 0-1 norm as sparse for many variables
        for c in list(df.columns):
            df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())

        # Get rid of anything not in range 0-1
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        df = df.mask(df > 1, 0)
        df = df.mask(df < 0, 0)
        print('norm complete')
        return df


    def main(self):
        hmdb_df = self.load_hmdb_df()
        ids = list(hmdb_df.hmdb_ids)
        mols = list(hmdb_df.Molecule)
        m_df = self.mordred(mols)
        n_df = m_df.copy(deep=True)
        n_df = self.mord_norm(n_df)

        m_list = list(m_df.to_numpy())
        n_list = list(n_df.to_numpy())

        m_out = dict(zip(ids, m_list))
        n_out = dict(zip(ids, n_list))

        return [m_out, n_out]


class HmdbRd(object):
    def __init__(self, input_path, mord_table):
        self.input_path = input_path
        self.mord_table = mord_table


    def load_hmdb_df(self):
        # input file from: HMDB_rdkit_output_analysis.ipynb
        hmdb_df = pd.read_pickle(self.input_path)
        hmdb_df = hmdb_df.rename(columns={'id': 'hmdb_ids'})
        return hmdb_df


    def hmdb_sanitize(self, filtered_df):
        #Drop rows with na molecule
        sanitized_df = filtered_df.dropna(subset=['Molecule'], axis=0, how='any')
        return sanitized_df


    def charge(self, molecule):
        charge = Chem.rdmolops.GetFormalCharge(molecule)
        return charge


    def charges(self, x_df):
        x_df['z'] = x_df.apply(lambda x: self.charge(x['Molecule']), axis=1)
        return x_df


    def exact_mass(self, molecule):
        exact = Chem.rdMolDescriptors.CalcExactMolWt(molecule)
        return exact


    def exact_masses(self, x_df):
        x_df['exact_m'] = x_df.apply(lambda x: self.exact_mass(x['Molecule']), axis =1)
        return x_df


    def morgan_fp(self, molecule):
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(
            molecule, radius=2, nBits=1024)
        return fp


    def fp_to_array(self, fp):
        # Convert the RDKit explicit vectors into numpy arrays
        np_fps = []
        arr = np.zeros((1,))
        rdkit.DataStructs.ConvertToNumpyArray(fp, arr)
        arr.astype(int)
        np_fps.append(arr)
        return np_fps


    def molecule_fp_array(self, molecule):
        fp = self.morgan_fp(molecule)
        bits = self.fp_to_array(fp)
        bits = np.array(bits)
        bits = np.squeeze(bits, axis=None)
        bits = bits.astype(int)
        bits = list(bits)
        return bits


    def fingerprint_1024(self, x_df):
        # 1024 bit chemical fingerprint, standard settings
        # written up in "hmdb_water_CO2_results_Exp001"
        x_df['bits'] = x_df['Molecule'].apply(lambda x: self.molecule_fp_array(x))
        return x_df


    def concat(self, one, two):
        concat = np.append(one,two)
        return concat


    def find_mordreds(self, x_df):
        x_df['mordred'] = x_df['hmdb_ids'].apply(lambda x: self.mord_table[0][x])
        return x_df


    def find_mord_norms(self, x_df):
        x_df['mord_norm'] = x_df['hmdb_ids'].apply(lambda x: self.mord_table[1][x])
        return x_df


    def bool_fill(selfs, bool):
        if bool is True:
            return True
        elif bool is False:
            return False
        else:
            tf = random.choice([True, False])
            return tf


    def trues_false_rando(self, x_df):
        ids = 'trues'
        bool_dict = {'trues': True, 'falses': False, 'rando': 'Random'}

        for k, v in bool_dict.items():
            x_df[k] = x_df.apply(lambda x: self.bool_fill(v), axis=1)
        return x_df


    def fp_feats(self, x_df):
        # Fingerprint concat. with features
        id = 'fp_feats'
        x_df[id] = x_df.apply(lambda x: self.concat(x['bits'], x['mord_norm']), axis=1)
        return x_df


    def hmdb_rd_loop(self):
        # 1) Loads hmdb, 2) calculates various RD-kit things

        hmdb_df = self.load_hmdb_df()
        hmdb_out = self.hmdb_sanitize(hmdb_df)
        hmdb_out = self.charges(hmdb_out)
        hmdb_out = self.exact_masses(hmdb_out)
        hmdb_out = self.fingerprint_1024(hmdb_out)
        hmdb_out = self.find_mordreds(hmdb_out)
        hmdb_out = self.find_mord_norms(hmdb_out)
        hmdb_out = self.fp_feats(hmdb_out)
        hmdb_out = self.trues_false_rando(hmdb_out)
        return hmdb_out


### Body ###
start_time = time.time()

input_path = 'hmdb_out_molecule' # 'hmdb_test.pickle'
output_path = 'hmdb_mol_mord.pickle'

mord_loop = MordClass(input_path)
mord_table = mord_loop.main()


db_loop = HmdbRd(input_path, mord_table)
hmdb_df = db_loop.hmdb_rd_loop()
# Output, 148s
hmdb_df.to_pickle(output_path)
elapsed_time = time.time() - start_time
# 12,411s
print ('Elapsed time:\n')
print (elapsed_time)

print('\nExecuted without error\n')
print(output_path)

