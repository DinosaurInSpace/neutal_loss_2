import pandas as pd
import numpy as np
from structures_to_search_dicts import target_structures, target_loss_formula
import time
import rdkit.Chem as Chem
from rdkit.Chem import rdFMCS
from scipy import stats


# import pandas
# from pandas import DataFrame as df
# import pickle
# import matplotlib.pyplot as plt
# from sklearn.metrics import jaccard_score
# from structures_to_search_dicts import target_nl_mass, target_short_codes
# from rdkit.Chem import rdmolops
# from rdkit.Chem import rdMolDescriptors
# from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
# from rdkit.Chem import AllChem
# import rdkit.DataStructs
# from rdkit.Chem import Fragments


"""
nl_02_join.py:

Purpose:
This script is designed to join neutral loss search results with hmdb metadata prior to 
module 3 filtering/model building/testing).

Steps include:
1) Load data from both pickles.
2) Split data frame into one and many ids.     
3) One ID is simple join between both inputs.
4) Many is a complicated join.
    -Standard fields are passed as mode, common substructure, or concatonation of names
    -Variable fields are passed as mode.
5) Join one and many, then export to pickle.

Next script in series is:
"nl_03_filter_model"
Previous script ion series is:
"nl_01_preprocess"

"""

class OneID(object):
    def __init__(self, hmdb_df, input_dict):
        self.hmdb_df = hmdb_df
        self.input_dict = input_dict


    def clean(self, one_id_list):
        one_id = one_id_list[0]
        return one_id


    def column_clean(self, one_id_df):
        one_id_df['hmdb_ids'] = one_id_df['hmdb_ids'].apply(lambda x: self.clean(x))
        return one_id_df


    def check_hmdb_id(self, one_id_df):
        # There are drug bank hits in here numbered by hmdb!  These don't hit against
        # the hmdb metadata, and can be dropped as expert throws a nan as type float.
        one_id_df['expert_type'] = one_id_df['expert'].apply(type)
        good_ids = one_id_df[one_id_df['expert_type'] == list]
        return good_ids


    def main_loop(self, one_id_df):
        # Changes list type to str, then simple join on id.
        one_id_df = self.column_clean(one_id_df)
        joined_df = pd.merge(one_id_df, self.hmdb_df, how='left', on='hmdb_ids')
        joined_df = self.check_hmdb_id(joined_df)
        joined_df['join_index'] = np.nan
        joined_df['MCS_smarts'] = np.nan
        return joined_df


class ManyID(object):
    def __init__(self, hmdb_df, input_dict, joined_df, index_list):
        self.hmdb_df = hmdb_df
        self.input_dict = input_dict
        self.joined_df = joined_df
        self.index_list = index_list


    def find_smarts_common(self, rdobjs):
        # Find common substructure for several hits
        # Try as invalid objects can fail!
        rdobjs = list(rdobjs)
        if len(rdobjs) > 1:
            try:
                compare = rdFMCS.FindMCS(rdobjs, timeout=180)
                smarts = compare.smartsString
                return smarts
            except:
                return None
        else:
            try:
                smarts = Chem.MolToSmarts()
                return smarts
            except:
                return None


    def smarts_to_rd(self, smarts):
        # Try as invalid objects can fail!
        try:
            rdobj = Chem.MolFromSmarts(smarts)
            return rdobj
        except:
            return None


    def bin_array_avg(self, np_array_list):
        # join arrays to 2d and calculate mode by column
        array_2d = np.array(np_array_list)
        modes = list(stats.mode(array_2d, axis=0).mode)
        return modes


    def find_present_columns(self, all_columns):
        present_columns = []
        for column in all_columns:
            if '_Present' in column:
                present_columns.append(column)
            else:
                continue
        return present_columns


    def find_rd_columns(self, all_columns):
        rd_columns = []
        for column in all_columns:
            if '_RDtarget' in column:
                rd_columns.append(column)
            else:
                continue
        return rd_columns


    def mode_variable_present(self, present_df):
        #All T or F, can treat as np array
        array_2d = present_df.to_numpy()
        modes = stats.mode(array_2d, axis=0).mode
        keys = present_df.columns
        values = modes.tolist()
        variable_dict = dict(zip(keys,values[0]))
        return variable_dict


    def mode_variable_rd(self, rd_df):
        #Should be the same throughout, use first row.
        keys = rd_df.columns
        values = list(rd_df.iloc[0])
        variable_dict = dict(zip(keys, values))
        return variable_dict


    def check_hmdb_id(self, many_id_df):
        # There are drug bank hits in here numbered by hmdb!  These don't hit against
        # the hmdb metadata, and can be dropped as expert throws a nan as type float.
        many_id_df['expert_type'] = many_id_df['expert'].apply(type)
        good_ids = many_id_df[many_id_df['expert_type'] == list]
        return good_ids


    def hmdb_merge(self, hmdb_list):
        # Get list of hmdb_ids, lookup as df from self.hmdb_df
        hmdb_hits = self.hmdb_df[self.hmdb_df['hmdb_ids'].isin(hmdb_list)]
        hmdb_hits = self.check_hmdb_id(hmdb_hits)

        if hmdb_hits.empty is False:

            smarts = self.find_smarts_common(hmdb_hits.Molecule)
            molecule = self.smarts_to_rd(smarts)
            join_index = self.index_list[0]
            print(join_index)

            # Ensures join_indexes are not repeated
            self.index_list = self.index_list[1:]

            # Make dict of standard metadata
            # Can't make valid inchi or smarts for substructure smarts
            standard_dict = {'formula': hmdb_hits.formula.mode().iloc[0],
                        'hmdb_ids': '_'.join(hmdb_list),
                        'inchi': None,
                        'mol_name': '_'.join(hmdb_hits.mol_name),
                        'Smiles': None,
                        'MCS_smarts': smarts,
                        'Molecule': molecule,
                        'in_expt': 'True',
                        'z': hmdb_hits.z.mode().iloc[0],
                        'exact_m': hmdb_hits.exact_m.mode().iloc[0],
                        'bits': self.bin_array_avg(list(hmdb_hits.bits))[0],
                        'expert': self.bin_array_avg(list(hmdb_hits.expert))[0],
                        'fp_1024_expert': self.bin_array_avg(list(hmdb_hits.fp_1024_expert))[0],
                        'expert_key': hmdb_hits.expert_key.iloc[0],
                        'trues': hmdb_hits.trues.mode().iloc[0],
                        'falses': hmdb_hits.falses.mode().iloc[0],
                        'rando': hmdb_hits.rando.mode().iloc[0],
                        'mordreds': self.bin_array_avg(list(hmdb_hits.mordreds))[0],
                        'mord_norms': self.bin_array_avg(list(hmdb_hits.mord_norms))[0],
                        'fp_feats': self.bin_array_avg(list(hmdb_hits.fp_feats))[0],
                        'formulas': hmdb_hits.x.mode().iloc[0],
                        'join_index': join_index
                        }

            # Make dict of variable metadata
            present_columns = self.find_present_columns(hmdb_hits.columns)
            rd_columns = self.find_rd_columns(hmdb_hits.columns)
            present_dict = self.mode_variable_present(hmdb_hits[present_columns])
            rd_dict = self.mode_variable_rd(hmdb_hits[rd_columns])
            present_dict.update(rd_dict)

            # Join standard and variable metadata, to df, append to self.df
            standard_dict.update(present_dict)
            dict_list = [standard_dict]
            out_df = pd.DataFrame(dict_list)
            self.joined_df = self.joined_df.append(out_df, ignore_index = True)

            return join_index

        else:
            return None


    def main_loop(self, many_id_df):
        """
        1) Pass hmdb ids to second function
        2) Merge rows with of hmdb ids according to logic
        3) Add to self.joined_df with join index
        4) Return join index, delete first entry from join index
        5) Join self.joined_df and many id on join index
        """
        many_id_df['join_index'] = many_id_df['hmdb_ids'].apply(lambda x: self.hmdb_merge(x))
        many_merged_df = pd.merge(many_id_df, self.joined_df, how='left', on='join_index')

        # Clean-up headers before final joins in main:
        many_merged_df.drop('hmdb_ids_x', axis=1, inplace=True)
        many_merged_df.rename(columns={'hmdb_ids_y': 'hmdb_ids'}, inplace=True)
        return many_merged_df


### Body ###
start_time = time.time()

# Setup classes
hmdb_joined_df = pd.DataFrame()
formula_dict = target_loss_formula
input_file = 'en_nl_01_output_df_exptl.pickle'
input_hmdb = 'en_nl_01_hmdb_df_exptl.pickle'

input_df = pd.read_pickle(input_file)
hmdb_df = pd.read_pickle(input_hmdb)
join_index_list = range(0, 999999) # Or number of rows at least

one = OneID(hmdb_df, formula_dict)
many = ManyID(hmdb_df, formula_dict, hmdb_joined_df, join_index_list)

# Run main loops
one_df = input_df[input_df.num_ids == False].copy(deep=True)
many_df = input_df[input_df.num_ids == True].copy(deep=True)

one_join = one.main_loop(one_df)
many_join = many.main_loop(many_df)

final_df = pd.concat([one_join, many_join], ignore_index=True, join="outer", sort=True)

# Clean-up and export
final_df.drop('formula_y', axis=1, inplace=True)
final_df.rename(columns={'formula_x': 'formula'}, inplace=True)

final_df.to_pickle('en_nl_02_join_exptl.pickle')

# 13,937! Slow, but reduced fails on find MCS.
# Optional feature, could be removed for speed...

print('\nExecuted without error\n')
elapsed_time = time.time() - start_time
print ('Elapsed time:\n')
print (elapsed_time)