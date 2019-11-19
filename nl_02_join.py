import pandas as pd
import numpy as np
from structures_to_search_dicts import target_structures, target_loss_formula
import time
import rdkit.Chem as Chem
from rdkit.Chem import rdFMCS
from scipy import stats
import argparse

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

Example command:
python nl_02_join.py --o en_nl_exptl_stats_output_01.pickle --h en_nl_exptl_stats_hmdb_01.pickle
"""

class OneID(object):
    def __init__(self, hmdb_df, input_dict):
        self.hmdb_df = hmdb_df
        self.input_dict = input_dict


    def clean(self, one_id_list):
        # Entries are a list with one item.
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
        #print(type(one_id_df.hmdb_ids[0]))
        print(self.hmdb_df.iloc[0,1])
        joined_df = pd.merge(one_id_df, self.hmdb_df, how='left', on='hmdb_ids')
        joined_df = self.check_hmdb_id(joined_df)
        joined_df['join_index'] = np.nan
        joined_df = joined_df.rename(columns={'formula_x': 'formula'})
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
        # Disabled as very slow
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
                        'in_expt': 'True',
                        'z': hmdb_hits.z.mode().iloc[0],
                        'exact_m': hmdb_hits.exact_m.mode().iloc[0],
                        'bits': self.bin_array_avg(list(hmdb_hits.bits))[0],
                        'expert': self.bin_array_avg(list(hmdb_hits.expert))[0],
                        'expert_key': hmdb_hits.expert_key.iloc[0],
                        'trues': hmdb_hits.trues.mode().iloc[0],
                        'falses': hmdb_hits.falses.mode().iloc[0],
                        'rando': hmdb_hits.rando.mode().iloc[0],
                        'mordred': self.bin_array_avg(list(hmdb_hits.mordred))[0],
                        'mord_norm': self.bin_array_avg(list(hmdb_hits.mord_norm))[0],
                        'fp_feats': self.bin_array_avg(list(hmdb_hits.fp_feats))[0],
                        #'formulas': hmdb_hits.x.mode().iloc[0],
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


def column_clean(x_df):
    # Organize and limit extraneous columns carried forward.
    head = ['formula', 'hmdb_ids', 'mol_name', 'polarity', 'adduct', 'ds_id',
            'num_ids', 'z', 'exact_m', 'inchi',  'Smiles', 'in_expt']
    mid_targets = ['_Present', 'fdr', 'colocalization_', 'loss_intensity_share_',
               'n_loss_only_', 'n_loss_wparent_']
    tail = ['trues', 'falses', 'rando', 'fp_feats', 'expert','bits', 'mord_norm',
            'join_index']
    cols = list(x_df.columns)
    middle = []
    for targ in mid_targets:
        for c in cols:
            if c.find(targ) != -1:
                middle.append(c)
    master = head + middle + tail
    x_df = x_df[master].copy(deep=True)
    return x_df


### Body ###
start_time = time.time()

# Setup classes and input files
hmdb_joined_df = pd.DataFrame()
formula_dict = target_loss_formula

parser = argparse.ArgumentParser(description='')
parser.add_argument("--o", default=None, type=str, help="nl_01_preprocess out pickle")
parser.add_argument("--h", default=None, type=str, help="nl_01_preprocess hmdb pickle")
args = parser.parse_args()
input_file = args.o
input_hmdb = args.h

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

one_join = column_clean(one_join)
many_join = column_clean(one_join)

final_df = pd.concat([one_join, many_join], ignore_index=True, join="outer", sort=True)

# Export
out_stub = input_file.split('_output_01')[0]
out_file = out_stub + '_output_02.pickle'
final_df.to_pickle(out_file)

# 784s
elapsed_time = time.time() - start_time
print('Elapsed time:\n')
print('\nExecuted without error\n')
print(elapsed_time)
print(out_file)