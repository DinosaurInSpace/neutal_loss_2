import pandas as pd
import numpy as np
from structures_to_search_dicts import target_structures, target_loss_formula
import rdkit.Chem as Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import rdkit.DataStructs
import time
import random


"""
nl_01_preprocess:

Purpose:
This script is designed to perform pre-processing on neutral loss searches from
metaspace prior to module 2 (handle multiple hits and join), and module 3 
filtering/model building/testing.

Steps include:
1) Parse metaspace search results and export pickle.     
2) Parse hmdb and apply RDkit to molecules in 1) and export pickle

Previous script in series is:
"nl_00_hdmb_pre.py"

Next script in series is:
"nl_02_join.py"

"""

class PreprocessLoop(object):

    def __init__(self, target_dict, input_df):
        self.target_dict = target_dict
        self.input_df = input_df

    def check_polarity(self, adduct):
        if adduct == '-H' or adduct == '+Cl':
            return 'negative'
        elif adduct == '+H' or adduct == '+Na' or adduct == '+K':
            return 'positive'
        else:
            return 'unknown'


    def count_ids(self, ids):
        if len(ids) == 1:
            return False
        else:
            return True


    def n_loss_only(self, lir):
        if float(lir) == 1:
            return True
        else:
            return False


    def n_loss_wparent(self, lir):
        if float(lir) > 0 and float(lir) < 1:
            return True
        else:
            return False


    def join_nl_searches(self):
        input_df = self.input_df
        input_df = input_df.sort_values(by='hmdb_ids')

        # Annotate polarity
        input_df['polarity'] = input_df['adduct'].apply(lambda row: self.check_polarity(row))

        # Annotate multiple ID
        input_df['num_ids'] = input_df['hmdb_ids'].apply(lambda row: self.count_ids(row))

        # Cludge to fix water loss as three smarts targets:
        input_df['loss_intensity_share_H2Oc'] = input_df['loss_intensity_share_H2O']
        input_df['loss_intensity_share_H2Od'] = input_df['loss_intensity_share_H2O']
        input_df['loss_intensity_share_H2Oe'] = input_df['loss_intensity_share_H2O']

        # Annotate loss-type for each target group
        for long_name, formula in self.target_dict.items():
            n_loss_o = 'n_loss_only_' + str(formula)
            n_loss_wp = 'n_loss_wparent_' + str(formula)
            lir = 'loss_intensity_share_' + str(formula)
            print(lir)
            input_df[n_loss_o] = input_df[lir].apply(lambda row: self.n_loss_only(row))
            input_df[n_loss_wp] = input_df[lir].apply(lambda row: self.n_loss_wparent(row))

        return input_df

class hmdb_rd(object):
    def __init__(self, target_dict):
        self.target_dict = target_dict

    def load_hmdb_df(self, file_mask):
        # input file from: HMDB_rdkit_output_analysis.ipynb
        hmdb_df = pd.read_pickle(file_mask)
        hmdb_df = hmdb_df.rename(columns={'id': 'hmdb_ids'})
        return hmdb_df

    def get_expt_ids(self, input_df):
        # Sort for all unique values in list of list hmdb_ids
        # https://stackoverflow.com/questions/38895856/python-pandas-how-to-compile-all-lists-in-a-column-into-one-unique-list
        hmdb_ids = input_df.hmdb_ids.tolist()
        hmdb_unique = list(set([a for b in hmdb_ids for a in b]))
        return hmdb_unique

    def hmdb_filt(self, theo_id_row, expt_ids):
        for expt_id in expt_ids:
            if expt_id in theo_id_row:
                return True
        else:
            return False

    def hmdb_filter(self, expt_ids, hmdb_df):
        # Filter hmdb_df for id's observed in this experiment

        x_df = hmdb_df

        x_df['in_expt'] = x_df['hmdb_ids'].apply(lambda row: self.hmdb_filt(row, expt_ids))
        hmdb_out = x_df[x_df.in_expt == True]
        return hmdb_out


    def hmdb_sanitize(self, filtered_df):
        #Drop rows with na molecule
        sanitized_df = filtered_df.dropna(subset=['Molecule'], axis=0, how='any')
        return sanitized_df


    def add_targets(self, sanitized_df):
        # Add rdkit filter objects, should be in previous?
        # Handle case of multiple targets (e.g. CDE) SMARTS "or" clause?
        # Implemented in hmdb_structure_parser

        for target, smarts in target_structures.items():
            substruct_object = Chem.MolFromSmarts(smarts)
            target_name = target_loss_formula[target] + '_RDtarget'
            sanitized_df[target_name] = substruct_object
        return sanitized_df


    def target_present(self, structure, target):
        # Can find arbitrary structures
        search_result = structure.HasSubstructMatch(target)
        return search_result


    def cde(self, c, d, e):
        result = bool(c) or bool(d) or bool(e)
        return result


    def targets_present(self, targeted_df):
        # Score for presennce of rd objects in one column as product of two
        # Implemented in hmdb_structure_searcher
        headers = []
        for target, formula in target_loss_formula.items():
            target_name = formula + '_RDtarget'
            headers.append(target_name)

        x_df = targeted_df
        for formula in headers:
            r = formula.split('_')
            res = r[0] + '_Present'
            x_df[res] = x_df.apply(lambda x: self.target_present(x['Molecule'],
                                                                 x[formula]), axis=1)

        x_df['H2Ocde_Present'] = x_df.apply(lambda x: self.cde(x.H2Oc_Present,
                                                                        x.H2Od_Present,
                                                                        x.H2Oe_Present),
                                                                        axis=1)

        return x_df


    def charge(self, molecule):
        charge = Chem.rdmolops.GetFormalCharge(molecule)
        return charge


    def charges(self, searched_df):
        x_df = searched_df
        x_df['z'] = x_df.apply(lambda x: self.charge(x['Molecule']), axis=1)
        return x_df


    def exact_mass(self, molecule):
        exact = Chem.rdMolDescriptors.CalcExactMolWt(molecule)
        return exact


    def exact_masses(self, searched_df):
        x_df = searched_df
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


    def molecure_fp_array(self, molecule):
        fp = self.morgan_fp(molecule)
        bits = self.fp_to_array(fp)
        bits = np.array(bits)
        bits = np.squeeze(bits, axis=None)
        bits = bits.astype(int)
        bits = list(bits)
        return bits


    def fingerprint_1024(self, searched_df):
        # 1024 bit chemical fingerprint, standard settings
        # written up in "hmdb_water_CO2_results_Exp001"
        searched_df['bits'] = searched_df['Molecule'].apply(lambda x: self.molecure_fp_array(x))
        return searched_df


    def expert_fps(self, searched_df):
        # Fingerprint made from "targets_present" bits
        temp_headers = list(target_loss_formula.values())
        headers = []

        for head in temp_headers:
            new_head = head + '_Present'
            headers.append(new_head)

        present_df = searched_df[headers].copy(deep=True)
        #Working!

        list_of_lists = []
        for row in present_df.itertuples():
            expert_fp = np.array(row)
            expert_fp = np.delete(expert_fp, 0)
            expert_fp = list(expert_fp)
            list_of_lists.append(expert_fp)

        searched_df['expert'] = list_of_lists
        return searched_df

    def concat(self, one, two):
        concat = one + two
        return concat


    def fp_1024_experts(self, x_df):
        # Fingerprint concat. of 1024 and expert
        id = 'fp_1024_expert'
        x_df[id] = x_df.apply(lambda x: self.concat(x['bits'], x['expert']), axis=1)
        return x_df


    def bool_fill(selfs, bool):
        if bool is True:
            return True
        elif bool is False:
            return False
        else:
            tf = random.choice([True, False])
            return tf

    def trues(self, x_df):
        id = 'trues'
        x_df[id] = x_df.apply(lambda x: self.bool_fill(True), axis=1)
        return x_df


    def falses(self, x_df):
        id = 'falses'
        x_df[id] = x_df.apply(lambda x: self.bool_fill(False), axis=1)
        return x_df


    def randoms(self, x_df):
        id = 'rando'
        x_df[id] = x_df.apply(lambda x: self.bool_fill('Random'), axis=1)
        return x_df


    def fp_feats(self, x_df):
        # Fingerprint concat. with features
        id = 'fp_feats'
        x_df[id] = x_df.apply(lambda x: self.concat(x['bits'], x['mord_norm']), axis=1)
        return x_df


    def formula(self, mol):
        form = CalcMolFormula(mol)
        return form


    def formulas(self, x_df):
        # Fingerprint concat. with features
        id = 'formula'
        x_df[id] = x_df.apply(lambda x: self.formula(x['Molecule']), axis=1)
        return x_df


    def hmdb_rd_loop(self, input_df):
        # 1) Loads hmdb, 2) finds id's in this expt, 3) calculates various RD-kit things

        hmdb_df = self.load_hmdb_df('hmdb_out_molecule')
        expt_ids = self.get_expt_ids(input_df)
        hmdb_filtered = self.hmdb_filter(expt_ids, hmdb_df)
        hmdb_san = self.hmdb_sanitize(hmdb_filtered)
        hmdb_targeted = self.add_targets(hmdb_san)
        hmdb_searched = self.targets_present(hmdb_targeted)

        hmdb_searched = self.charges(hmdb_searched)
        hmdb_searched = self.exact_masses(hmdb_searched)
        hmdb_searched = self.fingerprint_1024(hmdb_searched)
        hmdb_searched = self.expert_fps(hmdb_searched)
        hmdb_searched = self.fp_1024_experts(hmdb_searched)
        hmdb_searched = self.trues(hmdb_searched)
        hmdb_searched = self.falses(hmdb_searched)
        hmdb_searched = self.randoms(hmdb_searched)

        hmdb_searched = self.mordreds(hmdb_searched)
        hmdb_searched = self.mord_norms(hmdb_searched)
        hmdb_searched = self.fp_feats(hmdb_searched)
        hmdb_searched = self.formulas(hmdb_searched)


        return hmdb_searched


def keys(x):
    y = ['H2Oc_Present', 'H2Od_Present', 'H2Oe_Present',
         'CN_Present', 'NH2_Present', 'COH_Present',
         'CO2H_Present', 'H2O_Present']
    return y


### Body ###
start_time = time.time()

# Setup main loop classes
input_dict = target_loss_formula
input_file = 'en_nl_exptl_stats.pickle'
input_df = pd.read_pickle(input_file)
pre_loop = PreprocessLoop(input_dict, input_df)
db_loop = hmdb_rd(input_dict)

# Run main loops
output_df = pre_loop.join_nl_searches()
hmdb_df = db_loop.hmdb_rd_loop(output_df)

# Fix for cde issue
hmdb_df.rename({'H2Ocde_Present':'H2O_Present'}, axis=1, inplace=True)

# Manual for now, ID of bits in expert
hmdb_df['expert_key'] = hmdb_df.apply(lambda x: keys(x), axis = 1 )

# Output, 148s
output_df.to_pickle('en_nl_01_output_df_exptl.pickle')
hmdb_df.to_pickle('en_nl_01_hmdb_df_exptl.pickle')

print('\nExecuted without error\n')

elapsed_time = time.time() - start_time
print ('Elapsed time:\n')
print (elapsed_time)