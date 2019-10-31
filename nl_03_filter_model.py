import pandas as pd
from structures_to_search_dicts import target_structures, target_loss_formula
import time
from nl_03_params import search_params
from nl_03_params import model_params
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from sklearn.tree import export_graphviz







import rdkit.Chem as Chem
from rdkit.Chem import rdFMCS
from scipy import stats


import pandas
from pandas import DataFrame as df
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from structures_to_search_dicts import target_nl_mass, target_short_codes
from rdkit.Chem import rdmolops
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import AllChem
import rdkit.DataStructs
from rdkit.Chem import Fragments

"""
nl_02_join.py:

Purpose:
This script is designed to filter neutral loss search results then use them for model building 
and testing.

Steps include:
1) Load joined data.
2) Generate combinatorial filters: fdr, coloc, polarity, true result (loss obs), 
prediction (expert/fp), model to test.  Dict of lists.     
3) Loop to filter, split test/train, build model, test model.
4) Export and visualize results.
    -What are chemical FP bits?
    -What are expert bits?

Previous script ion series is:
"nl_02_join.py"
"""

class CombiFilters(object):
    def __init__(self, search_params):
        self.search_params = search_params


    def main_loop(self):
        # Initialize with columns!
        filter_df = pd.DataFrame(columns=['n',
                                          'model',
                                          'polarity',
                                          'struct_target',
                                          'obs_loss_type',
                                          'theo_prediction',
                                          'fdr',
                                          'coloc'])

        x = self.search_params

        n=0
        for model in x['models']:
            for polarity in x['polarities']:
                for struct_target in x['struct_targets']:
                    for obs_loss_type in x['obs_loss_types']:
                        for theo_prediction in x['theo_predictions']:
                            for fdr in x['any_fdrs']:
                                for coloc in x['any_colocalizations']:
                                    n +=1
                                    row = {'n': n,
                                           'model': model,
                                           'polarity': polarity,
                                           'struct_target': struct_target,
                                           'obs_loss_type': obs_loss_type,
                                           'theo_prediction': theo_prediction,
                                           'any_fdr': fdr,
                                           'any_coloc': coloc
                                           }
                                    filter_df = filter_df.append(row, ignore_index=True)

        return filter_df


class BuildTest(object):
    def __init__(self, join_df, out_list):
        self.join_df = join_df
        self.out_list = out_list


    def column_finder(self, column_list, target_string):
        out_columns = []
        for column in column_list:
            if target_string in column:
                out_columns.append(column)
        return(out_columns)


    def any_fdr(self, df_row):
        column_list = self.column_finder(list(df_row.index), 'fdr')
        best_fdr = df_row[column_list].min()
        best_fdr = float(best_fdr)
        return best_fdr


    def any_coloc(self, df_row):
        column_list = self.column_finder(list(df_row.index), 'colocalization')
        best_coloc = df_row[column_list].max()
        best_coloc = float(best_coloc)
        return best_coloc


    def join_update(self, join_df):
        #Add 'any fdr' to filter by, using best in case of parent and loss.
        join_df['any_fdr'] = join_df.apply(self.any_fdr, axis=1)

        # Add 'any coloc' to filter by, using best in case of parent and loss.
        join_df['any_coloc'] = join_df.apply(self.any_coloc, axis=1)

        join_df.astype({'any_fdr': 'float','any_coloc': 'float'})
        return join_df


    def build_columns(self, row):
        obs = row.obs_loss_type + '_' + row.struct_target
        theo = row.theo_prediction
        columns = ['any_fdr', 'any_coloc', 'polarity', obs, theo]
        return columns


    def filter_3x(self, x_df, filter):
        pol = str(filter.polarity)
        fdr = float(filter.any_fdr)
        coloc = float(filter.any_coloc)

        x_df = x_df[(x_df['any_fdr'] <= fdr) &
                    (x_df['any_coloc'] >= coloc) &
                    (x_df['polarity'] == pol)
                    ]
        return x_df


    def select_train(self, model, x_train, y_train):
        # model_params loaded from "nl_03_params.py"
        v = model_params

        if model is 'kneighbors':
            x = model_params['kneighbors']['n_neighbors']
            return KNeighborsClassifier(n_neighbors=x).fit(x_train, y_train)

        elif model is 'linear_regression':
            x = model_params['linear_regression']['max_iter']
            return LinearSVC(max_iter=x).fit(x_train, y_train)

        elif model is 'naive_bayes':
            x = None
            return BernoulliNB().fit(x_train, y_train)

        elif model is 'decision_tree':
            x = model_params['decision_tree']['max_depth']
            y = model_params['decision_tree']['random_state']
            return DecisionTreeClassifier(max_depth=x,
                                          random_state=y).fit(x_train, y_train)

        elif model is 'random_forest':
            # Error if features < max features
            x = model_params['random_forest']['max_features']
            y = model_params['random_forest']['n_estimators']
            z = model_params['random_forest']['random_state']
            return RandomForestClassifier(max_features=7,
                                          n_estimators=y,
                                          random_state=z).fit(x_train, y_train)

        elif model is 'gb_machine':
            x = model_params['gb_machine']['random_state']
            y = model_params['gb_machine']['max_depth']
            return GradientBoostingClassifier(random_state=x,
                                              max_depth=y).fit(x_train, y_train)

        elif model is 'sv_machine':
            x = model_params['sv_machine']['kernel']
            y = model_params['sv_machine']['C']
            z = model_params['sv_machine']['gamma']
            return SVC(kernel=x, C=y, gamma=z).fit(x_train, y_train)

        elif model is 'neural_network':
            x = model_params['neural_network']['solver']
            y = model_params['neural_network']['random_state']
            return MLPClassifier(solver=x, random_state=y).fit(x_train, y_train)

        else:
            exit('Model_unknown')


    def filter_loop(self, curr_filter_series):
        filter = curr_filter_series
        print(filter.n)
        print('\n')
        print(filter)

        # Build columns and df
        col_f_filter = self.build_columns(filter)

        current_df = self.join_df[col_f_filter].copy(deep=True)

        # Filter on fdr, coloc, and polarity
        current_df = self.filter_3x(current_df, filter)

        # nans should,be gone and types should be consistent...
        obs_y = np.array(current_df.iloc[:,-2].astype('bool'))
        current_df['out_x'] = current_df.iloc[:, -1].apply(list)
        theo_x = np.array(list(current_df['out_x'])).astype(bool)

        # Split and get counts
        x_train, x_test, y_train, y_test = train_test_split(theo_x, obs_y, random_state=0)
        train_l = y_train.shape[0]
        test_l = y_test.shape[0]
        size = (train_l, test_l, train_l + test_l)

        # Select and train model
        model = self.select_train(filter.model, x_train, y_train)

        # Test model
        accuracy_train = model.score(x_train, y_train)
        accuracy_test = model.score(x_test, y_test)

        y_predict = model.predict(x_test)

        # Add values to filter and return
        filter['n_train_test_total'] = size
        filter['n_true_train'] = np.count_nonzero(y_train)
        filter['n_true_test'] = np.count_nonzero(y_test)
        filter['accuracy_train'] = accuracy_train
        filter['accuracy_test'] = accuracy_test

        '''
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
        filter['tp'] = tp
        filter['fp'] = fp
        filter['tn'] = tn
        filter['fn'] = fn

        # https://en.wikipedia.org/wiki/Confusion_matrix
        filter['tpr'] = tp / (tp + fn)
        filter['tnr'] = tn / (tn + fp)
        filter['fnr'] = fn / (fn + tp)
        filter['fpr'] = fp / (fp + tn)
        filter['ppv'] = tp / (tp + fp)
        filter['npv'] = tn / (tn + fn)
        filter['fdr'] = fp / (fp + tp)
        '''

        self.out_list.append(filter)

        return filter.n


    def main_loop(self, filters):
        self.join_df = self.join_update(self.join_df)

        # Use apply to vectorize filtering steps!
        filters['model_results'] = filters.apply(self.filter_loop, axis=1)

        return


class ExportViz(object):
    def __init__(self, join_df):
        self.join_df = join_df


    def main_loop(self, model_df):
        pass

def tidy(df):
    df = df.drop(columns=['fdr', 'coloc'])

    return df

### Body ###
start_time = time.time()

# Setup classes
formula_dict = target_loss_formula
input_file = 'nl_02_join_final.pickle'
join_df = pd.read_pickle(input_file)
out_list = []

find_filters = CombiFilters(search_params)
setup_models = BuildTest(join_df, out_list)
print_results = ExportViz(join_df)

# Run main loop t=61s
filters = find_filters.main_loop()
model_df = setup_models.main_loop(filters)
#results = print_results.main_loop(model_df)

out_df = pd.DataFrame(out_list)
out_df = tidy(out_df)
out_df.to_pickle('nl_03_water_both.pickle')

print('\nExecuted without error\n')
elapsed_time = time.time() - start_time
print ('Elapsed time:\n')
print (elapsed_time)