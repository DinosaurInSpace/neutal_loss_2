import pandas as pd
from pandas import DataFrame as df
import numpy as np

from structures_to_search_dicts import target_structures, target_loss_formula
import time
from nl_03_params import search_params
from nl_03_params import model_params

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import argparse

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

Next script in series is:
"tbd"

Example command line:

python nl_03_filter_model.py --i en_nl_exptl_stats_output_02.pickle

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
                                          'prediction',
                                          'any_fdr',
                                          'any_coloc'])

        x = self.search_params

        n=0
        for model in x['models']:
            for polarity in x['polarities']:
                for struct_target in x['struct_targets']:
                    for obs_loss_type in x['obs_loss_types']:
                        if model is 'direct':
                            y_pred = 'direct_comp'
                        elif model != 'direct':
                            y_pred = 'theo_predictions'
                        for y in x[y_pred]:
                            for fdr in x['any_fdrs']:
                                for coloc in x['any_colocalizations']:
                                    n +=1
                                    row = {'n': n,
                                           'model': model,
                                           'polarity': polarity,
                                           'struct_target': struct_target,
                                           'obs_loss_type': obs_loss_type,
                                           'prediction': y,
                                           'any_fdr': fdr,
                                           'any_coloc': coloc
                                           }
                                    filter_df = filter_df.append(row, ignore_index=True)

        filter_df = filter_df.set_index(['n'])
        return filter_df


class BuildTest(object):
    def __init__(self, join_df, out_list):
        self.join_df = join_df
        self.out_list = out_list


    def any(self, df_row, target, co_or_fdr):
        # Searches for best fdr or coloc per row, only considering
        # parent and NL under investigation
        if len(target) == 4 and target[0:3] == 'H2O':
            target = 'H2O'
        if co_or_fdr == 'fdr':
            nl = 'fdr_' + target
            return(float(min(df_row['fdr'], df_row[nl])))
        elif co_or_fdr == 'colocalization':
            nl = 'colocalization_' + target
            return (float(df_row[nl]))


    def join_update(self, join_df, target):
        #Add 'any fdr' to filter by, using best in case of parent and loss.
        join_df['any_fdr'] = join_df.apply((lambda x: self.any(x,
                                                               target,
                                                               'fdr')),
                                                                axis=1)

        # Add 'any coloc' to filter by, using best in case of parent and loss.
        join_df['any_coloc'] = join_df.apply((lambda x: self.any(x,
                                                               target,
                                                               'colocalization')),
                                                                axis=1)

        join_df.astype({'any_fdr': 'float','any_coloc': 'float'})
        return join_df


    def build_columns(self, row):
        obs = row.obs_loss_type + '_' + row.struct_target
        theo = row.prediction
        columns = ['any_fdr', 'any_coloc', 'polarity', obs, theo]
        return columns


    def filter_3x(self, x_df, filter):
        pol = str(filter.polarity)
        fdr = float(filter.any_fdr)
        coloc = float(filter.any_coloc)

        # Coloc = 0 include as well!
        x_df = x_df[(x_df['any_fdr'] <= fdr) &
                    ((x_df['any_coloc'] >= coloc) |
                      (x_df['any_coloc'] == 0)) &
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

    def confuse(self, obs_y, theo_y):
        # Copy from confuse ipynb
        con = confusion_matrix(list(obs_y), list(theo_y))
        if con.shape == (1, 1):
            return [np.nan, np.nan, np.nan, np.nan]

        elif con.shape == (2, 2):
            tn, fp, fn, tp = con.ravel()
            sens = tpr = tp / (tp + fn)
            spec = tnr = tn / (tn + fp)
            f1 = (2 * tp) / (2 * tp + fp + fn)
            acc = (tp + tn) / (tp + tn + fp + fn)
            return [sens, spec, f1, acc]

        else:
            exit(1)


    def filter_loop(self, filter):
        # filter is current filter from apply
        print(filter.index)
        print('\n')
        print(filter)

        # Build columns and df
        col_f_filter = self.build_columns(filter)
        current_df = self.join_df[col_f_filter].copy(deep=True)

        # Filter on fdr, coloc, and polarity
        current_df = self.filter_3x(current_df, filter)

        # nans should,be gone and types should be consistent...
        obs_y = np.array(current_df.iloc[:,-2]) #.astype('bool'))
        theo_x = np.array(current_df.iloc[:, -1]) #.astype('bool'))
        #current_df['out_x'] = current_df.iloc[:, -1].apply(list)
        #theo_x = np.array(list(current_df['out_x'])).astype(bool)

        # No fancy stuff needed to directly compare two columns:

        if type(theo_x[0]).__module__ != np.__name__ and type(theo_x[0]) != list:
            results = self.confuse(obs_y, theo_x)
            size = len(obs_y)
            t_size = sum(obs_y)
            accuracy_test = results[3]
            accuracy_train = accuracy_test
            specificity = results[0]
            sensitivity = results[1]
            f1 = results[2]

        else:
            obs_y = list(obs_y)
            theo_x = list(theo_x)
            print('complicated')
            x_train, x_test, y_train, y_test = train_test_split(theo_x, obs_y, random_state=0)
            train_l = len(y_train)
            test_l = len(y_test)
            size = (train_l, test_l, train_l + test_l)
            t_size = (np.count_nonzero(y_train),
                      np.count_nonzero(y_test),
                      np.count_nonzero(y_train + y_test))

            # Select and train model
            model = self.select_train(filter.model, x_train, y_train)

            # Test model
            accuracy_train = model.score(x_train, y_train)
            accuracy_test = model.score(x_test, y_test)

            y_predict = model.predict(x_test)
            results = self.confuse(y_test, y_predict)
            specificity = results[0]
            sensitivity = results[1]
            f1 = results[2]

        # Add values to filter and return
        filter['n_train_test_total'] = size
        filter['n_train_test_true'] = t_size
        filter['accuracy_train'] = accuracy_train
        filter['accuracy_test'] = accuracy_test
        filter['specificity'] = specificity
        filter['sensitivity'] = sensitivity
        filter['f1'] = f1

        self.out_list.append(filter)
        return filter.index


    def main_loop(self, filters, target):
        # Creates 'any_fdr' and 'any_coloc' columns
        self.join_df = self.join_update(self.join_df, target)

        # Use apply to vectorize filtering steps!
        filters['model_results'] = filters.apply(self.filter_loop, axis=1)

        return


class ExportViz(object):
    # To write
    def __init__(self, join_df):
        self.join_df = join_df


    def main_loop(self, model_df):
        pass

        return df

### Body ###
start_time = time.time()

# Setup input files
parser = argparse.ArgumentParser(description='')
parser.add_argument("--i", default=None, type=str, help="nl_02_output_02.pickle")
args = parser.parse_args()
input_file = args.i


# Setup classes
formula_dict = target_loss_formula
join_df = pd.read_pickle(input_file)
out_list = []

find_filters = CombiFilters(search_params)
setup_models = BuildTest(join_df, out_list)
#print_results = ExportViz(join_df)

# Run main loop t=61s
filters = find_filters.main_loop()

for target in filters.struct_target.unique():
    current_filters = filters[filters.struct_target == target]
    model_df = setup_models.main_loop(current_filters, target)
    #results = print_results.main_loop(model_df)

# Clean-up and export
out_df = pd.DataFrame(out_list)

out_stub = input_file.split('_output_02')[0]
out_file = out_stub + '_output_03.pickle'
out_df.to_pickle(out_file)

elapsed_time = time.time() - start_time
print ('Elapsed time:\n')
print (elapsed_time)
print('\nExecuted without error\n')
print(out_file)

