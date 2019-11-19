
"""
This file contains the parameters to be using for filtering the joined dataframe in the script
"nl_03_filter_model"

"""

# trues, falses, rando, mord_norm, fp_feats

search_params = {'models': ['direct', 'random_forest'],
                 'polarities': ['negative', 'positive'],
                 'struct_targets': ['H2Oe'],
                 'obs_loss_types': ['n_loss_wparent'],
                 'theo_predictions': ['expert', 'bits', 'mord_norm', 'fp_feats'],
                 'direct_comp': ['H2O_Present', 'trues', 'falses', 'rando'],
                 'any_fdrs': [0.2 , 0.1 , 0.05],
                 'any_colocalizations': [0, 0.5, 0.75]
                 }

model_params = {'kneighbors': {'n_neighbors': 3},
                'linear_regression': {'max_iter': 10000},
                'naive_bayes': {},
                'decision_tree': {'max_depth': 5, 'random_state': 0},
                'random_forest': {'max_features': 32, 'n_estimators': 100, 'random_state':0},
                'gb_machine': {'random_state': 0, 'max_depth': 3},
                'sv_machine': {'kernel': 'rbf', 'C': 10, 'gamma': 1.0},
                'neural_network': {'solver': 'lbfgs', 'random_state':0}
                }

# Backup!
default_params = {'models': ['decision_tree'],
                 'polarities': ['negative', 'positive'],
                 'struct_targets': ['NH2', 'CN', 'CO2H', 'COH', 'H2Oe'],
                 'obs_loss_types': ['n_loss_only', 'n_loss_wparent'],
                 'theo_predictions': ['expert', 'bits', 'fp_1024_expert'],
                 'any_fdrs': [0.2 , 0.1 , 0.05],
                 'any_colocalizations': [0, 0.5, 0.75, 0.9, 0.95]
                 }

# Currently supported options for selecting above
possible_params = {'models': ['kneighbors',
                              'linear_regression',
                              'naive_bayes',
                              'decision_tree',
                              'random_forest',
                              'gb_machine',
                              'sv_machine',
                              'neural_network'
                              ],
                 'polarities': ['negative', 'positive'],
                 'struct_targets': ['many, see: "structures_to_search_dicts.py"'],
                 'obs_loss_types': ['n_loss_only', 'n_loss_wparent'],
                 'theo_predictions': ['expert', 'bits', 'fp_1024_expert'], # More could be added
                 'any_fdrs': ['float 0-1'],
                 'any_colocalizations': ['float 0-1']
                 }