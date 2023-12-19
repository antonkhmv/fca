import sys
import os
sys.path.insert(0, os.pardir)

import pandas as pd
import optuna
import fca_utils
import argparse


mobile_target_variable = "price_range"

high_variance_cols = ['ram', 'px_width', 'battery_power', 'px_height']
medium_variance_cols = ['int_memory', 'mobile_wt']
low_variance_cols = ['fc', 'n_cores', 'pc', 'sc_h', 'sc_w', 'talk_time', 'clock_speed', 'm_dep']
numeric_cols = high_variance_cols + medium_variance_cols + low_variance_cols

RANDOM_SEED = 100
TARGET = 'price_range'
DS_NAME = 'mobile'


def create_binary_data_config(trial):
    onehot_cols = [] 
    factorize_cols = []
    qcut_cols = {}

    options = [2, 3, 4, 10, None]
    n_cuts = options[trial.suggest_int("numeric_cols", 0, len(options) - 1)]
    
    if n_cuts is not None:
        qcut_cols = dict.fromkeys(numeric_cols, n_cuts)
            
    data_config = {
        'qcut_cols': qcut_cols,
        'onehot_cols': onehot_cols,
        'factorize_cols': factorize_cols,
        'binarize_all': True
    }
    return data_config, None



def create_pattern_data_config(trial):
    onehot_cols = [] 
    factorize_cols = []
    qcut_cols = {}
    
    options = [2, 3, 4, 10, None]
    n_cuts = options[trial.suggest_int("numeric_cols", 0, len(options) - 1)]
    
    if n_cuts is not None:
        qcut_cols = dict.fromkeys(numeric_cols, n_cuts)
            
    data_config = {
        'qcut_cols': qcut_cols,
        'onehot_cols': onehot_cols,
        'factorize_cols': factorize_cols,
        'binarize_all': False
    }
    return data_config, []


def create_objective(df, y_target, classifier, create_data_config):
    
    def objective(trial):
        method = trial.suggest_categorical("method", [
            'standard',
            'standard-support',
            'ratio-support'
        ])
        
        data_config, categorical = create_data_config(trial)
        
        if method == 'ratio-support':
            _alpha = trial.suggest_float("alpha-ratio", 0.5, 5.0, log=True)
        else:
            _alpha = trial.suggest_float("alpha-support", 0.05, 0.95, log=True)
        
        config = {
            "classifier": classifier,
            "alpha": _alpha,
            "method": method,
            "categorical": categorical,
            "data_config": data_config
        }
        
        trial.set_user_attr("config", config)    
        config["classifier"] = eval(config["classifier"])
        return fca_utils.run_prediction(trial, config, df, y_target=y_target, n_folds=5)
    
    return objective


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--type", type=str)
    parser.add_argument("--n_trials", type=int, default=50)
    args = parser.parse_args()
    
    assert args.type in ['binary', 'pattern']
    
    df = pd.read_csv("data.csv")
    
    study_name = f"{DS_NAME}_{args.type}"
    storage = f"sqlite:///{study_name}.db"
    
    if args.restart:
        file_to_rm = os.path.split(storage)[-1]
        if os.path.isfile(file_to_rm):
            os.remove(file_to_rm)
        sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
        study = optuna.create_study(
            study_name=study_name, 
            storage=storage, 
            sampler=sampler,
            direction='maximize'
        )
    else:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
    
        if args.type == "binary":
            study.optimize(
                create_objective(
                    df, TARGET,
                    classifier="fcalc.classifier.BinarizedBinaryClassifier", 
                    create_data_config=create_binary_data_config
                ),
                n_trials=args.n_trials
            )
        else:
            study.optimize(
                create_objective(
                    df, TARGET,
                    classifier="fcalc.classifier.PatternBinaryClassifier", 
                    create_data_config=create_pattern_data_config
                ),
                n_trials=args.n_trials
            )
    