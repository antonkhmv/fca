import sys
import os
sys.path.insert(0, os.pardir)

import pandas as pd
import optuna
import fca_utils
import argparse


RANDOM_SEED = 100
TARGET = 'price'
DS_NAME = 'housing'


def create_binary_data_config(trial):    
    qcut_cols = {}
    onehot_cols = ["furnishingstatus"]
    factorize_cols = []
    
    if trial.suggest_int("onehot_encode_numeric", 0, 1):
        onehot_cols.extend(['bedrooms', 'bathrooms', 'stories', 'parking'])
    else:
        qcut_cols = {
            "bedrooms": trial.suggest_int("bedrooms", 1, 3),
            "bathrooms": trial.suggest_int("bathrooms", 1, 2),
            "stories": trial.suggest_int("stories", 1, 2),
            "parking": trial.suggest_int("parking", 1, 2),
        }
    
    if trial.suggest_int("factorize_area", 0, 1):
        factorize_cols.extend(["area"])
    else:
        qcut_cols["area"] = trial.suggest_int("area", 1, 10, log=True)  # 0 is possible

    # Value = 1 => remove from config
    for col, value in list(qcut_cols.items()):
        if value == 1:
            del qcut_cols[col]
        
    data_config = {
        'qcut_cols': qcut_cols,
        'onehot_cols': onehot_cols,
        'factorize_cols': factorize_cols,
        'binarize_all': True
    }
    return data_config, None


def create_pattern_data_config(trial):
    
    qcut_cols = {}
    onehot_cols = []
    factorize_cols = []
    categorical_cols = ['furnishingstatus']

    qcut_cols = {
        "area": trial.suggest_int("area", 1, 10, log=True),  # 0 is possiblem
        "bedrooms": trial.suggest_int("bedrooms", 1, 3),
        "bathrooms": trial.suggest_int("bathrooms", 1, 2),
        "stories": trial.suggest_int("stories", 1, 2),
        "parking": trial.suggest_int("parking", 1, 2),
    }

    # Value = 1 => remove from config
    for col, value in list(qcut_cols.items()):
        if value == 1:
            del qcut_cols[col]
        
    data_config = {
        'qcut_cols': qcut_cols,
        'onehot_cols': [],
        'factorize_cols': [],
        'binarize_all': False
    }
    
    return data_config, categorical_cols


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
    