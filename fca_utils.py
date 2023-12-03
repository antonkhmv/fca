import pandas as pd
import numpy as np
import fcalc

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from itertools import product
from collections import Counter


class Tracker:
    def __init__(self, mode="max"):
        if mode not in ['min', 'max']:
            raise ValueError(repr(self.mode) + " unknown mode")
        self.mode = mode
        self.values = {None: float({'min': 'inf', 'max': '-inf'}[mode])}
        self.configs = {None: None}
        self.best_index = None
        self.index = 0
    
    def track(self, value, config):
        if self.is_best(value):
            self.best_index = self.index
        self.values[self.index] = value
        self.configs[self.index] = config
        self.index += 1

    def get_best(self):
        return self.values[self.best_index]
        
    def get_best_config(self):
        return self.configs[self.best_index]

    def is_best(self, value):
        if self.mode == 'max':
            return value > self.get_best()
        elif self.mode == 'min':
            return value < self.get_best()


def build_configs(param_grid):
    sets = [[(key, value) for value in values] for key, values in param_grid.items()]
    return list(map(dict, product(*sets)))


def common_prepare(
    df_raw: pd.DataFrame,
    y_target: str,
    bool_columns: list,
    categorical_cols: list,
    ordinal_columns: list,
    bool_subs={"yes": True, "no": False},
):
    
    assert sorted(df_raw.columns.tolist()) == sorted(bool_columns + categorical_cols + ordinal_columns + [y_target])

    df = pd.concat([
        df_raw[[y_target] + ordinal_columns + categorical_cols],
        df_raw[bool_columns].replace(bool_subs),
    ],
    axis=1)
    if len(df[y_target].unique()) > 2:
        df[y_target] = pd.qcut(df[y_target], 2, labels=False)
    return df


def prepare_data(
        df: pd.DataFrame,
        qcut_cols: dict,
        onehot_cols: list,
        factorize_cols: list,
        binarize_all: bool = False
    ):
    df = df.copy()
    columns = list(qcut_cols.keys()) + onehot_cols + factorize_cols
    parts = [df.drop(columns, axis=1)]

    for i, columns in enumerate([qcut_cols, onehot_cols, factorize_cols]):
        if i == 0:
            for column, n in columns.items():
                parts.append(pd.qcut(df[column], n, labels=False, duplicates='drop'))
        elif i == 1:
            if columns:
                parts.append(pd.get_dummies(df[columns].astype(str)).astype(int))
        elif i == 2:
            for column in columns:
                if pd.api.types.is_numeric_dtype(df.dtypes[column]):
                    series = df[column]
                    uinque_values = sorted(series.unique())
                    values = pd.Series(np.arange(len(uinque_values)), index=uinque_values, name=series.name).loc[series]
                    values.index = series.index
                    parts.append(values)
                else:
                    values, _ = pd.factorize(series)
                    parts.append(pd.Series(values, name=column))

    result = pd.concat(parts, axis=1)
    
    if binarize_all:
        assert (result.dtypes == 'int64').all(), result.dtypes
        nonbinary = []
        for col in result:
            if len(result[col].unique()) > 2:
                nonbinary.append(col)
        if nonbinary:
            result = pd.concat([result.drop(nonbinary, axis=1), pd.get_dummies(result[nonbinary].astype(str))], axis=1)
        result = result.astype(bool)
    
    return result


def iter_folds(df, classifier, y_target, n_folds, config, method, **kwargs):
    
    df_prepared = prepare_data(df, **config)
    X = df_prepared.drop(y_target, axis=1)
    y = df_prepared[y_target]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cat_names = kwargs.get('categorical', None)
    if cat_names:
        kwargs['categorical'] = np.arange(X.shape[1])[X.columns.isin(cat_names)]

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, y_train = X.loc[train_index], y.loc[train_index]
        X_test, y_test = X.loc[test_index], y.loc[test_index]

        clf = classifier(X_train.values, y_train.to_numpy(), method=method, **kwargs)
        clf.predict(X_test.values)
        yield X_train, y_train, X_test, y_test, clf


def calulate_best_intersections(df, classifier, y_target, n_folds, config, method, **kwargs):
    pos_intersections = Counter()
    neg_intersections = Counter()
    
    for X_train, y_train, X_test, y_test, clf in iter_folds(df, classifier, y_target, n_folds, config, method, **kwargs):
            
        train_pos = X_train[y_train == True]
        train_neg = X_train[y_train == False]
        
        positive = []
        negative = []

        if classifier == fcalc.classifier.BinarizedBinaryClassifier:
            for i in range(len(X_test)):
                for k, train, output in zip([0, 1], [train_pos, train_neg], [positive, negative]):
                    for j in range(len(train)):
                        X_inter = tuple(train.columns[X_test.values[i] & train.values[j]])
                        if clf.support[k][0][i][j] >= clf.support[k][1][i][j]: # support > counter
                            output.append(X_inter)

        else:
            categorical = kwargs.get('categorical')
            noncat_names = np.arange(X_train.shape[1])[~X_train.columns.isin(categorical)]
            cat_names = np.arange(X_train.shape[1])[X_train.columns.isin(categorical)]
            
            # Equality tolerance
            X_tolerance = 0.1 * X_train.iloc[:, noncat_names].std()

            for i in range(len(X_test)):
                for k, train, output in zip([0, 1], [train_pos, train_neg], [positive, negative]):
                    for j in range(len(train)):
                        X_inter = tuple()
                        
                        if len(noncat_names) > 0:
                            intsec_num = np.abs(X_test.values[i, noncat_names] - train.values[j, noncat_names]) < X_tolerance
                            X_inter += tuple(train.iloc[:, noncat_names].columns[intsec_num])

                        if len(cat_names) > 0:
                            intsec_cat = fcalc.patterns.CategoricalPattern(X_test.values[i, cat_names], train.values[j, cat_names])
                            X_inter += tuple(train.iloc[:, cat_names].columns[intsec_cat.mask])

                        if clf.support[k][0][i][j] >= clf.support[k][1][i][j]: # support > counter
                            output.append(X_inter)
                        
        pos_intersections.update(Counter(positive))
        neg_intersections.update(Counter(negative))
                                
    return pos_intersections, neg_intersections


def grid_search(df, classifier, y_target, n_folds, methods, configs, **kwargs):
    results = []
    params = list(product(methods, configs))
    print(f"Fitting {len(params)} configurations")
    
    tracker = Tracker()
    
    for method, config in params:
        f1_score_1 = []
        f1_score_0 = []
        f1_macro = []
            
        for _, _, _, y_test, clf in iter_folds(df, classifier, y_target, n_folds, config, method, **kwargs):
            f1_score_1.append( f1_score(y_test, clf.predictions > 0) )
            f1_score_0.append( f1_score(1 - y_test, clf.predictions <= 0) )
            f1_macro.append( f1_score(y_test, clf.predictions > 0, average='macro') )

        metric = np.mean(f1_macro)
        tracker.track(metric, (method, config))
        
        print(
            tracker.index,
            f"{classifier=}",
            f"{config=}", 
            f"{method=}",
            f"f1_macro (mean)={metric}",
            f"f1_cls_1={f1_score_1}",
            f"f1_cls_0={f1_score_0}",
            f"f1_macro={f1_macro}",
            "",
            sep='\n'
        )
    method, config = tracker.get_best_config()
    intersections = calulate_best_intersections(df, classifier, y_target, n_folds, config, method, **kwargs)
    return tracker.get_best_config(), intersections, tracker.get_best()
