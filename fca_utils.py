import pandas as pd
import numpy as np
import fcalc
import yaml
import optuna

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from collections import Counter


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
        assert (result.dtypes == 'int64').all(), [result.dtypes, parts[0].dtypes]
        nonbinary = []
        for col in result:
            if len(result[col].unique()) > 2:
                nonbinary.append(col)
        if nonbinary:
            result = pd.concat([result.drop(nonbinary, axis=1), pd.get_dummies(result[nonbinary].astype(str))], axis=1)
        result = result.astype(bool)
    
    return result


def cross_validation(config, df, y_target, n_folds):
    config = dict(config)
    df_prepared = prepare_data(df, **config.pop("data_config"))
    X = df_prepared.drop(y_target, axis=1)
    y = df_prepared[y_target]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    classifier = config.pop("classifier")
    cat_names = config.pop('categorical')
    
    if cat_names:
        config['categorical'] = np.where(X.columns.isin(cat_names))[0]

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, y_train = X.loc[train_index], y.loc[train_index]
        X_test, y_test = X.loc[test_index], y.loc[test_index]

        clf = classifier(X_train.values, y_train.to_numpy(), **config)
        clf.predict(X_test.values)
        yield X_train, y_train, X_test, y_test, clf


def run_prediction(trial, config, df, y_target, n_folds):
    f1_score_1 = []
    f1_score_0 = []
    f1_macro = []
    
    for _, _, _, y_test, clf in cross_validation(config, df, y_target, n_folds):
        f1_score_1.append( f1_score(y_test, clf.predictions > 0) )
        f1_score_0.append( f1_score(1 - y_test, clf.predictions <= 0) )
        f1_macro.append( f1_score(y_test, clf.predictions > 0, average='macro') )

    trial.set_user_attr("f1_score_1", f1_score_1)
    trial.set_user_attr("f1_score_0", f1_score_0)
    trial.set_user_attr("f1_macro", f1_macro)

    if np.isnan(f1_macro).any():
        return 0.0
    
    return np.mean(f1_macro)


def calulate_best_intersections(trial, df, y_target, n_folds):
    
    config = dict(trial.user_attrs['config'])
    config['classifier'] = eval(config['classifier'])
    
    pos_intersections = Counter()
    neg_intersections = Counter()
    
    for X_train, y_train, X_test, y_test, clf in cross_validation(config, df, y_target, n_folds):
            
        train_pos = X_train[y_train == True]
        train_neg = X_train[y_train == False]
        
        positive = []
        negative = []

        if config["classifier"] == fcalc.classifier.BinarizedBinaryClassifier:
            for i in range(len(X_test)):
                for k, train, output in zip([0, 1], [train_pos, train_neg], [positive, negative]):
                    for j in range(len(train)):
                        X_inter = tuple(train.columns[X_test.values[i] & train.values[j]])
                        if clf.support[k][0][i][j] >= clf.support[k][1][i][j]: # support > counter
                            output.append(X_inter)

        else:
            categorical = config.get('categorical')
            noncat_names = np.where(~X_train.columns.isin(categorical))[0]
            cat_names = np.where(X_train.columns.isin(categorical))[0]
            
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


def merge(tuples):
    result = {}
    total = 0
    for features, count in tuples:
        for feature in features:
            result[feature] = result.get(feature, 0) + count
            total += count
    return [(key, value / total) for key, value in sorted(result.items(), key=lambda x: -x[1])][:10]


def print_results(best_trial, result_name, intersections):
    print("F1 values for each fold:")
    for name, values in best_trial.user_attrs.items():
        if name.startswith("f1_"):
            print(f"{name} \t {' '.join(f'{value:.3f}' for value in values)}")
    print()
    print(f"Best config for {result_name}:")
    print(yaml.dump(best_trial.user_attrs["config"]))
    print()
    
    positive_intersections = intersections[0].most_common(10)
    negative_intersections = intersections[1].most_common(10)
    
    print("Most important positive intersections:")
    print("\n".join(map(str, positive_intersections)))
    print()
    print("Most important positive features:")
    print("\n".join(map(str, merge(positive_intersections))))
    print()
    print("Most important negative intersections:")
    print("\n".join(map(str, negative_intersections)))
    print()
    print("Most important negative features:")
    print("\n".join(map(str, merge(negative_intersections))))
    print()
    print(f"Best f1_macro score for {result_name}: {best_trial.values[0]:.3f}")

    
def load_study(results_name):
    study = optuna.load_study(
        study_name=f"{results_name}", 
        storage=f"sqlite:///{results_name}.db"
    )
    return study


def show(df, target, study):
    best_intersections = calulate_best_intersections(
        study.best_trial, 
        df=df, 
        y_target=target, 
        n_folds=5
    )

    print_results(study.best_trial, study.study_name, best_intersections)
