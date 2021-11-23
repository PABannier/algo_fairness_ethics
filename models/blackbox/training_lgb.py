#!/usr/local/bin/python3

import numpy as np
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from preprocessing import get_preprocessed_data


def fit_lgbm(train, val, cat_features=None, num_rounds=1500, lr=0.1, bf=0.1):
    """
    Train LightGBM classification model
    Optimizes AUC metric
    """
    X_train, y_train = train
    X_valid, y_valid = val
    metric = "auc"

    params = {
        "num_leaves": 31,
        "objective": "binary",
        "learning_rate": lr,
        "boosting": "gbdt",
        "bagging_freq": 5,
        "bagging_fraction": bf,
        "feature_fraction": 0.9,
        "metric": metric,
    }

    early_stop = 20
    verbose_eval = 20

    d_train = lgb.Dataset(
        X_train, label=y_train, categorical_feature=cat_features
    )
    d_valid = lgb.Dataset(
        X_valid, label=y_valid, categorical_feature=cat_features
    )
    watchlist = [d_train, d_valid]

    print("training LGB:")
    model = lgb.train(
        params,
        train_set=d_train,
        num_boost_round=num_rounds,
        valid_sets=watchlist,
        verbose_eval=verbose_eval,
        early_stopping_rounds=early_stop,
    )

    # predictions
    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

    print("best_score", model.best_score)
    log = {
        "train/auc": model.best_score["training"]["auc"],
        "valid/auc": model.best_score["valid_1"]["auc"],
    }
    return model, y_pred_valid, log


if __name__ == "__main__":
    # Settings
    folds = 5
    kf = StratifiedKFold(n_splits=folds)

    # Loading data
    X, y, _, categorical_features, _ = get_preprocessed_data()

    # Fitting
    y_oof = np.zeros(X.shape[0])

    for train_idx, valid_idx in kf.split(X, y):
        train_data = X.iloc[train_idx, :], y[train_idx]
        valid_data = X.iloc[valid_idx, :], y[valid_idx]

        model, y_pred_valid, log = fit_lgbm(
            train_data,
            valid_data,
            cat_features=categorical_features,
            num_rounds=1000,
            lr=0.05,
            bf=0.7,
        )
        y_oof[valid_idx] = y_pred_valid

    print("OOF AUC score: ", roc_auc_score(y, y_oof))
    print("OOF F1 score:", f1_score(y, y_oof > 0.5))

    with open("../../outputs/proba_lgb_blackbox.npx", "wb") as outfile:
        np.save(outfile, y_oof)
    print("Proba vector saved!")
