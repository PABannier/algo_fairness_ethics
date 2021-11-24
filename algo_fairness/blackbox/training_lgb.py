#!/usr/local/bin/python3

import lightgbm as lgb


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
