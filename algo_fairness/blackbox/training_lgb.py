#!/usr/local/bin/python3

import lightgbm as lgb


def fit_lgbm(train, val, cat_features=None, num_rounds=1500, lr=0.1, bf=0.1):
    """
    Train LightGBM classification model
    Optimizes AUC metric
    """
    X_train, y_train = train
    X_valid, y_valid = val

    print("training LGB:")
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

    # predictions
    y_pred_valid = model.predict(X_valid)
    return model, y_pred_valid
