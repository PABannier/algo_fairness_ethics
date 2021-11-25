import joblib

import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from algo_fairness.blackbox.preprocessing import get_preprocessed_data
from algo_fairness.blackbox.training_lgb import fit_lgbm

OUT_PATH = "../outputs/proba_lgb_blackbox.npx"

# Settings
folds = 5
kf = StratifiedKFold(n_splits=folds)

# Loading data
X, y, _, categorical_features = get_preprocessed_data(
    "../data/data_project.xlsx", "CreditRisk (y)"
)

# OOF model validation
y_oof = np.zeros(X.shape[0])

for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    train_data = X.iloc[train_idx, :], y[train_idx]
    valid_data = X.iloc[valid_idx, :], y[valid_idx]

    model, y_pred_valid = fit_lgbm(
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

with open(OUT_PATH, "wb") as outfile:
    np.save(outfile, y_oof)
print("Proba vector saved!")

# Train-test-split with models

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)
model, y_pred = fit_lgbm(
    (X_train, y_train),
    (X_test, y_test),
    cat_features=categorical_features,
    num_rounds=1000,
    lr=0.05,
    bf=0.7,
)

with open(f"../outputs/lgb_bb_model.pkl", "wb") as outfile:
    joblib.dump(model, outfile)
