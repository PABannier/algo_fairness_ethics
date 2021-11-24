import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from algo_fairness.blackbox.preprocessing import get_preprocessed_data
from algo_fairness.blackbox.training_lgb import fit_lgbm

OUT_PATH = "../outputs/proba_lgb_blackbox.npx"

# Settings
folds = 5
kf = StratifiedKFold(n_splits=folds)

# Loading data
X, y, _, categorical_features = get_preprocessed_data("CreditRisk (y)")

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

with open(OUT_PATH, "wb") as outfile:
    np.save(outfile, y_oof)
print("Proba vector saved!")
