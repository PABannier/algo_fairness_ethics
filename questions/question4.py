import numpy as np

from algo_fairness.surrogate.train_lr_surrogate import train_lr_surrogate
from algo_fairness.surrogate.train_tree_surrogate import train_tree_surrogate
from algo_fairness.blackbox.preprocessing import get_preprocessed_data

X = get_preprocessed_data("../data/data_project.xlsx", "CreditRisk (y)")[0]

with open("../outputs/proba_lgb_blackbox.npx", "rb") as infile:
    y_hat = np.load(infile)

train_lr_surrogate(X, y_hat)
train_tree_surrogate(X, y_hat)
