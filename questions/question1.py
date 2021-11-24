from algo_fairness.surrogate.train_lr_surrogate import train_lr_surrogate
from algo_fairness.blackbox.preprocessing import get_preprocessed_data
from algo_fairness.surrogate.train_tree_surrogate import train_tree_surrogate

X, y_hat, _, _ = get_preprocessed_data("../data/data_project.xlsx", "y_hat")
train_lr_surrogate(X, y_hat)
train_tree_surrogate(X, y_hat)
