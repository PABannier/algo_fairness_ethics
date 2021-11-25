import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from algo_fairness.blackbox.preprocessing import get_preprocessed_data

OUT_PATH = "../outputs/fpdp/"

if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)


def plot_fpdp(bins, p_vals, var_name, out_path=None):
    fig, ax = plt.subplots(1, 1)
    plt.plot(bins, p_vals)
    ax.axhline(y=0.05, c="red")

    if out_path:
        plt.savefig(out_path + f"{var_name}.jpeg")

    plt.title(var_name)
    plt.show()


def gen_fpdp_for_numerical_var(df, var_name):
    min_val, max_val = df[var_name].min(), df[var_name].max()
    intervals = np.linspace(min_val, max_val, 10)

    p_vals = np.zeros_like(intervals)

    for i in range(len(intervals)):
        if i < len(intervals) - 1:
            min_v, max_v = intervals[i], intervals[i + 1]
            sub_df = df[(df[var_name] >= min_v) & (df[var_name] <= max_v)]
            contingency_table = pd.crosstab(
                sub_df["Gender"].values, sub_df["y_hat"].values
            )
            try:
                p_val = chi2_contingency(contingency_table)[1]
            except:
                p_val = 1
            p_vals[i] = p_val

    return intervals, p_vals


def gen_fpdp_for_categorical_var(df, var_name):
    unique_vals = df[var_name].unique()

    p_vals = np.zeros_like(unique_vals)

    for i, unique_val in enumerate(unique_vals):
        sub_df = df[df[var_name] == unique_val]
        contingency_table = pd.crosstab(
            sub_df["Gender"].values, sub_df["y_hat"].values
        )
        p_val = chi2_contingency(contingency_table)[1]
        p_vals[i] = p_val

    return unique_vals, p_vals


X, y, numerical_features, categorical_features = get_preprocessed_data(
    "../data/data_project.xlsx", "CreditRisk (y)"
)

with open("../outputs/proba_lgb_blackbox.npx", "rb") as infile:
    y_hat = np.load(infile)

# Building tables
df = X.copy()
df["y_hat"] = y_hat
df["y"] = y

# Iterating over numerical features

for numerical_feature in numerical_features:
    bins, p_vals = gen_fpdp_for_numerical_var(df, numerical_feature)
    plot_fpdp(bins, p_vals, numerical_feature, OUT_PATH)
