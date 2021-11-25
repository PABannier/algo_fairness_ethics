import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from algo_fairness.blackbox.preprocessing import get_preprocessed_data


def run_chi2_test(df):
    contingency_table = pd.crosstab(df["Gender"].values, df["y_hat"].values)
    return chi2_contingency(contingency_table)


X, y, numerical_features, categorical_features = get_preprocessed_data(
    "../data/data_project.xlsx", "CreditRisk (y)"
)

with open("../outputs/proba_lgb_blackbox.npx", "rb") as infile:
    y_hat = np.load(infile)

# Building tables
df = X.copy()
df["y_hat"] = y_hat
df["y"] = y

# Statistical parity
p_val_0 = run_chi2_test(df)[1]

print("======= STATISTICAL PARITY ========")
print("p-value: ", p_val_0)

print("\n")

# Conditional statistical parity
sub_df_1 = df[df["Group"] == 0]
sub_df_2 = df[df["Group"] == 1]

p_val_1 = run_chi2_test(sub_df_1)[1]
p_val_2 = run_chi2_test(sub_df_2)[1]

print("======== CONDITIONAL STATISTICAL PARITY (p-values) ========")
print("Group 0: ", p_val_1)
print("Group 1: ", p_val_2)

print("\n")

# Equalized odds
sub_df_3 = df[df["y"] == 0]
sub_df_4 = df[df["y"] == 1]

p_val_3 = run_chi2_test(sub_df_3)[1]
p_val_4 = run_chi2_test(sub_df_4)[1]

print("========= EQUALIZED ODDS (p-values) =========")
print("Y=0: ", p_val_3)
print("Y=1:", p_val_4)

print("\n")
