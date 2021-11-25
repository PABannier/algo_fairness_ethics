import joblib

from sklearn.model_selection import train_test_split

from algo_fairness.blackbox.preprocessing import get_preprocessed_data
from algo_fairness.tools.shap import shap_viz

OUT_DIR = "../outputs/shap.jpeg"

X, y, numerical_features, categorical_features = get_preprocessed_data(
    "../data/data_project.xlsx", "CreditRisk (y)"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

with open("../outputs/lgb_bb_model.pkl", "rb") as infile:
    model = joblib.load(infile)

shap_viz(model, X_train, 3, "waterfall")
