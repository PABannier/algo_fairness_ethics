#!/usr/local/bin/python3

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_lr_surrogate(X, y_hat):
    """
    Trains a Linear Regression white-box surrogate to interpret 
    black-box model predictions

    Args:
        X: Design matrix
        y_hat: Prediction (proba) vector
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y_hat, test_size=0.3)

    model = sm.OLS(y_train, X_train)
    results = model.fit()
    print("==== FITTING RESULTS ====")
    print(results)

    print("\n")

    # Inference
    y_pred = model.predict(X_test)

    # Some performance metrics
    print("===== METRICS =====")
    print("R2: ", r2_score(y_test, y_pred))
    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
