#!/usr/local/bin/python3

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from dtreeviz.trees import dtreeviz


def train_tree_surrogate(X, y_hat, random_state=0):
    """
    Trains a Decision Tree Classifier surrogate to interpret
    black-box model predictions

    Args:
        X: Design matrix
        y_hat: Prediction (proba) vector
        random_state (int, optional): Random state for seeding. Defaults to 0.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_hat, test_size=0.3, random_state=random_state
    )

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    viz = dtreeviz(model, X_train, y_train, target_name="CreditRisk")
    viz.view()

    # Inference
    y_pred = model.predict(X_test)

    # Some performance metrics
    print("==== METRICS ====")
    print("R2: ", r2_score(y_test, y_pred))
    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
