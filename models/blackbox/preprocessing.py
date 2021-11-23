#!/usr/local/bin/python3

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def get_preprocessed_data(target, scaling):
    """
    Preprocessing data with label encoding the categorical features
    Args: 
        - target (string): target variable
        - scaling (boolean): whether to use Standard Scaling or not
    """
    
    df = pd.read_excel("../../data/data_project.xlsx")

    numerical_features = [
        "CreditDuration",
        "CreditAmount",
        "InstallmentRate",
        "Age",
        "NumberOfCredits",
    ]

    categorical_features = [
        "CreditHistory",
        "EmploymentDuration",
        "Housing",
        "Purpose",
        "Savings",
        "Group",
        "Gender",
    ]
    
    features = numerical_features + categorical_features

    # assert features == categorical_features + numerical_features
    
    df = df.dropna(subset=[target])

    X, y = df[features], df[target]

    # Label-encoding categorical variables
    for cat_feature in categorical_features:
        X[cat_feature].fillna("Unknown", inplace=True)
        encoder = LabelEncoder()
        X[cat_feature] = encoder.fit_transform(X[cat_feature])
        
    # Standard scalling numerical columns
    if scaling == True:
        scaler = StandardScaler()
        X[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Check no NaN values
    assert X.isna().sum().sum() == 0, "NaN values"

    return X, y, numerical_features, categorical_features, target
