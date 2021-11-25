from PyALE import ale

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 16, 9

def ale_viz(model, X_train, features, include_CI):
    """
     Plots the accumulation local effect of a given input feature's effect on the prediction of a ML model on average, taking into consideration the correlation between the features
     Args:
         model : trained model
         X_train : X_train used to train the model
         features (list) : list of the features whose influence on the prediction of the ML model is evaluated
         include_CI (boolean) : confidence interval of 0.95
    """
    for i in features:
        ale(
            X=X_train,
            model=model,
            feature=[i],
            grid_size=5,
            include_CI=include_CI,
            C=0.95,
        )