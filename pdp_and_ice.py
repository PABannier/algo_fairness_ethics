from sklearn.inspection import PartialDependenceDisplay

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 16, 9

def pdp_ice(model, X_train, features, kind="both", subsample=50, n_jobs=3, grid_resolution=20, random_state=0):
    """
    Plots the partial dependence, both individual (ICE) and averaged one (PDP).
    Args:
        model : trained model to explain
        X_train : dataset used to train the model
        features (list): list of features
        kind {‘average’, ‘individual’, ‘both’}  : 
            kind='average' results in the traditional PD plot.
            kind='individual' results in the ICE plot.
            kind='both' results in the overlay of both PD and ICE plots.
        subsample (int) : number of ICE curves to display
    """
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_train,
        features,
        kind=kind,
        subsample=subsample,
        n_jobs=n_jobs,
        grid_resolution=grid_resolution,
        random_state=random_state,
        ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
        pd_line_kw={"color": "tab:orange", "linestyle": "--"},
    )
    display.figure_.subplots_adjust(hspace=0.5)