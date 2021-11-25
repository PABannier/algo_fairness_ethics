import shap

def shap_viz(model, X_train, instance, plot_type):
    """
    Visualizes different shap plots to explain the model's predictions using SHAP
    Args:
        model : trained model
        X_train : X_train used to train the model
        instance (int) : instance to be visualized
        plot_type {'waterfall', 'force', 'beeswarm', 'bar'} : 
        * waterfall : visualize the first prediction's explanation
        * force : visualizes the first prediction's explanation with a force plot
        * beeswarm : summarizes the effects of all the features
        * bar : produces stacked bars for multi-class outputs
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    
    if plot_type = 'waterfall':
        shap.plots.waterfall(shap_values[instance])

    elif plot_type = 'force':
        shap.plots.force(shap_values[instance])
        
    elif plot_type = 'beeswarm':
        shap.plots.beeswarm(shap_values)

    elif plot_type = 'bar':
        shap.plots.bar(shap_values)

