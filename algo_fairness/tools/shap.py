import shap
import matplotlib.pyplot as plt


def shap_viz(model, X_train, feature_names, out_path=None):
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
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(
        shap_values, X_train, feature_names=feature_names, plot_type="bar"
    )

    if out_path:
        plt.draw()
        plt.savefig(out_path)
        print(f"SHAP saved at {out_path}!")
    plt.show()
