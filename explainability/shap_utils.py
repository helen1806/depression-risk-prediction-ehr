import shap
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def generate_shap_explanations(model, X_background: pd.DataFrame, X_explain: pd.DataFrame, output_dir: str = "outputs", model_type: str = "tree"):
    os.makedirs(output_dir, exist_ok=True)

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_explain)

        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]

    else:

        f = lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x)
        explainer = shap.KernelExplainer(f, shap.kmeans(X_background, min(100, len(X_background))))
        shap_values_obj = explainer(X_explain)
        shap_values_to_plot = shap_values_obj.values
        expected_value = explainer.expected_value

    plt.figure()
    shap.summary_plot(shap_values_to_plot, X_explain, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()

    if model_type == "tree":

        exp = shap.Explanation(values=shap_values_to_plot[0],
                             base_values=expected_value,
                             data=X_explain.iloc[0],
                             feature_names=X_explain.columns)
    else:
        exp = shap_values_obj[0]

    shap.waterfall_plot(exp, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_waterfall.png"), dpi=300, bbox_inches='tight')
    plt.close()
