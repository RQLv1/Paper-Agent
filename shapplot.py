import shap, joblib, os
import pandas as pd
import matplotlib.pyplot as plt
shap.initjs()

class Shapplot:
    def __init__(self, model_res_dir=r"outputs/saved_models", 
                 input_dir=r"outputs"):
        self.model_res_dir = model_res_dir
        self.input_dir = input_dir

    def shap_plot(self):

        df = pd.read_csv(os.path.join(self.model_res_dir, "model_res.csv"))
        model_name = df[df['best_r2'] == max(df['best_r2'])]['model'].values[0]
        X = pd.read_csv(os.path.join(self.input_dir, "X.csv"), index_col= 0)
        model = joblib.load(f"{self.model_res_dir}/{model_name}_best_model.joblib")

        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 5))
        shap_values = explainer.shap_values(X)
        excluded_features = [col for col in X.columns if 'name' in col.lower() or 'unit' in col.lower()]
        display_features = [col for col in X.columns if col not in excluded_features]
        display_feature_indices = [X.columns.get_loc(col) for col in display_features]
        X_display = X[display_features]
        shap_values_display = shap_values[:, display_feature_indices]
        shap.summary_plot(shap_values_display, X_display, show=False, max_display=8, plot_type="bar")

        fig = plt.gcf()
        ax = plt.gca()
        current_xlabel = ax.get_xlabel()
        ax.set_xlabel(current_xlabel, color='black', fontdict={'weight': 'bold', 'size': 12})

        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_color('black')
        for label in ax.get_yticklabels(): 
            label.set_fontweight('bold')
            label.set_color('black')
        if len(fig.axes) > 1 and fig.axes[-1] is not ax:
            cbar_ax = fig.axes[-1]
            if hasattr(cbar_ax, 'yaxis') and cbar_ax.get_ylabel():
                 cbar_ax.yaxis.label.set_fontweight('bold')
                 cbar_ax.yaxis.label.set_color('black')
            for label in cbar_ax.get_xticklabels() + cbar_ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_color('black')

        save_path = os.path.join(self.input_dir, "shap.png")
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
