import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# File locations
parent_dir = './'
model_loc = f'{parent_dir}models/STING_CRISPR.joblib'
X_loc = f'{parent_dir}data/X.csv'
figS2_loc = f'{parent_dir}out/figS2.pdf'

# Load STING_CRISPR model
model = joblib.load(model_loc)
explainer = shap.TreeExplainer(model.steps[-1][1])

# Load STING_CRISPR input features
X_df = pd.read_csv(X_loc)
X_df.columns = ['Cas9_'+col[1:] for col in X_df.columns]
X = X_df.to_numpy()

# Compute and summarize SHAP values
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, feature_names=X_df.columns,
                    show=False, max_display=len(X_df.columns))
plt.tight_layout()
plt.savefig(figS2_loc)
plt.close()