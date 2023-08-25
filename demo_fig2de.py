from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

import seaborn as sns
import matplotlib.pyplot as plt

# This Python script uses the following data and model files:
# 1. An input CSV file containing 672 datapoints
#    from the crisprSQL dataset in the paper
input_feat_loc = 'data/input_features.csv'
# 2. XGBoost model containing Nanoenv-Cas9-WNA which predicts
#    CRISPR-Cas9 (off-)target cleavage activity.
model_bin_loc = 'models/nanoenv_cas9_wna.bin'
# 3. CSV file containing the 28 CRISPR-Cas9 (off-)target cleavage activity labels

# We will save the predictions in the "out" folder
out_loc = 'out/preds.csv'
###########################


# True CRISPR-Cas9 cleavage activity values from Jones Jr et al.
labels_df = pd.read_csv("data/activity_labels.csv")
labels = dict(zip(labels_df['CMUT'], labels_df['Activity']))

# Prepare true activity label values
y_true = np.repeat(list(labels.values()), 24)

# Read input features
X = pd.read_csv(input_feat_loc)

# Load Nanoenv-Cas9-WNA
bst = XGBRegressor()
bst.load_model(model_bin_loc)

# Predict CRISPR-Cas9 cleavage activity values
y_pred = bst.predict(X)

# Number of samples for the Spearman and Pearson correlation distributions
# Note: Change this to 10_000_000 to get the numbers in the paper,
#       though this will take ~12 hours to run.
n = 1000 # 10_000_000

# Calculate Spearman and Pearson correlation values
pred_reshped = np.reshape(y_pred, (28, 24))
cols = np.random.randint(0, 24, size=(n, 28))

row = list(range(28))
rows = np.array([row] * n)

res = pred_reshped[rows, cols]

y_28 = list(labels.values())
spearmans = np.array([spearmanr(y_28, res[i])[0] for i in range(n)])
pearsons = np.array([pearsonr(y_28, res[i])[0] for i in range(n)])

# Create plots for Figure 2D
sns.histplot(spearmans, kde=True, stat="density", linewidth=0, label='Spearman')
sns.histplot(pearsons, kde=True, stat="density", linewidth=0, color='orange',
             label='Pearson')
plt.xlabel('Correlation')
plt.xlim(right=1)
plt.legend()
plt.savefig("out/fig2d.pdf")
plt.close()

# Numbers corresponding to Figure 2E
fig2e_dict = {
'Correlation': ['Minimum', 'Maximum', 'Mean', 'Standard Deviation'],
'Spearman': [spearmans.min(), spearmans.max(), spearmans.mean(), spearmans.std()],
'Pearson': [pearsons.min(), pearsons.max(), pearsons.mean(), pearsons.std()]
}
pd.DataFrame(fig2e_dict).round(3).to_csv("out/fig2e.csv")
