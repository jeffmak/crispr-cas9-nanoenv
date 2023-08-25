from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

# This Python script uses the following data and model files:
# 1. An input CSV file containing the STING descriptors for a single
#    PDB file (e.g. the first PDB snapshot for trajectory CMUT1).
#    Feel free to change the input feature values to those of any other PDB
#    snapshot.
input_feat_loc = 'data/example_input.csv'

# 2. XGBoost model containing Nanoenv-Cas9-WNA which predicts
#    CRISPR-Cas9 (off-)target cleavage activity.
model_bin_loc = 'models/nanoenv_cas9_wna.bin'

# We will save the predictions in the "out" folder
out_loc = 'out/preds_one.csv'
###########################

# Read input features
X = pd.read_csv(input_feat_loc)

# Load Nanoenv-Cas9-WNA
bst = XGBRegressor()
bst.load_model(model_bin_loc)

# Predict CRISPR-Cas9 cleavage activity value
y_pred = bst.predict(X)[0]

# Print results
print('Predicted CRISPR-Cas9 Cleavage Activity:', y_pred)
