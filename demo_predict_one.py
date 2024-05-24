import numpy as np
import pandas as pd
import joblib

###########################
# Repository location
parent_dir = './'

# This Python script uses the following data and model files:
# 1. An input CSV file containing the STING features for a single
#    PDB file (e.g., the first PDB snapshot for trajectory CMUT1).
#    Feel free to change the input feature values to those of any other PDB
#    snapshot.
example_input_loc = f'{parent_dir}data/example_input.csv'

# 2. XGBoost model containing Nanoenv-Cas9-WNA which predicts
#    CRISPR-Cas9 (off-)target cleavage activity.
model_loc = f'{parent_dir}models/STING_CRISPR.joblib'
###########################

# Load feature values from example input
X = pd.read_csv(example_input_loc).to_numpy()

# Load STING_CRISPR model
pipeline = joblib.load(model_loc)
model = pipeline.steps[-1][1]

# Predict CRISPR-Cas9 cleavage activity values
y_pred = model.predict(X)[0]

# Print prediction
print(f'Predicted CRISPR-Cas9 Cleavage Activity: {y_pred:.5f}')

# Expected output:
# Predicted CRISPR-Cas9 Cleavage Activity: 0.09281