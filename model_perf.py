import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr, pearsonr

###########################
# Repository location
parent_dir = './'

# This Python script uses the following data and model files:
# 1. An input CSV file containing 672 datapoints
#    from the crisprSQL dataset in the paper
X_loc = f'{parent_dir}data/X.csv'

# 2. XGBoost model containing Nanoenv-Cas9-WNA which predicts
#    CRISPR-Cas9 (off-)target cleavage activity.
model_loc = f'{parent_dir}models/STING_CRISPR.joblib'

# 3. CSV file containing the 28 CRISPR-Cas9 (off-)target cleavage activity labels
cmut_activity_loc = f'{parent_dir}data/activity_labels.csv'
###########################

# Load input feature values
X = pd.read_csv(X_loc).to_numpy()

# Load STING_CRISPR model
pipeline = joblib.load(model_loc)
model = pipeline.steps[-1][1]

# The test set is the last 4 snapshots for each of the 28 trajectories
_X = X.reshape((28, 24, 30))
_X = _X[:, 20:, :]
X_test = _X.reshape((28*4, 30))

# Predict CRISPR-Cas9 cleavage activity values
y_pred = model.predict(X_test)

# Obtain ground truth labels
cmut_activities = pd.read_csv(cmut_activity_loc)['Activity'].to_numpy()
_y_test = np.repeat(cmut_activities[:, np.newaxis], 4, axis=1)
y_test = _y_test.reshape((28*4))

# Compute test metrics
spear = spearmanr(y_test, y_pred)[0]
pear = pearsonr(y_test, y_pred)[0]
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print test metrics
print(f'Spearman: {spear:.3f}')
print(f'Pearson: {pear:.3f}')
print(f'MSE: {mse:.2e}')
print(f'MAE: {mae:.2e}')

# Expected output:
# Spearman: 0.819
# Pearson: 0.916
# MSE: 5.92e-04
# MAE: 1.68e-02