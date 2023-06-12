from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

# This Python script uses the following data and model files:
# 1. An input CSV file containing 672 datapoints
#    from the crisprSQL dataset in the paper
input_feat_loc = 'data/input_features.csv'
# 2. XGBoost model containing Nanoenv-Cas9-WNA which predicts
#    CRISPR-Cas9 (off-)target cleavage activity.
model_bin_loc = 'models/nanoenv_cas9_wna.bin'

# We will save the predictions in the "out" folder
out_loc = 'out/preds.csv'
###########################


# True CRISPR-Cas9 cleavage activity values from Jones Jr et al.
labels = {'CMUT1': 0.0928059068718,    # Matched (origNTS - pos - newNTS)
          'CMUT2': 0.0787201649863,    # A19C
          'CMUT3': 0.119446527474,     # C18G
          'CMUT4': 0.114618189415,     # A19T (was previously 0.119446527, which is wrong!!)
          'CMUT5': 0.132345410767,     # C18A
          # 'CMUT6': 0.0702299155412,  # *G20A
          'CMUT7': 0.14436947128,      # C18T
          'CMUT8': 0.181542289862,     # A19G
          'CMUT9': 0.00859648371669,   # C16G
          'CMUT10': 0.115995295517,    # G17A
          'CMUT11': 0.0561840566924,   # A11T
          'CMUT12': 0.111148283845,    # C16T
          'CMUT13': 0.106781628351,    # G17C
          'CMUT14': 0.0201204142155,   # A12G
          # 'CMUT15': 0.191606618035,  # *G20T
          'CMUT16': 0.00586891296755,  # A11G
          'CMUT17': 0.00208172468785,  # T14A
          'CMUT18': 0.00228748213652,  # A12C
          'CMUT19': 0.0932489445754,   # G17T
          'CMUT20': 0.00389589061704,  # A13G
          'CMUT21': 0.0196421394474,   # C16A
          'CMUT22': 0.00465650328521,  # A11C
          'CMUT23': 0.000971813092723, # A15G
          'CMUT24': 0.0815231184044,   # A12T
          'CMUT25': 0.00116725554782,  # A13C
          'CMUT26': 0.0111729840733,   # A13T
          'CMUT27': 0.00493182712775,  # A15T
          'CMUT28': 0.000458729016148, # T14G
          'CMUT29': 0.0945242769362,   # A15C
          'CMUT30': 0.0235738673331    # T14C
          }

# Prepare true activity label values
y_true = np.repeat(list(labels.values()), 24)

# Read input features
X = pd.read_csv(input_feat_loc)

# Load Nanoenv-Cas9-WNA
bst = XGBRegressor()
bst.load_model(model_bin_loc)

# predict CRISPR-Cas9 cleavage activity values
y_pred = bst.predict(X)

# Print results
print('Predicted CRISPR-Cas9 Cleavage Activities:', y_pred)
print('Spearman correlation:', spearmanr(y_pred, y_true)[0])
print('Pearson correlation:', pearsonr(y_pred, y_true)[0])

# Save the predictions
pd.DataFrame(y_pred).to_csv(out_loc)
