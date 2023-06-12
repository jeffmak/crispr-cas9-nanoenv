from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr


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

bst = XGBRegressor()  # init model
bst.load_model('nanoenv_cas9_wna.bin')  # load data

X = pd.read_csv('input_features.csv')
y_pred = bst.predict(X)

y_true = np.repeat(list(labels.values()), 24)

print('predictions:', y_pred)
print('Spearman:', spearmanr(y_pred, y_true)[0])
print('Ppearman:', pearsonr(y_pred, y_true)[0])
