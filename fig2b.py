from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def true_labels(labels, cmuts, num_snaps=24):
  """ Obtain labels for XGBoost traning and testing

  Parameters:
    labels (dict(str,list(float))): dictionary mapping CMUT# to the list of
      activity labels across the specified CMUT trajectory, for each CMUT#.
    cmuts (list(int)): list of values which correspond to CMUTs
    num_snaps (int): number of snapshots per trajectory (default is 24)

  Returns:
    df_y (np.array): activity labels
  """
  y = np.array([labels['CMUT'+str(cmut)] for cmut in cmuts])
  y1 = np.expand_dims(y, axis=1)
  y1a = np.repeat(y1, num_snaps, axis=1)
  df_y = np.reshape(y1a, (len(cmuts)*num_snaps))
  return df_y

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

cmuts = [1, 2, 3, 4, 5,
         7, 8, 9, 10, 11, 12, 13, 14,
         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

mut_expts = {'CMUT4': (19,'A'), 'CMUT8': (19,'C'), 'CMUT2': (19,'G'),
             'CMUT7': (18,'A'), 'CMUT3': (18,'C'), 'CMUT5': (18,'T'),
             'CMUT19': (17,'A'), 'CMUT13': (17,'G'), 'CMUT10': (17,'T'),
             'CMUT12': (16,'A'), 'CMUT9': (16,'C'), 'CMUT21': (16,'T'),
             'CMUT27': (15,'A'), 'CMUT23': (15,'C'), 'CMUT29': (15,'G'),
             'CMUT28': (14,'C'), 'CMUT30': (14,'G'), 'CMUT17': (14,'T'),
             'CMUT26': (13,'A'), 'CMUT20': (13,'C'), 'CMUT25': (13,'G'),
             'CMUT24': (12,'A'), 'CMUT14': (12,'C'), 'CMUT18': (12,'G'),
             'CMUT11': (11,'A'), 'CMUT16': (11,'C'), 'CMUT22': (11,'G')}

model_name = 'Nanoenv-Cas9-WNA'
X_csv_loc = 'data/input_features.csv'
xgb_model_loc = 'models/nanoenv_cas9_wna.bin'

# raw materials
y_28 = labels.values()
y_672 = true_labels(labels, cmuts, num_snaps=24)
X_672 = pd.read_csv(X_csv_loc).to_numpy()

# canonical prediction method
model = XGBRegressor()
model.load_model(xgb_model_loc)
pred_672 = model.predict(X_672)

df = pd.DataFrame({'cmut': [x for x in labels for _ in range(24)], 'pred': pred_672, 'y': y_672})
df['pos'] = df['cmut'].apply(lambda x: 20 if x == 'CMUT1' else mut_expts[x][0])
df['error'] = (df['y'] - df['pred']) ** 2

temp = df[['pos','error']].groupby('pos').mean().reset_index()

ax = sns.barplot(x='pos', y='error', data=temp)
ax.invert_xaxis()
ax.set_xticklabels([11, 12, 13, 14, 15, 16, 17, 18, 19, 'On-Target'])
ax.set_xlabel("PAM-Distal Position")
ax.set_ylabel("Mean Squared Error")
ax.set_ylim(bottom=0, top=0.00022)
plt.tight_layout()
plt.savefig("out/fig2b.pdf")
plt.close()
