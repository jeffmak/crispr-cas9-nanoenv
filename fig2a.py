from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

import seaborn as sns
import matplotlib.pyplot as plt



cmuts = [1, 2, 3, 4, 5,
         7, 8, 9, 10, 11, 12, 13, 14,
         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
high_cmut_nums = {10,12,13,3,4,5,7,8}
mid_cmut_nums = {29,19,24,2,11,30,14,21,26}
low_cmut_nums = {9,16,27,22,20,18,17,25,23,28}


def predict_trajectory_activities(model, X, cmuts, num_snaps=24, save_loc=None, num_trajs=28):
  """ Plot of predicted activity values for all PDB snapshot, where:
      - the x-axis represents each snapshot (1-24)
      - the y-axis represents predicted CRISPR activity
      - each line represents a CMUT trajectory (i.e. 24 snapshots)
      - the black line represents the on-target trajectory (i.e. CMUT1)
      - green lines represent CMUT trajectories with high activity (i.e. > 0.1)
      - blue lines represent CMUT trajectories with medium activity (i.e. 0.01 - 0.1)
      - red lines represent CMUT trajectories with low activity (i.e. < 0.01)

  Parameters:
    model (xgb.XGBRegressor): XGBoost model for producing (the paper uses models with 200 trees)
    X (np.ndarray): input dataset
    cmuts (list(int)): list of CMUT values, e.g. 1 represents CMUT1
    num_nts_or_aas (int): number of nucleotides or amino acid residues
    feats_per_res (int): Number of nanoenvironment features per residue (default is 5)
    num_snaps (int): number of snapshots per trajectory (default is 24)
    save_loc (str): file location to save the plot (default is None)
    num_trajs (int): number of CMUT trajectories (default 28)

  Effects:
    Creates the plot as described above.
    If save_loc is not None, save the plot to save_loc.

  Returns:
    None
  """
  # num_nts_or_aas*feats_per_res
  X_2 = np.reshape(X, (num_trajs, num_snaps, X.shape[1]))
  x = np.arange(1,num_snaps+1)
  for i, cmut in enumerate(cmuts):
    if cmut == 1: # CMUT1
      color = 'black'
      label = 'On-Target'
    elif cmut in high_cmut_nums:
      color = 'green'
      label = 'High (> 0.1)'
    elif cmut in mid_cmut_nums:
      color = 'blue'
      label = 'Medium (0.01 - 0.1)'
    elif cmut in low_cmut_nums:
      color = 'red'
      label = 'Low (< 0.01)'
    else:
      raise NotImplementedError
    plt.plot(x, model.predict(X_2[i]),
            color=color,
            linewidth=2 if label == 'On-Target' else 0.5, label=label)
  handles, xlabels = plt.gca().get_legend_handles_labels()
  by_label = dict(zip(xlabels, handles))
  plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
  plt.xlabel('Snapshot')
  plt.ylabel("Predicted Activity")
  if save_loc is not None:
    plt.savefig(save_loc, bbox_inches='tight')
  plt.close()


model_bin_loc = 'models/nanoenv_cas9_wna.bin'


bst = XGBRegressor()
bst.load_model(model_bin_loc)
model = bst

input_feat_loc = 'data/input_features.csv'
X = pd.read_csv(input_feat_loc).to_numpy()

num_snaps = 24
save_loc = "out/fig2a.pdf"
num_trajs = 28

predict_trajectory_activities(model, X, cmuts, num_trajs=num_trajs,
                                  save_loc=save_loc)
