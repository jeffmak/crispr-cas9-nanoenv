import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.ticker import MaxNLocator
import matplotlib.transforms as mtransforms
import shap

##################
# Helper functions
def which_group_res(res):
    """ Given a STING_CRISPR residue as input,
        return its associated residue group.
    """
    if res in [1016, 1017]:
        return 'Group 1'
    elif res in [728, 730, 732, 733, 734]:
        return 'Group 2'
    elif res in [837, 838, 839]:
        return 'Group 3'
    elif res in [136, 164, 317, 402, 408, 411, 415]:
        return 'Group 4'
    else:
        return 'Other'

def which_group(feat):
    """ Given a STING_CRISPR input feature as input,
        return its associated residue group.
    """
    res = int(feat.split('_')[0][1:])
    return which_group_res(res)

def group2index(res):
    """ Given a STING_CRISPR residue as input,
        return its associated residue group's index.
    """
    if res in [1016, 1017]:
        return 1
    elif res in [728, 730, 732, 733, 734]:
        return 2
    elif res in [837, 838, 839]:
        return 3
    elif res in [136, 164, 317, 402, 408, 411, 415]:
        return 4
    elif res in [268, 908, 919, 1010, 1025, 1122]:
        return 5
    else:
        return 0

##################
# Other useful values

# List of residues in STING_CRISPR
res_list = [136, 164, 268, 317, 402,
            408, 411, 415, 728, 730,
            732, 733, 734, 837, 838,
            839, 908, 919, 1010, 1016,
            1017, 1025, 1122]

# Mappings relating to the five residue groups 
group2color = {
    'Group 1': "#ff0000",
    'Group 2': "#ff8000",
    'Group 3': "#ffa6d9",
    'Group 4': "#ffff00",
    'Other':   "#bfbfff"
}
x_order = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Other']
x_order_dict = {
    'Group 1': [1016, 1017],
    'Group 2': [728, 730, 732, 733, 734],
    'Group 3': [837, 838, 839],
    'Group 4': [136, 164, 317, 402, 408, 411, 415],
    'Other': [268, 908, 919, 1010, 1025, 1122]
}

##################
# File locations
parent_dir = './'
model_loc = f'{parent_dir}models/STING_CRISPR.joblib'
X_loc = f'{parent_dir}data/X.csv'
residue_dists_loc = f'{parent_dir}data/residue_dists.csv'
fig4f_loc = f'{parent_dir}data/fig4f.png'
fig4g_loc = f'{parent_dir}data/fig4g.png'
fig4_loc = f'{parent_dir}out/fig4.pdf'

# Load STING_CRISPR model
model = joblib.load(model_loc)
explainer = shap.TreeExplainer(model.steps[-1][1])

# Load STING_CRISPR input features
X_df = pd.read_csv(X_loc)
X = X_df.to_numpy()

# Compute SHAP values
shap_values = explainer.shap_values(X)

# Load pairwise residue distances for STING_CRISPR residues
dist_df = pd.read_csv(residue_dists_loc, index_col=[0])
dist_df.columns = [int(col) for col in dist_df.columns]
cg = sns.clustermap(dist_df, cmap='mako_r', vmax=12, figsize=(18, 9))
temp = dist_df.iloc[cg.dendrogram_row.reordered_ind,
                         cg.dendrogram_col.reordered_ind] < 12
plt.close() # we don't need the clustermap

# Setup relevant residue count and SHAP importance dataframes
res_group_cols = X_df.columns.to_series().apply(which_group)
shap_values_df = pd.DataFrame(shap_values, columns=res_group_cols)
group_df = shap_values_df.transpose().reset_index().groupby('index').sum().abs().mean(axis=1)

group_df = group_df.to_frame()
group_df = group_df.loc[x_order]
group_df = group_df.reset_index().rename({'index': 'Residue Groups', 0: 'SHAP Importance'}, axis=1)

count_df = res_group_cols.value_counts().to_frame()
count_df = count_df.loc[x_order]
count_df = count_df.reset_index().rename({'index': 'Residue Groups', 'count': 'Feature Count'}, axis=1)

##################
# Create figure
fig, axd = plt.subplot_mosaic([['A'] * 2 + ['B']*2 + ['C']*2,
                               ['D'] * 3 + ['E'] * 3,
                               ['F']*3 + ['G']*3], 
                              figsize=(14, 14), height_ratios=[2, 0.9, 2.5],
                              )

######### Plot A ##########
# Threshold the residue indices,
# and assign indices for residue group coloring
dist_df_2 = (dist_df.iloc[cg.dendrogram_row.reordered_ind,
                         cg.dendrogram_col.reordered_ind]).copy()
for res in dist_df_2.columns:
    for res2 in dist_df_2.columns:
        if dist_df_2.loc[res][res2] < 12 and group2index(res) == group2index(res2):
            dist_df_2.loc[res][res2] = group2index(res)
        else:
            dist_df_2.loc[res][res2] = 0

# Create plot
palette = ["#000000","#ff0000","#ff8000",
           "#ffa6d9","#ffff00","#bfbfff"]
sns.heatmap(dist_df_2, cmap=palette, cbar=False, ax=axd['A'])

######### Plot B ##########
sns.barplot(x='Residue Groups', y='Feature Count', data=count_df,
            palette=palette[1:], ax=axd['B'])
axd['B'].set_xticklabels(axd['B'].get_xticklabels(), rotation=45)

######### Plot C ##########
sns.barplot(x='Residue Groups', y='SHAP Importance', data=group_df,
            palette=palette[1:], ax=axd['C'])
axd['C'].set_xticklabels(axd['C'].get_xticklabels(), rotation=45)


######### Plot D ##########
# Create plot D and E's color palette
plot_x_order = [x for x_lst in x_order_dict.values() for x in x_lst]
plot_palette = [group2color[which_group_res(x)] for x in plot_x_order]

# Compute and plot feature counts
list21_res = sorted([int(s.split('_')[0][1:]) for s in X_df.columns])
df0 = pd.DataFrame({'CRISPR-Cas9 Residue': list21_res})
sns.countplot(x='CRISPR-Cas9 Residue', data=df0,
                palette=plot_palette, ax=axd['D'],
                order=plot_x_order)

# Rotate x-axis labels
axd['D'].set_xticklabels(axd['D'].get_xticklabels(), rotation=45)

# Configure y-axis
max_y = 3.5
axd['D'].yaxis.set_major_locator(MaxNLocator(integer=True))
axd['D'].set_ylim(0, max_y)
axd['D'].set_ylabel("Feature Count")

######### Plot E ##########
# Compute SHAP importances for the 5 residue groups
df = pd.DataFrame(shap_values, columns=X_df.columns)
df_t = df.transpose().reset_index()
df_t['index'] = df_t['index'].apply(lambda s: int(s.split('_')[0][1:]))
res_shap_vals = df_t.groupby('index').sum().reindex().transpose()
df2 = pd.DataFrame(res_shap_vals.abs().mean())
df2.columns = ['SHAP Importance']
df3 = df2.reindex(pd.Index(np.arange(1, 1369, 1), name='CRISPR-Cas9 Residue'))\
        .fillna(0).reset_index()

# Create plot
sns.barplot(x='CRISPR-Cas9 Residue', y='SHAP Importance',
            data=df3[(df3['SHAP Importance'] > 0)],
            palette=plot_palette, ax=axd['E'],
            order=plot_x_order)

# Rotate x-axis labels
axd['E'].set_xticklabels(axd['E'].get_xticklabels(), rotation=45)

######### Plot F ##########
img2 = mpimg.imread(fig4f_loc)
axd['F'].imshow(img2)
axd['F'].set_xticks([])
axd['F'].set_yticks([])

######### Plot G ##########
img3 = mpimg.imread(fig4g_loc)
axd['G'].imshow(img3)
axd['G'].set_xticks([])
axd['G'].set_yticks([])

##################
# Add subfigure labels
for label, ax in axd.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-25/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=25, va='bottom', fontfamily='sans-serif', 
            weight='bold')


plt.tight_layout()
plt.savefig(fig4_loc)
plt.close()
