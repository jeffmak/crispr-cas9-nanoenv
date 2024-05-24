import numpy as np
import pandas as pd
import joblib
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

##################
# Helper functions
def which_residue_broad(res):
  if (1 <= res and res < 56) or (717 <= res and res < 765) or (926 <= res and res < 1100):
    return 'RuvC'
  elif 56 <= res and res < 94:
    return 'Bridge Helix'
  elif 94 <= res and res < 717:
    return 'REC'
  elif 765 <= res and res < 926:
    return 'HNH'
  elif 1100 <= res and res < 1368:
    return 'PAM-interacting'

##################
# File locations
parent_dir = './'
model_loc = f'{parent_dir}models/STING_CRISPR.joblib'
X_loc = f'{parent_dir}data/X.csv'
desc2desctype_loc = f'{parent_dir}data/desc2desctype.csv'
fig5_loc = f'{parent_dir}out/fig5.pdf'

##################

# Load STING_CRISPR model
model = joblib.load(model_loc)

# Load STING_CRISPR input features
X_df = pd.read_csv(X_loc)

# Read the table of STING descritpor - STING descriptor...
desc2desctype = pd.read_csv(desc2desctype_loc, index_col='columns')
desc2desctype = desc2desctype.fillna(np.nan).replace([np.nan], [None])

# ...and build a dictionary out of it
desc_class_dict = {}
for feat in X_df.columns:
    desc = '_'.join(feat.split('_')[1:])
    _desc = desc2desctype.loc[desc]
    if _desc['parent_type'] == 'Density/Sponge':
        if 'density' in feat:
            desc_class_dict[feat] = ('Density', _desc['agg_type'])
        elif 'sponge' in feat:
            desc_class_dict[feat] = ('Sponge', _desc['agg_type'])
        else:
            raise NotImplementedError
    else:
        desc_class_dict[feat] = (_desc['parent_type'], _desc['agg_type'])

# Compute SHAP values
explainer = shap.TreeExplainer(model.steps[-1][1])
X = X_df.to_numpy()
shap_values = explainer.shap_values(X)

desc_cols_list = [
    ([desc_class_dict[feat][0] for feat in X_df.columns], 'STING Descriptor Class'),
    ([which_residue_broad(int(feat.split('_')[0][1:])) for feat in X_df.columns], 'Cas9 Domain')
]

# Create plot
fig, axd = plt.subplot_mosaic([['STING Descriptor Class Count',
                                'Cas9 Domain Count'
                                ],
                            ['STING Descriptor Class SHAP',
                                'Cas9 Domain SHAP'
                            ]],
                            constrained_layout=True, figsize=(10, 10))

for desc_cols, desc_cols_name in desc_cols_list:
    shap_df = pd.DataFrame(shap_values, columns=desc_cols)

    df = pd.DataFrame(shap_df.transpose().index.value_counts())\
                        .reset_index()\
                        .rename({'index': desc_cols_name,
                            'count':'Feature Count'}, axis=1)
    ax = axd[f'{desc_cols_name} Count']
    sns.barplot(x=desc_cols_name, y='Feature Count', data=df, ax=ax)
    
    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', va='top', rotation_mode='anchor')

    desc_shap_values = shap_df.transpose().reset_index().groupby('index').sum().transpose()
    desc_shap_df = pd.DataFrame(desc_shap_values.abs().mean(axis=0)).sort_values(0, ascending=False).reset_index().rename({0: f'SHAP Importance','index':desc_cols_name}, axis=1)
    ax = axd[f'{desc_cols_name} SHAP']
    sns.barplot(x=desc_cols_name, y=f'SHAP Importance', data=desc_shap_df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', va='top', rotation_mode='anchor')

label_dict = {
    'STING Descriptor Class Count': 'A',
    'Cas9 Domain Count': 'B',
    'STING Descriptor Class SHAP': 'C',
    'Cas9 Domain SHAP': 'D'
}

for label, ax in axd.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-25/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label_dict[label], transform=ax.transAxes + trans,
            fontsize=25, va='bottom', fontfamily='sans-serif', 
            weight='bold')

plt.savefig(fig5_loc)
plt.close()