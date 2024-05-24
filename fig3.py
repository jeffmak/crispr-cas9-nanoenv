import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

##################
# File locations
parent_dir = './'
model_loc = f'{parent_dir}models/STING_CRISPR.joblib'
X_loc = f'{parent_dir}data/X.csv'
cmut_activity_loc = f'{parent_dir}data/activity_labels.csv'

fig3_save_loc = f'{parent_dir}out/fig3.pdf'

################
# Helper functions
def get_pos_name(nuc):
    """
    Prettify a residue/nucleotide of the format [A-D]{0-9}+.
    e.g., A100 -> Cas9_100
          ...
    In practice most of this is dead code since we
    only deal with the Cas9 protein, i.e., chain A of the PDB file.
    """
    chain = nuc[0]
    pos = int(nuc[1:])
    if chain == 'B' or chain == 'D':
        bd_pos = 21 - pos
        if bd_pos <= 0:
            bd_pos -= 1
            bd_pos_str = str(bd_pos)
        else:
            bd_pos_str = '+'+str(bd_pos)
        if chain == 'B':
            return 'sgRNA_'+str(bd_pos_str)
        else:
            return 'NTS_'+str(bd_pos_str)
    elif chain == 'C':
        c_pos = pos - 10
        if c_pos <= 0:
            c_pos -= 1
            c_pos_str = str(c_pos)
        else:
            c_pos_str = '+'+str(c_pos)
        return 'TS_'+str(c_pos_str)
    else:
        return 'Cas9_'+nuc[1:]

##################
# Load relevant data

# 1. Activity data
cmut_activity_df = pd.read_csv(cmut_activity_loc)
labels = {}
for _, row in cmut_activity_df.iterrows():
    labels[row['CMUT']] = row['Activity']

# 2. Interface names
interface_name = {'CMUT1': 'ON',
                'CMUT2': 'A19C',
                'CMUT3': 'C18G',
                'CMUT4': 'A19T',
                'CMUT5': 'C18A',
                'CMUT7': 'C18T',
                'CMUT8': 'A19G',
                'CMUT9': 'C16G',
                'CMUT10': 'G17A',
                'CMUT11': 'A11T',
                'CMUT12': 'C16T',
                'CMUT13': 'G17C',
                'CMUT14': 'A12G',
                'CMUT16': 'A11G',
                'CMUT17': 'T14A',
                'CMUT18': 'A12C',
                'CMUT19': 'G17T',
                'CMUT20': 'A13G',
                'CMUT21': 'C16A',
                'CMUT22': 'A11C',
                'CMUT23': 'A15G',
                'CMUT24': 'A12T',
                'CMUT25': 'A13C',
                'CMUT26': 'A13T',
                'CMUT27': 'A15T',
                'CMUT28': 'T14G',
                'CMUT29': 'A15C',
                'CMUT30': 'T14C'
                }

# 3. Mapping for Figure 3C
RNA_dict = {
    'A': 'U',
    'T': 'A',
    'C': 'G',
    'G': 'C'
}

# Number of input model features
n_final_feats = 30

##################
# Create figure
fig, axd = plt.subplot_mosaic([['A','A'],['B','C']], # ['Filter'],
                            constrained_layout=True, figsize=(8, 16/3))

# Load STING_CRISPR model
model = joblib.load(model_loc)

# Load STING_CRISPR input features
X_df = pd.read_csv(X_loc)

pretty_feature_names = []
for feat in X_df.columns:
    feat_lst = feat.split('_')
    feat_lst[0] = get_pos_name(feat_lst[0])
    pretty_feature_names.append('_'.join(feat_lst))

X_df.columns = pretty_feature_names

X = X_df.to_numpy()
X_672 = X[:672]
X_test = X_672.reshape((28, 24, n_final_feats))[:,20:,:].reshape((28*4, n_final_feats))
y_test_pred = model.steps[-1][-1].predict(X_test).reshape((28, 4))

######### Plot A ##########
y_test_dict = {'CMUT': [],
                'Guide-target Interface': [],
                'Activity': [],
                'Snapshot': [],
                'Predicted Activity': []}

for i, (cmut, activity) in enumerate(labels.items()):
    for j in range(4):
        y_test_dict['CMUT'].append(cmut)
        y_test_dict['Guide-target Interface'].append(interface_name[cmut])
        y_test_dict['Activity'].append(activity)
        y_test_dict['Snapshot'].append(20+j)
        y_test_dict['Predicted Activity'].append(y_test_pred[i][j])
y_test_df = pd.DataFrame(y_test_dict).sort_values('Activity')

y_test_df['Position'] = y_test_df['Guide-target Interface'].apply(lambda s: s if s == 'ON' else s[1:-1])
y_test_df['Test Squared Error'] = (y_test_df['Predicted Activity'] - y_test_df['Activity']) ** 2 

sns.boxplot(x='Guide-target Interface', y='Predicted Activity',
            data=y_test_df, ax=axd['A'])
ax = sns.scatterplot(x='CMUT', y='Activity', data=y_test_df, s=30, ax=axd['A'])
axd['A'].axhline(labels['CMUT1'], color='k', linestyle=':')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

for label, ax in axd.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=25, va='bottom', fontfamily='sans-serif', 
            weight='bold')


######### Plot B ##########
ax = sns.barplot(x='Position',y='Test Squared Error', 
            data=y_test_df, 
            order=['ON','19','18','17','16','15','14','13','12','11'],
            ax=axd['B'])


######### Plot C ##########
_X = X[:672].reshape((28, 24, n_final_feats))[:,20:,:].reshape((28*4, n_final_feats))
preds = model.steps[-1][1].predict(_X)

# construct df
results_dict = {'CMUT': [],
                'Mutation': [],
                'Activity': []}
for cmut, activity in labels.items():
    for _ in range(4):
        results_dict['CMUT'].append(cmut)
        results_dict['Activity'].append(activity)
        results_dict['Mutation'].append(interface_name[cmut])

results_df = pd.DataFrame(results_dict)
results_df['Predicted'] = preds
results_df['Test Squared Error'] = (results_df['Predicted'] - results_df['Activity']) ** 2
mut_df = results_df[results_df['Mutation'] != 'ON'].copy()
mut_df['Mismatch Type'] = mut_df['Mutation'].str[0].apply(lambda s: RNA_dict[s]) + ':d' + mut_df['Mutation'].str[-1]

ax2 = sns.barplot(x='Mismatch Type', y='Test Squared Error', data=mut_df, ax=axd['C'])
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

ax_ylim = ax.get_ylim()
ax2_ylim = ax2.get_ylim()
min_ylim = min(ax_ylim[0], ax2_ylim[0])
max_ylim = max(ax_ylim[1], ax2_ylim[1])
ax.set_ylim((min_ylim, max_ylim))
ax2.set_ylim((min_ylim, max_ylim))

plt.savefig(fig3_save_loc)
plt.close()