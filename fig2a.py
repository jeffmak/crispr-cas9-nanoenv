import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##################
# Repository location
parent_dir = './'

# This Python script uses the following data:
# An input CSV file containing the grid search + five-fold cross validation results
results_loc = f'{parent_dir}data/hyperparam_search_5cv_results.csv'

# Location for saving the figure
fig2a_loc = f'{parent_dir}out/fig2a.pdf'

##################
# Helper functions

def get_best_feat_sel_model_class(results_df):
    """ Given a Pandas Dataframe containing the 5CV results,
        find (m_1, m_2) which maximizes the Spearman
        correlation when averaged across the 10 different
        possible feature set sizes (5, 10, ..., 50) for m2
    """
    results_df_summary = results_df.groupby(['n_final_feats','feat_sel','model_class','model_type']).mean().reset_index()
    results_df_summary = results_df_summary.sort_values('spear', ascending=False)
    temp = results_df_summary.iloc[0]
    return (temp['feat_sel'], temp['model_class'])

def get_spear_diff_df(results_df, best_model_type):
    """
    Given a Pandas Dataframe containing the 5CV results
    and the best (m_1, m_2), compute and output the
    Spearman correlation changes when varying the final
    input feature size from 5 to 50. 
    """
    n_final_feats_lst = results_df['n_final_feats'].unique()
    spear_diff_dfs = []
    for fold in range(5):
        fold_df = results_df[results_df['fold'] == fold]
        best_model_type_df = fold_df[fold_df['model_type'] == best_model_type].sort_values('n_final_feats',ascending=True)
        spear_temp = best_model_type_df['spear'].to_numpy()
        spear_diff = spear_temp[1:] - spear_temp[:-1]

        spear_diff_df = pd.DataFrame({'fold': [fold] * 9,
                                    'n_final_feats': ((n_final_feats_lst[1:] + n_final_feats_lst[:-1]) / 2.0).tolist(),
                                    'spear_diff': spear_diff})
        spear_diff_dfs.append(spear_diff_df)
    spear_diff_df = pd.concat(spear_diff_dfs)
    return spear_diff_df

##################
# Parameters
threshold = 0.002

##################
# 1. find best (model_class, feat_sel, n_final_feats)

# Load results from grid search + five-fold cross validation
results_df = pd.read_csv(results_loc)

feat_sel, model_class = get_best_feat_sel_model_class(results_df)
best_model_type = f'{feat_sel}_{model_class}'

spear_diff_df = get_spear_diff_df(results_df, best_model_type)

temp = spear_diff_df.groupby('n_final_feats').mean()
for idx, row in temp.iterrows():
    if row['spear_diff'] < threshold:
        n_final_feats = int(idx - 2.5)
        break

# 2. create figure
fig, ax = plt.subplots(1,1, figsize=(4, 3))

# Auto-select number of final features
best_model_type_df = results_df[results_df['model_type'] == best_model_type]
sns.lineplot(x='n_final_feats', y='spear',
            data=best_model_type_df,
            ax=ax)
ax.axvline([n_final_feats], linestyle=':') # et
ax.set_xlabel('Number of Final Features')
ax.set_ylabel('Five-fold cross-validation\nSpearman Correlation')
ax.set_xticks(best_model_type_df['n_final_feats'].unique())

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

sns.lineplot(x='n_final_feats',y='spear_diff', data=spear_diff_df, color='orange', marker='d', ax=ax2)
ax2.axhline([threshold], color='k', linestyle=':')
ax2.set_xlabel('Feature Size Increases')
ax2.set_ylabel('5-fold cross-validation\nSpearman Correlation\nChange')

plt.tight_layout()
plt.savefig(fig2a_loc)
plt.close()