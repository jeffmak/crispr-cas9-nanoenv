import pandas as pd
import seaborn as sns
import numpy as np

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

list21 = ['Cas9_839_ep_average_WNADist', 'Cas9_734_density_CA_5_SW_5',
          'Cas9_733_dssp_kappa', 'Cas9_837_acc_isol_surfv', # kappa -> dssp_kappa
          'Cas9_837_stride_accessibility', 'Cas9_837_entd_CA_6_WNADist', # accessibility -> stride_accessibility
          'Cas9_734_dssp_kappa', 'Cas9_1016_sponge_CA_7_SW_5', # kappa -> dssp_kappa
          'Cas9_1015_dssp_kappa', 'Cas9_732_entd_LHA_5_IFR_WNASurf', # kappa -> dssp_kappa
          'Cas9_406_sponge_CA_7_IFR_SW_3', 'Cas9_271_dssp_kappa', # kappa -> dssp_kappa
          'Cas9_734_density_CA_5_IFR_SW_5', 'Cas9_731_entd_LHA_5_WNASurf',
          'Cas9_730_acc_ifr_surfv', 'Cas9_837_acc_complex_surfv',
          'Cas9_925_acc_ifr_surfv', 'Cas9_838_entd_CA_6_WNADist',
          'Cas9_136_cpo_85_30_CB_WNADist', 'Cas9_317_density_LHA_7_IFR_SW_5',
          'Cas9_731_wcn_avg_weighted_contact_number_k_5_WNADist'] # avg_weighted_contact_number_k_5_WNADist -> wcn_avg_weighted_contact_number_k_5_WNADist

def fill_protein_domains(ax, height):
  # color coding from https://pubs.acs.org/doi/10.1021/acs.biochem.1c00354
  ax.fill_between([0,60], 0, height, color='blue', alpha=0.2) # RuvC I
  ax.fill_between([60,94], 0, height, color='magenta', alpha=0.2) # Bridge Helix
  ax.fill_between([94,180], 0, height, color='lightgrey', alpha=0.2) # Rec I
  ax.fill_between([180,308], 0, height, color='grey', alpha=0.2) # Rec II
  ax.fill_between([308,496], 0, height, color='lightgrey', alpha=0.2) # Rec I
  ax.fill_between([496,718], 0, height, color='lightcoral', alpha=0.2) # Rec III
  ax.fill_between([718,775], 0, height, color='blue', alpha=0.2) # RuvC II
  ax.fill_between([775,909], 0, height, color='green', alpha=0.2) # HNH*
  ax.fill_between([909,1099], 0, height, color='blue', alpha=0.2) # RuvC
  ax.fill_between([1099,1368], 0, height, color='yellow', alpha=0.2) # PAM-interacting


x_name = 'CRISPR-Cas9 Amino Acid Residues'
list21_res = [int(s.split('_')[1]) for s in list21]

fig=plt.figure(figsize=(10,8/3*4))

gs=GridSpec(4,3) # 4 rows, 3 columns

ax1=fig.add_subplot(gs[0,:]) # First row, span all columns
ax3=fig.add_subplot(gs[1,0]) # Second row, first column
ax4=fig.add_subplot(gs[1,1]) # Second row, second column
ax5=fig.add_subplot(gs[1,2]) # Second row, third column
ax2=fig.add_subplot(gs[2,:]) # Third row, span all columns
ax6=fig.add_subplot(gs[3,0]) # Fourth row, first column
ax7=fig.add_subplot(gs[3,1]) # Fourth row, second column
ax8=fig.add_subplot(gs[3,2]) # Fourth row, third column

# locations: 0, 49, 99, ..., 1367
# names: 1, 50, 100, 1368
space = 100
locs = [0]
x = space-1
while x < 1367:
  locs.append(x)
  x += space

y_height = 9
df0 = pd.DataFrame({x_name: list21_res})
fig = sns.histplot(x=x_name, data=df0, bins=80, ax=ax1)
ax1.axhline(y=1.5, linestyle=':', color='black')
fill_protein_domains(ax1, y_height)
ax1.set_xlim(1, 1368)
ax1.set_ylim(0, y_height)
ax1.set_xlabel('CRISPR-Cas9 Amino Acid Residue')
ax1.set_ylabel("Number of Features")
ax1.set_xticks(locs)
ax1.set_xticklabels([str(x + 1) for x in locs])

df = pd.read_csv("data/first_fold_shap_values.csv")
df.rename(columns={'Cas9_733_kappa': 'Cas9_733_dssp_kappa',
                  'Cas9_837_accessibility': 'Cas9_837_stride_accessibility',
                  'Cas9_734_kappa': 'Cas9_734_dssp_kappa',
                  'Cas9_1015_kappa': 'Cas9_1015_dssp_kappa',
                  'Cas9_271_kappa': 'Cas9_271_dssp_kappa',
                  'Cas9_731_avg_weighted_contact_number_k_5_WNADist': 'Cas9_731_wcn_avg_weighted_contact_number_k_5_WNADist'}, inplace=True)

# g1 = [730, 731, 732, 733, 734]
# g2 = [837, 838, 839]
# g3 = [1015, 1016]

max_y = 4.1
sns.countplot(x=x_name, data=df0[(df0[x_name] >= 730) & (df0[x_name] < 735)],
              color=sns.color_palette()[0], ax=ax3)
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
ax3.set_ylim(0, max_y)
ax3.set_xlabel('CRISPR-Cas9 Amino Acid Residue')
ax3.set_ylabel("Number of Features")

sns.countplot(x=x_name, data=df0[(df0[x_name] >= 837) & (df0[x_name] < 840)],
              color=sns.color_palette()[0], ax=ax4)
ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
ax4.set_ylim(0, max_y)
ax4.set_xlabel('CRISPR-Cas9 Amino Acid Residue')
ax4.set_ylabel("Number of Features")

sns.countplot(x=x_name, data=df0[(df0[x_name] >= 1015) & (df0[x_name] < 1017)],
             color=sns.color_palette()[0], ax=ax5)
ax5.yaxis.set_major_locator(MaxNLocator(integer=True))
ax5.set_ylim(0, max_y)
ax5.set_xlabel('CRISPR-Cas9 Amino Acid Residue')
ax5.set_ylabel("Number of Features")

df_t = df.transpose().reset_index()
df_t['index'] = df_t['index'].apply(lambda s: int(s.split('_')[1]))
res_shap_vals = df_t.groupby('index').sum().reindex().transpose()
df2 = pd.DataFrame(res_shap_vals.abs().mean())
df2.columns = ['importance']
df3 = df2.reindex(pd.Index(np.arange(1, 1369, 1), name='residue'))\
         .fillna(0).reset_index()
df3['importance_cumsum'] = df3['importance'].cumsum()

sns.barplot(x='residue', y='importance', data=df3, ec='black', ax=ax2)
ax2a = ax2.twinx()
fig = sns.lineplot(x='residue', y='importance_cumsum', data=df3, ax=ax2a)
ax2.set_xticks(locs)
ax2.set_xticklabels([str(x + 1) for x in locs])
ylims = ax2.get_ylim()
fill_protein_domains(ax2, ylims[1])
ax2.set_xlim(0, 1368)
ax2.set_ylim(ylims)
ax2a.set_ylim(0)
ax2.set_xlabel('CRISPR-Cas9 Amino Acid Residue')
ax2.set_ylabel('SHAP CRISPR-Cas9\nResidue Importance')
ax2a.set_ylabel('Cumulative SHAP\nCRISPR-Cas9 Residue\nImportance')

#######
sns.barplot(x='residue', y='importance',
            data=df3[(df3['residue'] >= 730) & (df3['residue'] < 735)],
            color=sns.color_palette()[0], ax=ax6)
ax6.set_ylim(ylims)
ax6.set_xlabel('CRISPR-Cas9 Amino Acid Residue')
ax6.set_ylabel('SHAP CRISPR-Cas9\nResidue Importance')

sns.barplot(x='residue', y='importance',
            data=df3[(df3['residue'] >= 837) & (df3['residue'] < 840)],
            color=sns.color_palette()[0], ax=ax7)
ax7.set_ylim(ylims)
ax7.set_xlabel('CRISPR-Cas9 Amino Acid Residue')
ax7.set_ylabel('SHAP CRISPR-Cas9\nResidue Importance')

sns.barplot(x='residue', y='importance',
            data=df3[(df3['residue'] >= 1015) & (df3['residue'] < 1017)],
            color=sns.color_palette()[0], ax=ax8)
ax8.set_ylim(ylims)
ax8.set_xlabel('CRISPR-Cas9 Amino Acid Residue')
ax8.set_ylabel('SHAP CRISPR-Cas9\nResidue Importance')

plt.tight_layout()
plt.savefig("out/fig3.pdf")
plt.close()
