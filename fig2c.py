from scipy.stats import spearmanr, pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

df = pd.read_csv("data/perf/CRISPR_net_results.csv")
df['cmut'] = list(labels.keys())
df['activity'] = list(labels.values())

df2 = pd.read_csv("data/perf/finkel_crisproff_out.tsv",
                  sep='\t', usecols = ['off_target_seq','CRISPRoff_score'])
df2.rename({'off_target_seq': 'off_seq'}, axis=1, inplace=True)

df3 = pd.read_csv("data/perf/uCRISPR_finkel_data.out", sep=" ",skiprows=2)
df3.columns = ['on_seq','off_seq','uCRISPR_score']

df4 = pd.read_csv("data/perf/finkel_SpCas9_kinetic_model_kclv_out.csv")


# df
temp = df[['off_seq','CRISPR_Net_score','activity']].merge(df2[['off_seq','CRISPRoff_score']], on='off_seq')
temp2 = temp.merge(df3[['off_seq','uCRISPR_score']], on='off_seq')
combined = temp2.merge(df4, on='off_seq')
combined.rename({'CRISPR_Net_score': 'CRISPR-Net',
                 'CRISPRoff_score': 'CRISPRoff',
                 'uCRISPR_score': 'uCRISPR',
                 'SpCas9_kinetic_model_kclv': 'SpCas9KineticModel'}, axis=1, inplace=True)

# wna_df = pd.read_csv("data/perf/nanoenv_cas9_wna_8e-4_pred.csv")
# pointwise_df = pd.read_csv("data/perf/nanoenv_complex_pointwise_5e-4_pred.csv")
# combined['Nanoenv-Cas9-WNA'] = wna_df['Nanoenv-Cas9-WNA']
# combined['Nanoenv-Complex-Pointwise'] = pointwise_df['Nanoenv-Complex-Pointwise']

df0 = pd.read_csv("data/perf/finkel_cfd_scores.csv")
combined['CFD'] = df0['CFD']

df5 = pd.read_csv("data/perf/finkel_CNN_std.csv")
combined['CNN_std'] = df5['CNN_std']

df6 = pd.read_csv("data/perf/finkel_nanoenv_pair_predictions.csv")
combined['DeepCRISPR'] = df6['DeepCRISPR']
combined['CnnCrispr (Classification)'] = df6['CnnCrispr_classification']
# combined['piCRISPR'] = df6['piCRISPR_old']




feats = ['CFD', 'CNN_std', 'CRISPR-Net','CRISPRoff','uCRISPR',
         'SpCas9KineticModel','DeepCRISPR','CnnCrispr (Classification)',
        #  'piCRISPR'
         ]
correls = []
correl_type = []
featss = []
for feat in feats:
  featss.append(feat)
  correl_type.append('Spearman')
  correls.append(spearmanr(combined[feat], combined['activity'])[0])
  featss.append(feat)
  correl_type.append('Pearson')
  correls.append(pearsonr(combined[feat], combined['activity'])[0])

# featss.append('Nanoenv-Complex-Pointwise')
# correl_type.append('Spearman')
# correls.append(0.9553559917898199) # Spearman: 0.9553559917898199 +/- 0.030856224137285514
# featss.append('Nanoenv-Complex-Pointwise')
# correl_type.append('Pearson')
# correls.append(0.982320048891257) # Pearson: 0.982320048891257 +/- 0.016862054576648498

featss.append('Nanoenv-Cas9-WNA')
correl_type.append('Spearman')
correls.append(0.9593360322933769) # Spearman: 0.9593360322933769 +/- 0.029357767366982108
featss.append('Nanoenv-Cas9-WNA')
correl_type.append('Pearson')
correls.append(0.9802965590985614) # Pearson: 0.9802965590985614 +/- 0.018589596580879476

correl_df = pd.DataFrame({'feat': featss, 'correl_type': correl_type, 'value': correls})
correl_df.sort_values('value', inplace=True)
correl_df.to_csv("data/perf/correl_df.csv", index=None)


import matplotlib.ticker as ticker

# only keep models with non-negative correlation
correl_df_2 = correl_df[correl_df['value'] >= 0]

plt.figure(figsize=(5,3))
ax = sns.barplot(x='value', y='feat', data=correl_df_2, hue='correl_type')
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_ylabel("CRISPR-Cas9 (Off-)Target\nCleavage Activity\nPrediction Models")
ax.set_xlabel("Correlation")
plt.legend(title=None)
plt.tight_layout()
plt.savefig("out/fig2c.pdf")
plt.close()
