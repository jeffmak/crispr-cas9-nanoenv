from scipy.stats import spearmanr, pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# True CRISPR-Cas9 cleavage activity values from Jones Jr et al.
labels_df = pd.read_csv("data/activity_labels.csv")
labels = dict(zip(labels_df['CMUT'], labels_df['Activity']))

df0 = pd.read_csv("data/perf/uCRISPR_scores.out", sep=" ",skiprows=2)
df0.columns = ['on_seq','off_seq','uCRISPR_score']
df0['activity'] = list(labels.values())

df1 = pd.read_csv("data/perf/SpCas9_kinetic_model_kclv_out.csv")

combined = df0.merge(df1, on='off_seq')
combined.rename({'uCRISPR_score': 'uCRISPR',
                 'SpCas9_kinetic_model_kclv': 'SpCas9KineticModel'},
                 axis=1, inplace=True)


combined['CFD'] = pd.read_csv("data/perf/CFD_scores.csv")['CFD']
combined['CNN_std'] = pd.read_csv("data/perf/CNN_std_scores.csv")['CNN_std']
combined['DeepCRISPR'] = pd.read_csv("data/perf/DeepCRISPR_scores.csv")['DeepCRISPR']

feats = ['CFD', 'DeepCRISPR', 'uCRISPR', 'CNN_std', 'SpCas9KineticModel']
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


# These values are obtained from running demo_fig2de.py with n=10^7
featss.append('Nanoenv-Cas9-WNA')
correl_type.append('Spearman')
correls.append(0.9593360322933769) # Spearman: 0.9593360322933769 +/- 0.029357767366982108
featss.append('Nanoenv-Cas9-WNA')
correl_type.append('Pearson')
correls.append(0.9802965590985614) # Pearson: 0.9802965590985614 +/- 0.018589596580879476

correl_df = pd.DataFrame({'feat': featss, 'correl_type': correl_type, 'value': correls})
correl_df.sort_values('value', inplace=True)
correl_df.to_csv("out/fig2c_values.csv", index=None)

plt.figure(figsize=(5,3))
ax = sns.barplot(x='value', y='feat', data=correl_df, hue='correl_type')
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_ylabel("CRISPR-Cas9 (Off-)Target\nCleavage Activity\nPrediction Models")
ax.set_xlabel("Correlation")
plt.legend(title=None)
plt.tight_layout()
plt.savefig("out/fig2c.pdf")
plt.close()
