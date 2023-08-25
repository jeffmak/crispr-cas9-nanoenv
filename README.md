# Code accompanying "Accurate prediction of CRISPR-Cas9 off-target activity by learning to utilize protein 3D nanoenvironment descriptors"

This Github repository contains a sample Python script and CSV dataset for running the trained XGBoost model Nanoenv-Cas9-WNA produced in the paper "Accurate prediction of CRISPR-Cas9 off-target activity by learning to utilize protein 3D nanoenvironment descriptors".

![Method](figure1.png)

## Models
This repository contains the following models in the ```model``` folder:
|      File      |  Description   |
| ---------------|--------------- |
| nanoenv_cas9_wna.bin | XGBoost model, i.e., Nanoenv-Cas9-WNA |

## Data
This repository contains the following CSV datasets in the ```data``` folder:

| File | Description |
| --------------|------------ |
| activity_labels.csv | CRISPR-Cas9 off-target cleavage activities for the 28 (off-)target molecular dynamics trajectories |
| 
| input_features.csv | Contains the input feature data required for Nanoenv-Cas9-WNA. |

## Demo Scripts
Sample Python scripts to play around with.
|  File  | Description  |
| -------|------------- |
| demo_predict_one.py | Uses Nanoenv-Cas9-WNA to make a prediction for each of the 672 CRISPR-Cas9 (off-)target cleavage activity datapoint in input_features.csv. |
| demo_fig2de.py | Quantifies the Spearman and Pearson correlation performance of Nanoenv-Cas9-WNA (set n=10⁷ to reproduce Figures 2D-E)

## Reproducibility Scripts
Python scripts for reporducing main manuscript figures.
|  File  | Description  |
| -------|------------- |
| fig2a.py | Python script for reproducing Figure 2A |
| fig2b.py | Python script for reproducing Figure 2B |
| fig2c.py | Python script for reproducing Figure 2C |
| fig3.py | Python script for reproducing Figure 3 |
| fig4a.py | Python script for reproducing Figure 4A |

## Related Zenodo Repositories
|   DOI  | Description |  URL |
| -------|-------------|------|
| 10.5281/zenodo.7837070 | Structural stability analysis summary | https://zenodo.org/record/7837070 |
| 10.5281/zenodo.8028221 | CRISPR-Cas9 STING Descriptor Values | https://zenodo.org/record/8028221 |

# Requirements
```matplotlib==3.5.3 numpy==1.24.3 shap==0.38.1 scipy==1.7.3 scikit-learn==1.0.2 pandas==1.5.2 xgboost==1.5.0```

# Contacts
jeffrey.kelvin.mak@cs.ox.ac.uk or peter.minary@cs.ox.ac.uk
