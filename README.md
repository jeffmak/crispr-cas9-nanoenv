# Code accompanying "Accurate prediction of CRISPR-Cas9 off-target activity by learning to utilize protein 3D nanoenvironment descriptors"

This Github repository contains sample Python scripts and CSV datasets relating to the trained XGBoost model Nanoenv-Cas9-WNA produced in the paper "Accurate prediction of CRISPR-Cas9 off-target activity by learning to utilize protein 3D nanoenvironment descriptors".

![Method](figure1.png)

## Models
This repository contains the following models in the ```model``` folder:
|      File      |  Description   |
| ---------------|--------------- |
| nanoenv_cas9_wna.bin | XGBoost model, i.e., Nanoenv-Cas9-WNA |

## Data
This repository contains the following CSV files in the ```data``` folder:

| File | Description |
| --------------|------------ |
| activity_labels.csv | CRISPR-Cas9 off-target cleavage activities for the 28 (off-)target molecular dynamics trajectories |
| cas9_nucleotide_distances.csv | Distances between heteroduplex-proximal reisdues and heteroduplex nucleotides for all CMUT trajectories |
| example_input.csv | Input feature values required for Nanoenv-Cas9-WNA for the first PDB snapshot for trajectory CMUT1 |
| first_fold_shap_values.csv | SHAP values generated for the 672 PDB snapshots using Nanoenv-Cas9-WNA |
| input_features.csv | Input feature values required for Nanoenv-Cas9-WNA for all 672 PDB snapshots |

Containing Spearman and Pearson correlation performances, this repository also has the following files in the ```data/perf``` folder:

| File | Description |
| --------------|------------ |
| CFD_scores.csv | Predicted CFD scores for the 28 CMUT trajectories |
| CNN_std_scores.csv | Predicted CNN_std scores for the 28 CMUT trajectories |
| DeepCRISPR_scores.csv | Predicted CNN_std scores for the 28 CMUT trajectories |
| SpCas9_kinetic_model_kclv_out.csv | Predicted SpCas9KineticModel scores (https://www.nature.com/articles/s41467-022-28994-2) for the 28 CMUT trajectories |
| uCRISPR_scores.out | Predicted uCRISPR scores for the 28 CMUT trajectories |


## Demo Scripts
Sample Python scripts for users to play around with.
|  File  | Description  | Expected Output |
| -------|------------- | --------------- |
| demo_predict_one.py | Uses Nanoenv-Cas9-WNA to make a prediction for each of the 672 CRISPR-Cas9 (off-)target cleavage activity datapoint in input_features.csv | 
| demo_fig2de.py | Quantifies the Spearman and Pearson correlation performance of Nanoenv-Cas9-WNA (set n=10⁷ to reproduce Figures 2D-E) |

## Reproducibility Scripts
Python scripts for reporducing main manuscript figures.
|  File  | Description  |
| -------|------------- |
| fig2a.py | Python script for reproducing Figure 2A |
| fig2b.py | Python script for reproducing Figure 2B |
| fig2c.py | Python script for reproducing Figure 2C |
| fig3.py  | Python script for reproducing Figure 3  |
| fig4a.py | Python script for reproducing Figure 4A |

## Related Zenodo Repositories
|   DOI  | Description |  URL |
| -------|-------------|------|
| 10.5281/zenodo.7837070 | Structural stability analysis summary | https://zenodo.org/record/7837070 |
| 10.5281/zenodo.8028221 | CRISPR-Cas9 STING Descriptor Values | https://zenodo.org/record/8028221 |

# Installation
Install the required Python packages as listed below.

# Requirements
```matplotlib==3.5.3 numpy==1.24.3 shap==0.38.1 scipy==1.7.3 scikit-learn==1.0.2 pandas==1.5.2 xgboost==1.5.0```

# Contacts
jeffrey.kelvin.mak@cs.ox.ac.uk or peter.minary@cs.ox.ac.uk
