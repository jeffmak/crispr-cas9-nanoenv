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
| input_features.csv | Contains the input feature data required for Nanoenv-Cas9-WNA. |

## Sample Scripts

|  File  | Description  |
| -------|------------- |
| predict.py | Uses Nanoenv-Cas9-WNA to make a prediction for each of the 672 CRISPR-Cas9 (off-)target cleavage activity datapoint in input_features.csv. |

## Related Zenodo Repositories
|   DOI  | Description |  URL |
| -------|-------------|------|
| 10.5281/zenodo.7837070 | Structural stability analysis summary | https://zenodo.org/record/7837070 |
| 10.5281/zenodo.8028221 | CRISPR-Cas9 STING Descriptor Values | https://zenodo.org/record/8028221 |

# Requirements
```matplotlib==3.5.3 numpy==1.21.2 pandas==1.1.3 xgboost==1.6.2```

# Contacts
jeffrey.kelvin.mak@cs.ox.ac.uk or peter.minary@cs.ox.ac.uk
