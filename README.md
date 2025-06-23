# Single-Cell Transcriptomic Aging Clock

## Overview

This repository contains the code used for building and interpreting single-cell RNA-seq aging clocks using ElasticNet and LightGBM.
The goal is to predict donor biological age from immune cell transcriptomes and identify key aging-related genes.

---

## Repository Information

The latest version of this project is available in the **`main`** branch, which contains the final, optimized version of the preprocessing and modeling pipeline.  

If you're interested in earlier development stages (such as the implementation of the **replication models** from the original study, prior to enhancements), you can explore the other branches in this repository. These branches are preserved for transparency and reproducibility purposes.

### Repository URL

You can access the full project on GitHub here:  
**[https://github.com/Vlad13503/Aging-Clock](https://github.com/Vlad13503/Aging-Clock)**

---

## Project Structure

The only files available on this repository are the *src* folder, which contains all scripts that were used for preprocessing, training, applying, evaluating, and visualizing the results of this project, together with the *requirements* text file that contains the necessary Python dependencies to run the pipeline.
All the results can be generated using the scripts in the *src* file. However, to start the process, the data files are required.

---

## Data Access

The large `.h5ad` datasets used in this project are not included in the repository due to the large size (~10GB each).

You can manually download them from:

- [AIDA Phase 1 Data Freeze v1](https://cellxgene.cziscience.com/collections/ced320a1-29f3-47c1-a735-513c7084d508)
- [Yoshida PBMC data](https://cellxgene.cziscience.com/collections/03f821b4-87be-4ff4-b65a-b5fc00061da7)
- [Liu adaptive cells](https://cellxgene.cziscience.com/collections/ed9185e3-5b82-40c7-9824-b2141590c7f0)
- [eQTL data](https://cellxgene.cziscience.com/collections/dde06e0f-ab3b-46be-96a2-a8082383c4a1)
- [Stephenson data](https://cellxgene.cziscience.com/collections/ddfad306-714d-4cc0-9985-d9072820c530)

Then:

1. Create a folder called `data/` in the root of the repo (if it doesn't exist).
2. Rename all data files to `AIDA.h5ad`, `Yoshida.h5ad`, `Liu.h5ad`, `eQTL.h5ad`, and `Stephenson.h5ad`.
3. Place them in the `data/` folder.


---

## Installation

This project requires Python ≥ 3.9. To start, create a virtual environment and install the dependencies using:

```bash
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 1: Train ElasticNet Aging Clocks

Note that this is a memory intensive task, so you might require access to a high performance computer, for example Delft Blue Supercomputer.

Go to the terminal and make sure you are in the **src** folder, so from the root of you project execute:
```bash
cd src
```

Then train the linear models by running the *train_model.py* file (this will take a while):
```bash
python train_model.py
```

### Step 2: Evaluate ElasticNet on training dataset

Run the *evaluate.py* file to evaluate the obtained predictions:
```bash
python evaluate.py
```

### Step 3: Visualize the results for linear models

Run the *visualize.py* file to plot the results of the evaluation:
```bash
python visualize.py
```

### Step 4: Perform SHAP feature importance

Run the *SHAP_feature_importance.py* file to plot the importance scores:
```bash
python SHAP_feature_importance.py
```

### Step 5: Apply the linear models to the four external datasets

Run the *apply_external_models.py* file to evaluate the obtained external predictions:
```bash
python apply_external_models.py
```

### Step 6: Evaluate ElasticNet on external datasets

Run the *evaluate_external.py* file to evaluate the obtained predictions:
```bash
python evaluate_external.py
```

### Step 7: Visualize the results for the external datasets

Run the *visualize_external.py* file to plot the results of the external evaluation:
```bash
python visualize_external.py
```

---

### Step 8: Train LightGBM Aging Clocks

Now we will follow the same steps, but for the non-linear models. Note that this is an even more memory intensive task.

Train the non-linear models by running the *train_nonlinear_model.py* file (this will take a while):
```bash
python train_nonlinear_model.py
```

### Step 9: Evaluate LightGBM on training dataset

Run the *evaluate_nonlinear.py* file to evaluate the obtained predictions:
```bash
python evaluate_nonlinear.py
```

### Step 10: Visualize the results for non-linear models

Run the *visualize_nonlinear.py* file to plot the results of the evaluation:
```bash
python visualize_nonlinear.py
```

### Step 11: Perform SHAP feature importance

Run the *SHAP_feature_importance_nonlinear.py* file to plot the importance scores:
```bash
python SHAP_feature_importance_nonlinear.py
```

### Step 12: Apply the non-linear models to the four external datasets

Run the *apply_external_nonlinear.py* file to evaluate the obtained external predictions:
```bash
python apply_external_nonlinear.py
```

### Step 13: Evaluate LightGBM on external datasets

Run the *evaluate_external_nonlinear.py* file to evaluate the obtained predictions:
```bash
python evaluate_external_nonlinear.py
```

### Step 14: Visualize the results for the external datasets

Run the *visualize_external_nonlinear.py* file to plot the results of the external evaluation:
```bash
python visualize_external_nonlinear.py
```

### Optional, go through the other scripts

There are still some scripts left that were not mentioned above, which are used for plotting different comparisons and results.

