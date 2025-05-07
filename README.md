# Aging-Clock

## Overview

This project builds and evaluates cell-type-specific transcriptomic aging clocks using single-cell RNA-seq data.  
Predictions are made using pretrained ElasticNet models.

---

## Data Access

The large `.h5ad` dataset used in this project is not included in the repository due to its size (~10GB).

You can manually download it from:

- [AIDA Phase 1 Data Freeze v1](https://cellxgene.cziscience.com/collections/ced320a1-29f3-47c1-a735-513c7084d508)

Then:

1. Create a folder called `data/` in the root of the repo (if it doesn't exist).
2. Rename the file to `AIDA.h5ad`.
3. Place it in the `data/` folder.
