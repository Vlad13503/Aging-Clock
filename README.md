# Aging-Clock

## Overview

This project builds and evaluates cell-type-specific transcriptomic aging clocks using single-cell RNA-seq data.  
Predictions are made using pretrained ElasticNet models trained on the AIDA dataset.
The project also supports external validation across independent scRNA-seq datasets such as:
- Yoshida
- Liu
- eQTL
- Stephenson

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
