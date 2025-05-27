import os

import numpy as np
import pandas as pd
from scipy import sparse

from preprocessing import load_celltype_data as load_data_by_celltype
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold

def train_and_predict_by_fold(df, n_folds=5):
    donors = df["donor_id"].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    models = []
    test_preds = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(donors)):
        train_donors = donors[train_idx]
        test_donors = donors[test_idx]

        train_df = df[df["donor_id"].isin(train_donors)].copy()
        test_df = df[df["donor_id"].isin(test_donors)].copy()
        y_train = train_df["age"]

        gene_cols = df.columns.difference(["age", "donor_id", "cell_id"])
        X_all = df[gene_cols]
        X_sparse = sparse.csr_matrix(X_all.sparse.to_coo())

        mean = X_sparse.mean(axis=0).A1
        sq_mean = X_sparse.multiply(X_sparse).mean(axis=0).A1
        variances = sq_mean - mean ** 2

        N_TOP_GENES = 5000
        top_gene_indices = np.argsort(variances)[-N_TOP_GENES:]
        top_genes = X_all.columns[top_gene_indices]

        X_train = train_df[top_genes].copy()
        X_test = test_df[top_genes].copy()

        X_train = X_train.sparse.to_dense()
        X_test = X_test.sparse.to_dense()

        if X_train.shape[1] == 0:
            print(f"[Fold {fold_idx + 1}] Skipping: no genes left after variance filtering.")
            continue

        means = X_train.mean()
        stds = X_train.std(ddof=0) + 1e-8
        X_train = (X_train - means) / stds
        X_test = (X_test - means) / stds

        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=[0.1, 1.0, 10.0],
            cv=5,
            max_iter=10000,
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train, y_train)

        model_coeffs = pd.Series(model.coef_, index=X_train.columns)
        model_df = model_coeffs.to_frame().T
        model_df["intercept"] = model.intercept_
        models.append(model_df)

        y_pred = model.predict(X_test)
        pred_df = pd.DataFrame({
            "cell_id": test_df["cell_id"],
            "donor_id": test_df["donor_id"],
            "true_age": test_df["age"],
            "predicted_age": y_pred
        })
        test_preds.append(pred_df)

        print(
            f"[Fold {fold_idx + 1}] α={model.alpha_:.4f}, l1_ratio={model.l1_ratio_:.2f}, genes used: {len(X_train.columns)}")

    return pd.concat(models, ignore_index=True), pd.concat(test_preds, ignore_index=True)

def main(h5ad_path, output_model_dir, output_pred_dir, cell_types):
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(output_pred_dir, exist_ok=True)

    for cell_type in cell_types:
        print(f"Processing: {cell_type}")
        df = load_data_by_celltype(h5ad_path, cell_type)
        models_df, preds_df = train_and_predict_by_fold(df)

        models_df.to_csv(os.path.join(output_model_dir, f"{cell_type}_models5.csv"), index=False)
        preds_df.to_csv(os.path.join(output_pred_dir, f"{cell_type}_predictions.csv"), index=False)

        print(f"Saved models and predictions for {cell_type}")

if __name__ == "__main__":
    h5ad_path = "../data/AIDA.h5ad"
    output_model_dir = "../models_enhanced/"
    output_pred_dir = "../predictions_enhanced/"
    cell_types = [f.replace("Impute_avg_", "").replace(".csv", "") for f in os.listdir("../data_for_imputation") if f.endswith(".csv")]
    main(h5ad_path, output_model_dir, output_pred_dir, cell_types)
