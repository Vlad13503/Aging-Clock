import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from src.preprocessing import load_celltype_data

def train_lightgbm_cv(df, n_folds=5, top_k_genes=5000):
    donors = df["donor_id"].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_models = []
    all_preds = []
    all_shap = []

    gene_cols = df.columns.difference(["age", "donor_id", "cell_id"])
    X_all = df[gene_cols].sparse.to_dense()
    variances = X_all.var().sort_values(ascending=False)
    top_genes = variances.head(top_k_genes).index.tolist()

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(donors)):
        train_donors = donors[train_idx]
        test_donors = donors[test_idx]
        train_df = df[df["donor_id"].isin(train_donors)]
        test_df = df[df["donor_id"].isin(test_donors)]

        X_train = train_df[top_genes].sparse.to_dense()
        X_test = test_df[top_genes].sparse.to_dense()
        y_train = train_df["age"].values

        model = lgb.LGBMRegressor(
            objective="regression",
            metric="mae",
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            n_jobs=-1,
            verbosity=-1,
            random_state=fold_idx + 42
        )
        model.fit(X_train, y_train)
        all_models.append(model)

        y_pred = model.predict(X_test)
        pred_df = pd.DataFrame({
            "cell_id": test_df["cell_id"].values,
            "donor_id": test_df["donor_id"].values,
            "true_age": test_df["age"].values,
            "predicted_age": y_pred
        })
        all_preds.append(pred_df)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "gene": top_genes,
            "mean_abs_shap": mean_abs_shap,
            "fold": fold_idx + 1
        })
        all_shap.append(shap_df)

        print(f"[Fold {fold_idx+1}] MAE = {mean_absolute_error(test_df['age'], y_pred):.2f}")

    return all_models, pd.concat(all_preds, ignore_index=True), pd.concat(all_shap, ignore_index=True)

def main_lightgbm(h5ad_path, output_model_dir, output_pred_dir, cell_types):
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(output_pred_dir, exist_ok=True)

    for cell_type in cell_types:
        print(f"\nTraining for cell type: {cell_type}")
        df = load_celltype_data(h5ad_path, cell_type)
        models, preds_df, shap_df = train_lightgbm_cv(df)

        preds_df.to_csv(os.path.join(output_pred_dir, f"{cell_type}_predictions.csv"), index=False)
        shap_df.to_csv(os.path.join(output_pred_dir, f"{cell_type}_shap_summary.csv"), index=False)

        for i, model in enumerate(models):
            model.booster_.save_model(os.path.join(output_model_dir, f"{cell_type}_model_fold{i+1}.txt"))

        print(f"Saved predictions, SHAP, and models for {cell_type}")

if __name__ == "__main__":
    h5ad_path = "../data/AIDA.h5ad"
    output_model_dir = "../nonlinear_models/"
    output_pred_dir = "../predictions_nonlinear/"
    cell_types = [f.replace("Impute_avg_", "").replace(".csv", "") for f in os.listdir("../data_for_imputation") if f.endswith(".csv")]
    main_lightgbm(h5ad_path, output_model_dir, output_pred_dir, cell_types)
