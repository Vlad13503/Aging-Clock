import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from src.preprocessing import load_celltype_data

def train_lightgbm_cv(df, n_folds=5, top_k_genes=5000, n_trials=50):
    """Train LightGBM with nested CV and hyperparameter tuning via Optuna."""
    donors = df["donor_id"].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_models = []
    all_preds = []
    all_shap = []
    all_best_params = []

    gene_cols = df.columns.difference(["age", "donor_id", "cell_id"])
    X_all = df[gene_cols].sparse.to_dense()
    variances = X_all.var().sort_values(ascending=False)
    top_genes = variances.head(top_k_genes).index.tolist()

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(donors)):
        print(f"\n=== Outer Fold {fold_idx+1}/{n_folds} ===")
        train_donors = donors[train_idx]
        test_donors = donors[test_idx]
        train_df = df[df["donor_id"].isin(train_donors)]
        test_df = df[df["donor_id"].isin(test_donors)]

        X_train = train_df[top_genes].sparse.to_dense()
        X_test = test_df[top_genes].sparse.to_dense()
        y_train = train_df["age"].values
        y_test = test_df["age"].values

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction="minimize", pruner=pruner)

        def objective(trial):
            param = {
                "objective": "regression",
                "metric": "mae",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "n_jobs": -1,
                "seed": 42 + fold_idx,
                "learning_rate": trial.suggest_categorical("learning_rate", [0.005, 0.01, 0.02, 0.05, 0.1]),
                "num_leaves": trial.suggest_categorical("num_leaves", [16, 32, 64, 128, 256, 512]),
                "feature_fraction": trial.suggest_categorical("feature_fraction", [0.6, 0.7, 0.8, 0.9, 1.0]),
                "bagging_fraction": trial.suggest_categorical("bagging_fraction", [0.6, 0.7, 0.8, 0.9, 1.0]),
                "bagging_freq": trial.suggest_categorical("bagging_freq", [1, 3, 5, 7, 10]),
                "min_child_samples": trial.suggest_categorical("min_child_samples", [5, 10, 20, 30, 40, 50]),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
                "lambda_l1": trial.suggest_categorical("lambda_l1", [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]),
                "lambda_l2": trial.suggest_categorical("lambda_l2", [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
            }

            dtrain = lgb.Dataset(X_train, label=y_train)
            cv_results = lgb.cv(
                param,
                dtrain,
                nfold=3,
                num_boost_round=500,
                stratified=False,
                seed=42 + fold_idx,
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0),
                    optuna.integration.LightGBMPruningCallback(trial, "l1")
                ]
            )
            if "valid mae-mean" in cv_results:
                return min(cv_results["valid mae-mean"])
            elif "valid l1-mean" in cv_results:
                return min(cv_results["valid l1-mean"])
            else:
                raise ValueError(f"Unexpected cv_results keys: {list(cv_results.keys())}")

        study.optimize(objective, n_trials=n_trials, n_jobs=16, show_progress_bar=True)

        print(f"Best MAE inner CV: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

        all_best_params.append(study.best_params)

        best_params = study.best_params
        best_params.update({
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_jobs": -1,
            "seed": 42 + fold_idx
        })

        model = lgb.LGBMRegressor(**best_params, n_estimators=5000)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(0)
            ]
        )
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

        fold_mae = mean_absolute_error(test_df["age"], y_pred)
        print(f"[Fold {fold_idx+1}] Outer fold MAE = {fold_mae:.2f}")

    return all_models, pd.concat(all_preds, ignore_index=True), pd.concat(all_shap, ignore_index=True), all_best_params

def main_lightgbm(h5ad_path, output_model_dir, output_pred_dir, cell_types, n_trials=50):
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(output_pred_dir, exist_ok=True)

    for cell_type in cell_types:
        print(f"\nTraining for cell type: {cell_type}")
        df = load_celltype_data(h5ad_path, cell_type)

        models, preds_df, shap_df, best_params_list = train_lightgbm_cv(df, n_folds=5, top_k_genes=3000, n_trials=n_trials)

        preds_df.to_csv(os.path.join(output_pred_dir, f"{cell_type}_predictions.csv"), index=False)
        shap_df.to_csv(os.path.join(output_pred_dir, f"{cell_type}_shap_summary.csv"), index=False)

        for i, (model, params) in enumerate(zip(models, best_params_list)):
            model.booster_.save_model(os.path.join(output_model_dir, f"{cell_type}_model_fold{i+1}.txt"))
            pd.Series(params).to_csv(os.path.join(output_model_dir, f"{cell_type}_model_fold{i+1}_best_params.csv"))

        print(f"Saved predictions, SHAP, and models for {cell_type}")

if __name__ == "__main__":
    h5ad_path = "../data/AIDA.h5ad"
    output_model_dir = "../nonlinear_models_enhanced/"
    output_pred_dir = "../predictions_nonlinear_enhanced/"
    cell_types = [f.replace("Impute_avg_", "").replace(".csv", "") for f in os.listdir("../data_for_imputation") if f.endswith(".csv")]

    main_lightgbm(h5ad_path, output_model_dir, output_pred_dir, cell_types, n_trials=30)
