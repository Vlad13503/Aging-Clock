import os
import pandas as pd
from preprocessing import load_celltype_data as load_data_by_celltype
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold

def train_and_predict_by_fold(df, n_folds=5, alpha=1.0, l1_ratio=0.5):
    donors = df["donor_id"].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    models = []
    test_preds = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(donors)):
        train_donors = donors[train_idx]
        test_donors = donors[test_idx]

        train_df = df[df["donor_id"].isin(train_donors)].copy()
        test_df = df[df["donor_id"].isin(test_donors)].copy()

        X_train = train_df.drop(columns=["age", "donor_id", "cell_id"])
        y_train = train_df["age"]
        X_test = test_df.drop(columns=["age", "donor_id", "cell_id"])

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
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
    output_model_dir = "../models/"
    output_pred_dir = "../predictions/"
    cell_types = [f.replace("Impute_avg_", "").replace(".csv", "") for f in os.listdir("../data_for_imputation") if f.endswith(".csv")]
    main(h5ad_path, output_model_dir, output_pred_dir, cell_types)
