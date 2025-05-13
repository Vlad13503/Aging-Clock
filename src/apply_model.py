import pandas as pd
import argparse
import os
from preprocessing import load_celltype_data

def apply_model(cell_type, model_folder, imputation_folder, data_path, output_folder="./predictions/"):
    """
    Apply the pretrained ElasticNet models to predict age for a given cell type.

    Parameters:
        cell_type (str): The cell type we want to analyze.
        model_folder (str): Path to folder containing model CSVs.
        imputation_folder (str): Path to folder containing imputation CSVs.
        data_path (str): Path to the .h5ad data file.
        output_folder (str): Folder to save prediction results.
    """
    df = load_celltype_data(data_path, cell_type)

    impute_file = os.path.join(imputation_folder, f"Impute_avg_{cell_type}.csv")
    if not os.path.exists(impute_file):
        print(f"Imputation file for {cell_type} not found, skipping.")
        return

    model_file = os.path.join(model_folder, f"{cell_type}_models5.csv")
    if not os.path.exists(model_file):
        print(f"Model file for {cell_type} not found, skipping.")
        return

    impute = pd.read_csv(impute_file, index_col=0)
    models = pd.read_csv(model_file, index_col=0)

    results = []

    for i in range(len(models)):
        model = models.iloc[i].dropna()
        model_coeff = model.drop('intercept')
        model_intercept = model['intercept']

        model_genes = model_coeff.index
        available_genes = df.columns.intersection(model_genes)
        missing_genes = model_genes.difference(df.columns)

        input_data = df[available_genes].copy()

        if not missing_genes.empty:
            imputed_values = {}
            for gene in missing_genes:
                if gene in impute.columns:
                    imputed_values[gene] = impute[gene].values[0]
                else:
                    imputed_values[gene] = 0.0

            imputed_df = pd.DataFrame(
                {gene: [value] * len(input_data) for gene, value in imputed_values.items()},
                index=input_data.index
            )

            input_data = pd.concat([input_data, imputed_df], axis=1)

            print(f"Imputed {len(missing_genes)} missing genes for model {i + 1} for {cell_type}.")

        input_data = input_data[model_genes]

        preds = input_data.dot(model_coeff.values) + model_intercept

        temp_df = pd.DataFrame({
            "cell_id": df["cell_id"],
            "predicted_age": preds,
            "true_age": df["age"],
            "donor_id": df["donor_id"]
        })
        results.append(temp_df)

    # final_results = pd.concat(results).reset_index(drop=True)
    #
    # os.makedirs(output_folder, exist_ok=True)
    # output_file = os.path.join(output_folder, f"predictions_{cell_type}.csv")
    # final_results.to_csv(output_file, index=False)

    final_results = pd.concat(results)
    aggregated = final_results.groupby("cell_id").agg({
        "predicted_age": "mean",
        "true_age": "first",
        "donor_id": "first"
    }).reset_index()

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"predictions_{cell_type}.csv")
    aggregated.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply pre-trained ElasticNet model to new data")
    parser.add_argument("--model_folder", type=str, default="../models/", help="Folder containing trained models")
    parser.add_argument("--imputation_folder", type=str, default="../data_for_imputation/", help="Folder containing imputation files")
    parser.add_argument("--data_path", type=str, default="../data/AIDA.h5ad", help="Path to the input .h5ad file")
    parser.add_argument("--output_folder", type=str, default="../predictions/", help="Folder to save predictions")

    args = parser.parse_args()

    cell_types = [f.replace("Impute_avg_", "").replace(".csv", "") for f in os.listdir(args.imputation_folder) if f.endswith(".csv")]

    for cell_type in cell_types:
        try:
            apply_model(cell_type, args.model_folder, args.imputation_folder, args.data_path, args.output_folder)
        except Exception as e:
            print(f"Failed for {cell_type}: {e}")

    print("Done applying models for all available cell types!")
