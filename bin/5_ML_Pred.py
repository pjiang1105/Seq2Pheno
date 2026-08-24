import os
import pandas as pd
import joblib
import sys

MODEL_FILES = {
    "random_forest": "final_random_forest_model.pkl",
    "gradient_boosting": "final_gradient_boosting_model.pkl",
    "svr": "final_svr_model.pkl",
    "lasso_regression": "final_lasso_regression_model.pkl",
    "ridge_regression": "final_ridge_regression_model.pkl",
    "neural_network": "final_neural_network_model.h5",
}

DEFAULT_PHENOTYPE_ORDER = [
    "a-SMA",
    "CD163",
    "IBA-1",
    "MAC387",
    "PAX7",
    "Semi-fibrosis-score",
    "TNF-a",
    "TZ_Percentage_Fibrosis",
]


def find_phenotype_folders(ref_dir):
    discovered = []
    for name in os.listdir(ref_dir):
        folder = os.path.join(ref_dir, name)
        if os.path.isdir(folder) and os.path.exists(os.path.join(folder, "Best_Model.txt")):
            discovered.append(name)
    folders = [name for name in DEFAULT_PHENOTYPE_ORDER if name in discovered]
    folders.extend(sorted(name for name in discovered if name not in folders))
    if not folders:
        raise ValueError(f"No phenotype folders with Best_Model.txt found under '{ref_dir}'.")
    return folders


def load_best_model(ref_dir, phenotype):
    best_model_path = os.path.join(ref_dir, phenotype, "Best_Model.txt")

    with open(best_model_path, 'r') as f:
        best_model_name = f.read().strip().lower().replace(" ", "_")

    model_filename = MODEL_FILES.get(best_model_name)
    if not model_filename:
        raise ValueError(f"Unsupported best model '{best_model_name}' for phenotype '{phenotype}'.")

    model_file = os.path.join(ref_dir, phenotype, model_filename)
    if not os.path.exists(model_file):
        raise ValueError(f"Model file '{model_file}' not found for the best model '{best_model_name}'.")

    if model_file.endswith(".h5"):
        from tensorflow.keras.models import load_model

        return load_model(model_file)
    return joblib.load(model_file)


def main(encoded_features_file, outfile_name, ref_dir="./Ref"):
    encoded_df = pd.read_csv(encoded_features_file, sep='\t', index_col=0)
    compiled_predictions = pd.DataFrame(index=encoded_df.index)

    for phenotype in find_phenotype_folders(ref_dir):
        model = load_best_model(ref_dir, phenotype)
        predictions = model.predict(encoded_df)
        compiled_predictions[phenotype] = pd.Series(predictions).to_numpy().reshape(-1)

    compiled_predictions.reset_index().rename(columns={'index': 'Sample'}).to_csv(outfile_name, sep='\t', index=False)


if __name__ == "__main__":
    if len(sys.argv) not in [3, 5]:
        raise SystemExit("Usage: python3 ./bin/5_ML_Pred.py ENCODED_FEATURES.tsv OUT.tsv [--ref-dir REF_DIR]")

    encoded_features_file = sys.argv[1]
    outfile_name = sys.argv[2]
    ref_dir = "./Ref"
    if len(sys.argv) == 5:
        if sys.argv[3] != "--ref-dir":
            raise SystemExit("Optional argument must be --ref-dir REF_DIR")
        ref_dir = sys.argv[4]

    main(encoded_features_file, outfile_name, ref_dir)
