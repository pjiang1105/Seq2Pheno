#!/usr/bin/env python3
"""
Train a user-specific Seq2Pheno reference bundle.

The generated output is intentionally separate from the bundled Ref directory.
Use the resulting <output-dir>/Ref with the inference pipeline by setting:

    SEQ2PHENO_REF_DIR=<output-dir>/Ref snakemake --cores 8
"""

import argparse
import json
import math
import os
import shutil
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


MODEL_FILE_NAMES = {
    "Random Forest": "final_random_forest_model.pkl",
    "Gradient Boosting": "final_gradient_boosting_model.pkl",
    "SVR": "final_svr_model.pkl",
    "Ridge Regression": "final_ridge_regression_model.pkl",
    "Lasso Regression": "final_lasso_regression_model.pkl",
    "Neural Network": "final_neural_network_model.h5",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a custom Seq2Pheno autoencoder and phenotype-prediction reference bundle."
    )
    parser.add_argument("--expression-cpm", required=True, help="Training CPM matrix TSV. First column is sample IDs.")
    parser.add_argument("--phenotypes", required=True, help="Phenotype TSV with sample IDs and one or more phenotype columns.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: User_Training_<timestamp>.")
    parser.add_argument("--expression-sample-column", default=None, help="Sample column in expression file. Default: first column.")
    parser.add_argument("--phenotype-sample-column", default=None, help="Sample column in phenotype file. Default: first column.")
    parser.add_argument("--phenotype-columns", default=None, help="Comma-separated phenotype columns. Default: all numeric non-sample columns.")
    parser.add_argument("--latent-dim", type=int, default=32, help="Autoencoder latent dimension. Default: 32.")
    parser.add_argument("--autoencoder-epochs", type=int, default=100, help="Autoencoder training epochs. Default: 100.")
    parser.add_argument("--autoencoder-batch-size", type=int, default=32, help="Autoencoder batch size. Default: 32.")
    parser.add_argument("--predictor-epochs", type=int, default=200, help="Neural-network predictor epochs. Default: 200.")
    parser.add_argument("--predictor-batch-size", type=int, default=16, help="Neural-network predictor batch size. Default: 16.")
    parser.add_argument("--cv-splits", type=int, default=5, help="Cross-validation folds. Default: 5.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed. Default: 42.")
    parser.add_argument("--skip-neural-network", action="store_true", help="Skip neural-network phenotype model training.")
    parser.add_argument("--n-estimators", type=int, default=100, help="Tree estimators for RF/GB models. Default: 100.")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def sample_column(df, requested):
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Sample column '{requested}' not found.")
        return requested
    return df.columns[0]


def read_expression(path, sample_col):
    df = pd.read_csv(path, sep="\t")
    sample_col = sample_column(df, sample_col)
    genes = [col for col in df.columns if col != sample_col]
    if not genes:
        raise ValueError("Expression matrix must contain at least one gene column.")
    if df[sample_col].duplicated().any():
        duplicated = df.loc[df[sample_col].duplicated(), sample_col].head().tolist()
        raise ValueError(f"Expression sample IDs must be unique. Duplicates include: {duplicated}")

    numeric = df[genes].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    numeric[numeric < 0] = 0.0
    expression = pd.concat([df[[sample_col]].rename(columns={sample_col: "Sample"}), numeric], axis=1)
    return expression, genes


def read_phenotypes(path, sample_col):
    df = pd.read_csv(path, sep="\t")
    sample_col = sample_column(df, sample_col)
    if df[sample_col].duplicated().any():
        duplicated = df.loc[df[sample_col].duplicated(), sample_col].head().tolist()
        raise ValueError(f"Phenotype sample IDs must be unique. Duplicates include: {duplicated}")
    return df.rename(columns={sample_col: "Sample"})


def write_reference_files(expression, genes, ref_dir):
    ensure_dir(ref_dir)
    pd.DataFrame({"Gene": genes}).to_csv(os.path.join(ref_dir, "Gene_names_with_order.tsv"), sep="\t", index=False)

    values = expression[genes].to_numpy(dtype=np.float32)
    ranked = np.sort(values, axis=1)[:, ::-1]
    rank_means = ranked.mean(axis=0)
    ref = pd.DataFrame({"Rank": np.arange(1, len(genes) + 1), "CPM.Ave": rank_means})
    ref.to_csv(os.path.join(ref_dir, "Reference_Rank_Ave_CPM.tsv"), sep="\t", index=False)
    return rank_means


def rank_normalize(values, ref_cpm):
    values = values.astype(np.float32)
    ranks = values.argsort(axis=1)[:, ::-1].argsort(axis=1) + 1
    ref_with_zero = np.insert(ref_cpm, 0, 0.0).astype(np.float32)
    normalized = np.zeros_like(values, dtype=np.float32)
    non_zero = values != 0
    normalized[non_zero] = ref_with_zero[ranks[non_zero]]
    return np.round(normalized, 2)


def preprocess_expression(expression, genes, ref_cpm, output_dir):
    ensure_dir(output_dir)
    ordered_path = os.path.join(output_dir, "training.gene_filter_order.tsv")
    norm_path = os.path.join(output_dir, "training.Ref_Normalized_CPM.tsv")
    log_path = os.path.join(output_dir, "training.log10_Ref_Normalized_CPM.tsv")

    expression.to_csv(ordered_path, sep="\t", index=False)
    values = expression[genes].to_numpy(dtype=np.float32)
    normalized = rank_normalize(values, ref_cpm)
    normalized_df = pd.DataFrame(normalized, columns=genes)
    normalized_df.insert(0, "Sample", expression["Sample"].values)
    normalized_df.to_csv(norm_path, sep="\t", index=False, float_format="%.2f")

    log_values = np.log10(normalized + 1)
    log_df = pd.DataFrame(log_values, columns=genes)
    log_df.insert(0, "Sample", expression["Sample"].values)
    log_df.to_csv(log_path, sep="\t", index=False, float_format="%.4f")
    return log_path


def train_autoencoder(log_expression_path, genes, ref_dir, output_dir, args):
    from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    ensure_dir(output_dir)
    df = pd.read_csv(log_expression_path, sep="\t")
    sample_ids = df["Sample"]
    x = df[genes].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    scaler_params = pd.DataFrame({"Gene": genes, "mean": scaler.mean_, "scale": scaler.scale_})
    scaler_params.to_csv(os.path.join(ref_dir, "scaler_parameters.tsv"), sep="\t", index=False)

    input_layer = Input(shape=(x_scaled.shape[1],))
    encoded = Dense(1024, activation="relu")(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(256, activation="relu")(encoded)
    bottleneck = Dense(args.latent_dim, activation="linear", name="latent_space")(encoded)
    decoded = Dense(256, activation="relu")(bottleneck)
    decoded = Dense(1024, activation="relu")(decoded)
    output_layer = Dense(x_scaled.shape[1], activation="linear")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=bottleneck)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    callbacks = [CSVLogger(os.path.join(output_dir, "training_history.tsv"), separator="\t")]
    if x_scaled.shape[0] >= 10:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True))
        validation_split = 0.1
    else:
        validation_split = 0.0

    autoencoder.fit(
        x_scaled,
        x_scaled,
        epochs=args.autoencoder_epochs,
        batch_size=args.autoencoder_batch_size,
        shuffle=True,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    autoencoder.save(os.path.join(output_dir, "autoencoder_model.h5"))
    encoder.save(os.path.join(output_dir, "encoder_model.h5"))
    shutil.copy2(os.path.join(output_dir, "encoder_model.h5"), os.path.join(ref_dir, "encoder_model.h5"))

    encoded_features = encoder.predict(x_scaled)
    encoded_df = pd.DataFrame(encoded_features, columns=[f"Latent_{i + 1}" for i in range(encoded_features.shape[1])])
    encoded_df.insert(0, "Sample", sample_ids)
    encoded_path = os.path.join(output_dir, "encoded_features.tsv")
    encoded_df.to_csv(encoded_path, sep="\t", index=False)

    reconstructed = autoencoder.predict(x_scaled)
    reconstructed_df = pd.DataFrame(reconstructed, columns=genes)
    reconstructed_df.insert(0, "Sample", sample_ids)
    reconstructed_df.to_csv(os.path.join(output_dir, "reconstructed_data.tsv"), sep="\t", index=False)

    mse = np.mean((x_scaled - reconstructed) ** 2, axis=1)
    pd.DataFrame({"Sample": sample_ids, "Reconstruction_MSE": mse}).to_csv(
        os.path.join(output_dir, "reconstruction_errors.tsv"), sep="\t", index=False
    )
    return encoded_path


def selected_phenotypes(phenotype_df, requested):
    if requested:
        phenotypes = [item.strip() for item in requested.split(",") if item.strip()]
        missing = [col for col in phenotypes if col not in phenotype_df.columns]
        if missing:
            raise ValueError(f"Phenotype columns not found: {missing}")
        return phenotypes

    phenotypes = []
    for col in phenotype_df.columns:
        if col == "Sample":
            continue
        numeric = pd.to_numeric(phenotype_df[col], errors="coerce")
        if numeric.notna().sum() >= 2:
            phenotypes.append(col)
    if not phenotypes:
        raise ValueError("No numeric phenotype columns were found.")
    return phenotypes


def phenotype_dir_name(phenotype):
    return phenotype.replace("/", "_").replace("\\", "_")


def build_ml_models(args):
    return {
        "Random Forest": RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.random_seed),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=args.n_estimators, random_state=args.random_seed),
        "SVR": SVR(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(max_iter=10000),
    }


def build_nn(input_dim):
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.regularizers import l2

    model = Sequential(
        [
            Dense(128, activation="relu", kernel_regularizer=l2(0.001), input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def score_predictions(y_true, y_pred):
    rho, _ = spearmanr(y_true, y_pred)
    if isinstance(rho, float) and math.isnan(rho):
        rho = np.nan
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rho, rmse


def cross_validate_nn(x, y, kf, args):
    predictions = np.zeros(len(y), dtype=np.float32)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train = y.iloc[train_index]
        model = build_nn(x_train.shape[1])

        callbacks = []
        validation_split = 0.0
        if len(train_index) >= 10:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True))
            validation_split = 0.2

        model.fit(
            x_train,
            y_train,
            epochs=args.predictor_epochs,
            batch_size=args.predictor_batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0,
        )
        predictions[test_index] = model.predict(x_test, verbose=0).flatten()
    return predictions


def train_phenotype_models(encoded_path, phenotype_df, phenotypes, ref_dir, output_dir, args):
    ensure_dir(output_dir)
    encoded_df = pd.read_csv(encoded_path, sep="\t")
    merged = pd.merge(phenotype_df, encoded_df, on="Sample", how="inner")
    feature_cols = [col for col in merged.columns if col.startswith("Latent_")]
    if not feature_cols:
        raise ValueError("Encoded features must contain Latent_* columns.")

    summary_rows = []
    for phenotype in phenotypes:
        phenotype_output = ensure_dir(os.path.join(output_dir, phenotype_dir_name(phenotype)))
        phenotype_ref = ensure_dir(os.path.join(ref_dir, phenotype_dir_name(phenotype)))

        data = merged[["Sample", phenotype] + feature_cols].copy()
        data[phenotype] = pd.to_numeric(data[phenotype], errors="coerce")
        data = data.dropna(subset=[phenotype])
        if data.shape[0] < 3:
            print(f"Skipping phenotype '{phenotype}': fewer than 3 matched non-missing samples.")
            continue

        x = data[feature_cols]
        y = data[phenotype]
        cv_splits = min(args.cv_splits, data.shape[0])
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=args.random_seed)

        predictions_df = data[["Sample", phenotype]].copy()
        performance_rows = []

        for name, model in build_ml_models(args).items():
            y_pred = cross_val_predict(model, x, y, cv=kf)
            predictions_df[f"{name}.Pred"] = y_pred
            rho, rmse = score_predictions(y, y_pred)
            performance_rows.append({"Model": name, "Spearman_Rho": rho, "RMSE": rmse})

            model.fit(x, y)
            model_path = os.path.join(phenotype_output, MODEL_FILE_NAMES[name])
            joblib.dump(model, model_path)
            shutil.copy2(model_path, os.path.join(phenotype_ref, MODEL_FILE_NAMES[name]))

        if not args.skip_neural_network:
            y_pred = cross_validate_nn(x, y, kf, args)
            predictions_df["Neural Network.Pred"] = y_pred
            rho, rmse = score_predictions(y, y_pred)
            performance_rows.append({"Model": "Neural Network", "Spearman_Rho": rho, "RMSE": rmse})

            nn_final = build_nn(x.shape[1])
            nn_final.fit(x, y, epochs=args.predictor_epochs, batch_size=args.predictor_batch_size, verbose=0)
            nn_path = os.path.join(phenotype_output, MODEL_FILE_NAMES["Neural Network"])
            nn_final.save(nn_path)
            shutil.copy2(nn_path, os.path.join(phenotype_ref, MODEL_FILE_NAMES["Neural Network"]))

        predictions_df.to_csv(os.path.join(phenotype_output, "2_Predicted_vs_Observed.tsv"), sep="\t", index=False)

        performance = pd.DataFrame(performance_rows)
        performance["Spearman_Rho_for_selection"] = performance["Spearman_Rho"].fillna(-np.inf)
        performance = performance.sort_values(
            by=["Spearman_Rho_for_selection", "RMSE"], ascending=[False, True]
        ).drop(columns=["Spearman_Rho_for_selection"])
        performance.to_csv(
            os.path.join(phenotype_output, "2_crossval_performance_metrics_all_models.tsv"), sep="\t", index=False
        )

        best_model = performance.iloc[0]["Model"]
        with open(os.path.join(phenotype_output, "Best_Model.txt"), "w") as handle:
            handle.write(best_model)
        with open(os.path.join(phenotype_ref, "Best_Model.txt"), "w") as handle:
            handle.write(best_model)

        summary_rows.append(
            {
                "Phenotype": phenotype,
                "Ref_Folder": phenotype_dir_name(phenotype),
                "N_Samples": data.shape[0],
                "Best_Model": best_model,
                "Best_Spearman_Rho": performance.iloc[0]["Spearman_Rho"],
                "Best_RMSE": performance.iloc[0]["RMSE"],
            }
        )

    if not summary_rows:
        raise ValueError("No phenotype models were trained. Check sample matching and phenotype columns.")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(output_dir, "phenotype_training_summary.tsv"), sep="\t", index=False)
    return summary


def write_manifest(path, args, phenotypes, summary):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "expression_cpm": os.path.abspath(args.expression_cpm),
        "phenotypes": os.path.abspath(args.phenotypes),
        "phenotype_columns": phenotypes,
        "output_dir": os.path.abspath(args.output_dir),
        "custom_ref_dir": os.path.abspath(os.path.join(args.output_dir, "Ref")),
        "latent_dim": args.latent_dim,
        "autoencoder_epochs": args.autoencoder_epochs,
        "cv_splits": args.cv_splits,
        "trained_phenotypes": summary.to_dict(orient="records"),
    }
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2)


def main():
    args = parse_args()
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow is required for Seq2Pheno training because the autoencoder is a Keras model. "
            "Install TensorFlow or activate the Seq2Pheno training environment, then rerun this command."
        ) from exc

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    if args.output_dir is None:
        args.output_dir = f"User_Training_{timestamp}"

    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    ref_dir = ensure_dir(os.path.join(args.output_dir, "Ref"))
    preprocessed_dir = ensure_dir(os.path.join(args.output_dir, "Preprocessed"))
    autoencoder_dir = ensure_dir(os.path.join(args.output_dir, "Autoencoder"))
    phenotype_model_dir = ensure_dir(os.path.join(args.output_dir, "Phenotype_Models"))

    expression, genes = read_expression(args.expression_cpm, args.expression_sample_column)
    phenotypes_df = read_phenotypes(args.phenotypes, args.phenotype_sample_column)
    phenotypes = selected_phenotypes(phenotypes_df, args.phenotype_columns)

    print(f"Training expression samples: {expression.shape[0]}")
    print(f"Training genes: {len(genes)}")
    print(f"Phenotypes: {', '.join(phenotypes)}")

    ref_cpm = write_reference_files(expression, genes, ref_dir)
    log_expression_path = preprocess_expression(expression, genes, ref_cpm, preprocessed_dir)
    encoded_path = train_autoencoder(log_expression_path, genes, ref_dir, autoencoder_dir, args)
    summary = train_phenotype_models(encoded_path, phenotypes_df, phenotypes, ref_dir, phenotype_model_dir, args)
    write_manifest(os.path.join(args.output_dir, "training_manifest.json"), args, phenotypes, summary)

    print("\nTraining complete.")
    print(f"Custom reference directory: {os.path.abspath(ref_dir)}")
    print(f"Summary: {os.path.abspath(os.path.join(phenotype_model_dir, 'phenotype_training_summary.tsv'))}")


if __name__ == "__main__":
    main()
