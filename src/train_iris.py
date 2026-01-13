# src/train_iris.py
import os
import argparse
import itertools
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow
import mlflow.sklearn
from mlflow.data import from_pandas


# ---------------------------
# Utilitaires
# ---------------------------
def load_dataset(path="data/iris.csv"):
    """Charge le dataset Iris depuis un CSV et retourne un split train/test."""
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def create_run_dir(base="runs"):
    """Crée un dossier pour sauvegarder les modèles locaux."""
    run_dir = Path(base)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------
# Main
# ---------------------------
def main():
    # ---------------------------
    # Arguments CLI
    # ---------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, nargs="+", default=[50])
    parser.add_argument("--max-depth", type=int, nargs="+", default=[3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--dataset", type=str, default="data/iris.csv")
    parser.add_argument("--run-dir", type=str, default="runs/train")
    parser.add_argument("--experiment", type=str, default="baseline")
    args = parser.parse_args()

    # ---------------------------
    # Setup MLflow
    # ---------------------------
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(args.experiment)

    # ---------------------------
    # Chargement dataset
    # ---------------------------
    X_train, X_test, y_train, y_test = load_dataset(args.dataset)
    run_base_dir = create_run_dir(args.run_dir)

    # Log dataset comme MLflow Dataset
    dataset = from_pandas(
        X_train.join(y_train),
        source=args.dataset,
        name="iris_v2"
    )

    # ---------------------------
    # Grille d'hyperparamètres
    # ---------------------------
    configs = list(itertools.product(args.n_estimators, args.max_depth, args.seeds))

    for n_estimators, max_depth, seed in configs:
        run_name = f"RF_{n_estimators}_{max_depth}_s{seed}"
        print(f"--- Starting run: {run_name} ---")

        with mlflow.start_run(run_name=run_name):
            # Log paramètres
            mlflow.log_params({
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "seed": seed,
            })

            # Log dataset d'entrée
            mlflow.log_input(dataset, context="training")

            # Entraînement
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=seed
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log métriques
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="macro"))
            mlflow.log_metric("precision", precision_score(y_test, y_pred, average="macro"))
            mlflow.log_metric("recall", recall_score(y_test, y_pred, average="macro"))

            # Log modèle
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="rf_classifier"
            )

            # Sauvegarde locale optionnelle
            model_file = run_base_dir / f"{run_name}_model.pkl"
            mlflow.sklearn.save_model(model, str(model_file))

        print(f"--- Finished run: {run_name} ---\n")

    print("✅ All runs finished. Check MLflow UI at:", mlflow_uri)


# ---------------------------
# Script entry point
# ---------------------------
if __name__ == "__main__":
    main()
