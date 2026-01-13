import os
import itertools
import joblib
import mlflow
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from pathlib import Path

# ---------------------------
# Utilitaires
# ---------------------------
def load_dataset(path="data/iris.csv"):
    """
    Charge un dataset CSV simple.
    Pour l'exemple, Iris dataset utilisé.
    """
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def create_run_dir(base="runs"):
    """
    Crée un dossier pour sauvegarder les modèles locaux
    """
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, default="iris_classification", help="Nom de l'expérience MLflow")
    ap.add_argument("--n_estimators", nargs="+", type=int, default=[50, 100])
    ap.add_argument("--max_depth", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    ap.add_argument("--dataset", type=str, default="data/iris.csv")
    ap.add_argument("--run_dir", type=str, default="runs")
    args = ap.parse_args()

    # ---------------------------
    # MLflow setup
    # ---------------------------
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(args.experiment)

    # ---------------------------
    # Chargement du dataset
    # ---------------------------
    X_train, X_test, y_train, y_test = load_dataset(args.dataset)
    run_base_dir = create_run_dir(args.run_dir)

    

    # ---------------------------
    # Boucle sur hyperparamètres
    # ---------------------------
    configs = list(itertools.product(args.n_estimators, args.max_depth, args.seeds))

    for n_estimators, max_depth, seed in configs:
        run_name = f"RF_{n_estimators}_{max_depth}_s{seed}"
        print(f"--- Starting run: {run_name} ---")

        with mlflow.start_run(run_name=run_name):
            # Log params
            mlflow.log_params({
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "seed": seed,
            })

            # Modèle
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=seed
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="macro"))
            mlflow.log_metric("precision", precision_score(y_test, y_pred, average="macro"))
            mlflow.log_metric("recall", recall_score(y_test, y_pred, average="macro"))

            # Log model artifact
            model_file = run_base_dir / f"{run_name}_model.pkl"
            joblib.dump(model, model_file)
            mlflow.log_artifact(str(model_file), artifact_path="models")

        print(f"--- Finished run: {run_name} ---\n")

    print("✅ All runs finished. Check MLflow UI at:", mlflow_uri)

# ---------------------------
# Script entry point
# ---------------------------
if __name__ == "__main__":
    main()
