# src/optuna_iris.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import optuna
import mlflow
import mlflow.sklearn

# ---------------------------
# Chargement dataset
# ---------------------------
def load_dataset(path="data/iris.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_test, y_train, y_test = load_dataset()

# ---------------------------
# Fonction objectif pour Optuna
# ---------------------------
def objective(trial):
    # Hyperparamètres à optimiser
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    
    # MLflow start run
    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        })

        # Modèle
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log modèle
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="rf_iris_optuna")

    # Optuna maximise la métrique, donc on retourne accuracy
    return accuracy

# ---------------------------
# Main
# ---------------------------
def main():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("iris_optuna_experiment")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("✅ Best hyperparameters:", study.best_params)
    print("✅ Best accuracy:", study.best_value)

if __name__ == "__main__":
    main()
