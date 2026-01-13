from __future__ import annotations

"""
Script de lancement d'une "grille" de runs du pipeline ZenML pour Iris.

On lance plusieurs exécutions du pipeline avec différents hyperparamètres
(n_estimators, max_depth, seed, exp_name).
"""

from src.zenml_pipelines.iris_training_pipeline import iris_training_pipeline


def main() -> None:
    """Lance plusieurs runs du pipeline Iris avec différents paramètres."""

    # Liste de configurations à tester
    configs = [
        {"n_estimators": 50, "max_depth": 3, "seed": 42, "exp_name": "zenml_iris_RF_50_3_s42"},
        {"n_estimators": 50, "max_depth": 5, "seed": 123, "exp_name": "zenml_iris_RF_50_5_s123"},
        {"n_estimators": 100, "max_depth": 3, "seed": 42, "exp_name": "zenml_iris_RF_100_3_s42"},
        {"n_estimators": 100, "max_depth": 5, "seed": 123, "exp_name": "zenml_iris_RF_100_5_s123"},
    ]

    for cfg in configs:
        print(f"[ZenML] Lancement pipeline avec paramètres : {cfg}")
        # Chaque appel crée un nouveau run de pipeline dans ZenML
        iris_training_pipeline(**cfg)


if __name__ == "__main__":
    main()
