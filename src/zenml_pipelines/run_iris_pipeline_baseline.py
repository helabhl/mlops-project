from __future__ import annotations

"""
Script de lancement du pipeline ZenML Iris en mode "baseline".

Le pipeline ZenML déclenche directement un run,
les métriques vont dans MLflow, les artefacts dans MinIO.
"""

from .iris_training_pipeline import iris_training_pipeline  # ton pipeline ZenML Iris


def main() -> None:
    """Lance un run baseline du pipeline Iris via ZenML."""
    # Exécution baseline avec des paramètres simples
    iris_training_pipeline(
        n_estimators=50,
        max_depth=3,
        seed=42,
        exp_name="zenml_iris_baseline",
    )


if __name__ == "__main__":
    main()
