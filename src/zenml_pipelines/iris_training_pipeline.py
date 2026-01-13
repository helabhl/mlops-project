from zenml import pipeline
from src.zenml_steps.data_step import prepare_iris_dataset
from src.zenml_steps.train_step import train_iris_model
from src.zenml_steps.eval_step import summarize_iris_experiment


@pipeline
def iris_training_pipeline(
    n_estimators: int = 100,
    max_depth: int = 5,
    seed: int = 42,
    exp_name: str = "zenml_iris_experiment",
):
    """Pipeline ZenML complet pour Iris + RandomForest.

    Steps :
    1) Préparer / vérifier le dataset Iris (via DVC).
    2) Entraîner le modèle RandomForest (logs MLflow).
    3) Afficher un résumé et pointer vers MLflow.
    """
    prepare_iris_dataset()
    exp = train_iris_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        seed=seed,
        exp_name=exp_name
    )
    summarize_iris_experiment(exp)
