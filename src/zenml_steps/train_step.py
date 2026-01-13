import subprocess
from typing import Optional
from zenml import step


@step
def train_iris_model(
    n_estimators: int = 100,
    max_depth: int = 3,            
    seed: int = 42,               
    exp_name: str = "zenml_iris_baseline",
) -> Optional[str]:
    """Lance l'entraînement Iris via src/train_iris.py.

    Les métriques sont loggées dans MLflow.
    """
    cmd = [
        "python",
        "-m",
        "src.train_iris",
        "--n-estimators", str(n_estimators),
        "--max-depth", str(max_depth),   
        "--seeds", str(seed),             
        "--experiment", exp_name,
    ]

    print(f"[train_iris_model] Commande : {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print("[train_iris_model] Entraînement Iris terminé avec ERREUR.")
    else:
        print("[train_iris_model] Entraînement Iris terminé avec SUCCÈS.")

    return exp_name
