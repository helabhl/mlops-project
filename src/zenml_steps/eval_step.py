from typing import Optional
from zenml import step


@step
def summarize_iris_experiment(exp_name: Optional[str] = None) -> None:
    """Résumé simple du run Iris.

    L'analyse détaillée se fait dans MLflow UI.
    """
    print("[summarize_iris_experiment] Résumé du run Iris.")
    if exp_name:
        print(f"  - Nom d'expérience MLflow : {exp_name}")
    print("  - Ouvrez l'UI MLflow (http://localhost:5000) pour comparer les runs.")
