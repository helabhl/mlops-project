import subprocess
from zenml import step


@step
def prepare_iris_dataset() -> None:
    """Prépare / vérifie le dataset Iris via DVC.

    - si un remote DVC est configuré : dvc pull
    - sinon : message d'information
    """
    print("[prepare_iris_dataset] Vérification du dataset Iris avec DVC...")
    try:
        subprocess.run(["dvc", "pull"], check=False)
    except FileNotFoundError:
        print("[prepare_iris_dataset] DVC non trouvé dans l'environnement.")
