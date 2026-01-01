import os, joblib, json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# ----------------------------
# Charger dataset
# ----------------------------
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

# Sauvegarde dataset local
os.makedirs("data", exist_ok=True)
iris_df = X.copy()
iris_df['target'] = y
iris_df.to_csv("data/iris.csv", index=False)

# ----------------------------
# Split train/test
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Entraînement modèle
# ----------------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# ----------------------------
# Évaluation
# ----------------------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ----------------------------
# Sauvegarde modèle + métadonnées
# ----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/model.joblib")

meta = {
    "dataset": "sklearn.datasets.load_iris",
    "model": "RandomForestClassifier",
    "sklearn_version": joblib.__version__,
    "accuracy": acc,
    "target_names": list(map(str, iris.target_names)),
}

with open("models/model_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Modèle et métadonnées sauvegardés dans models/")
