from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# ---------------------------
# Charger le modèle
# ---------------------------
MODEL_PATH = "./runs/RF_50_5_s123_model.pkl"  
model = joblib.load(MODEL_PATH)

# ---------------------------
# Définition FastAPI
# ---------------------------
app = FastAPI(title="Iris Classifier API")

# Modèle pour les requêtes
class IrisInput(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

# Endpoint test
@app.get("/")
def read_root():
    return {"message": "Iris API is alive!"}

# Endpoint prédiction
@app.post("/predict")
def predict(input_data: IrisInput):
    data = {
        "sepal length (cm)": input_data.sepal_length,
        "sepal width (cm)": input_data.sepal_width,
        "petal length (cm)": input_data.petal_length,
        "petal width (cm)": input_data.petal_width,
    }

    df = pd.DataFrame([data])

    # Feature engineering identique à l'entraînement
    df["petal_area"] = df["petal length (cm)"] * df["petal width (cm)"]

    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}
