# Étape 1 : base image Python
FROM python:3.11-slim

# Variables environnement
ENV PYTHONUNBUFFERED=1

# Créer le dossier de l'app
WORKDIR /app

# Copier requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code
COPY . .

# Exposer le port
EXPOSE 8000

# Commande pour lancer FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
