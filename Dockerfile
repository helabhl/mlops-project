# 1. On part d'une version légère de Python
FROM python:3.11-slim

# 2. On crée le dossier de travail dans le conteneur
WORKDIR /app

# 3. On copie le fichier des dépendances
COPY requirements.txt .

# 4. On installe les librairies (Flask, pandas, mlflow, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# 5. On copie tout votre code dans le conteneur
COPY . .

# 6. On dit au conteneur quel port ouvrir (Flask utilise 5000 par défaut)
EXPOSE 5000

# 7. La commande pour lancer l'API quand le conteneur démarre
CMD ["python", "app.py"]