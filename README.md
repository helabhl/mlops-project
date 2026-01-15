#  MLOps Project - Iris Classification

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)
[![DVC](https://img.shields.io/badge/Data_Version_Control-Enabled-9cf.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![ZenML](https://img.shields.io/badge/ZenML-Pipeline-FF00FF.svg)](https://zenml.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Deployment-009688.svg)](https://fastapi.tiangolo.com/)

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Project Structure](#-project-structure)
- [Key Components](#-key-components)
    - [1. Data Versioning (DVC)](#1-data-versioning-dvc)
    - [2. Tracking & Experiments (MLflow & Optuna)](#2-tracking--experiments-mlflow--optuna)
    - [3. Pipelines (ZenML)](#3-pipelines-zenml)
    - [4. Deployment (Docker & FastAPI)](#4-deployment-docker--fastapi)
- [CI/CD Pipeline](#-cicd-pipeline)

---

## 📖 Project Overview
This project implements a complete **End-to-End MLOps Pipeline** for the classic Iris classification problem. It is designed to demonstrate best practices in software engineering for Machine Learning, including containerization, reproducibility, experiment tracking, and automated deployment.

---



## 🏗 Architecture
The pipeline integrates several modern tools:
*   **Git**: Code versioning.
*   **Docker & Docker Compose**: Containerization of services (MLflow, ZenML, API).
*   **DVC**: Data versioning and management.
*   **MLflow**: Experiment tracking and model registry.
*   **Optuna**: Hyperparameter optimization.
*   **ZenML**: Orchestration of ML pipelines (Data -> Train -> Eval).
*   **FastAPI**: Serving the model as a REST API.
*   **GitHub Actions**: CI/CD for automated deployment to AWS EC2.

---

## ⚙ Prerequisites
Ensure you have the following installed locally:
*   [Docker Desktop](https://www.docker.com/products/docker-desktop)
*   [Git](https://git-scm.com/)
*   [Python 3.11+](https://www.python.org/)

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/helabhl/mlops-project.git
cd mlops-project
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start Infrastructure (MLflow & ZenML)
We use Docker Compose to spin up the MLflow tracking server and ZenML database.
```bash
docker-compose up -d --build
```
> **Note**: 
> *   **MLflow UI**: Accessible at [http://localhost:5000](http://localhost:5000)
> *   **ZenML Server**: Accessible at [http://localhost:8080](http://localhost:8080)

---

## 📂 Project Structure
```text
mlops-project/
├── .github/workflows/   # CI/CD (GitHub Actions)
├── data/                # Dataset (managed by DVC)
├── iris_api/            # FastAPI Application code
│   └── app.py
├── src/                 # Source code for training & pipelines
│   ├── zenml_pipelines/ # ZenML pipeline definitions
│   ├── zenml_steps/     # ZenML steps (Data, Train, Eval)
│   └── optuna_iris.py   # Hyperparameter tuning script
├── .dvc/                # DVC configuration
├── docker-compose.yml   # Services orchestration
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🔑 Key Components

### 1. Data Versioning (DVC)
We use DVC to track the datasets without bloating the Git repository. The data is stored in an S3 bucket.

**Configure DVC Remote (First time setup):**
```bash
dvc remote add -d s3-storage s3://your-dvc-bucket/data
dvc remote modify s3-storage profile default
```

**Pull the latest data:**
```bash
dvc pull
```
**Add new data:**
```bash
dvc add data/iris.csv
git add data/iris.csv.dvc .gitignore
git commit -m "Update dataset"
dvc push
```

### 2. Tracking & Experiments (MLflow & Optuna)
Optimization logic is handled by **Optuna**, which logs every trial to **MLflow**.
Artifacts are stored in S3 at `s3://mlops12012026/mlflow-artifacts`.

**Run Hyperparameter Optimization:**
```bash
python src/optuna_iris.py
```
Check the results in the MLflow UI at `http://localhost:5000`.

### 3. Pipelines (ZenML)
ZenML orchestrates the steps from data loading to evaluation. We use an **S3-backed Stack** for robust artifact storage.

**Create and Register the Stack:**
```bash
# 1. Initialize ZenML
zenml init
zenml connect --url http://localhost:8080

# 2. Register Components
# Artifact Store (S3 Bucket: mlops12012026)
zenml artifact-store register s3_artifact_store --flavor=s3 --path=s3://mlops12012026/zenml-artifacts

# Experiment Tracker (MLflow)
zenml experiment-tracker register mlflow_tracker --flavor=mlflow --tracking_uri=http://localhost:5000 --tracking_username=user --tracking_password=password

# 3. Register and Activate the Stack
zenml stack register s3_stack \
    -a s3_artifact_store \
    -o default \
    -e mlflow_tracker

zenml stack set s3_stack
```

**Run the Training Pipeline:**
```bash
python src/zenml_pipelines/run_iris_pipeline_baseline.py
```

### 4. Deployment (Docker & FastAPI)
The model is served using FastAPI, containerized with Docker.

**Build and Run Locally:**
```bash
docker build -t iris-api .
docker run -p 5001:5001 iris-api
```

**Test the API:**
```bash
curl -X POST "http://localhost:5001/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

---

## 🔄 CI/CD Pipeline
The project uses **GitHub Actions** defined in `.github/workflows/deploy.yaml`.

**Workflow Steps:**
1.  **Trigger**: Push to `main` branch.
2.  **Build**: Docker image is built.
3.  **Push**: Image is pushed to **GitHub Container Registry (GHCR)**.
4.  **Deploy**: Connects to **AWS EC2** via SSH, pulls the new image, and restarts the container.



---
*Created by Hela BOUHLEL*
