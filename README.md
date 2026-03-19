# Career Switch Prediction: End-to-End MLOps Pipeline

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20S3%20%7C%20ECR-orange.svg)](https://aws.amazon.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-brightgreen.svg)](https://www.mongodb.com/atlas)

##  Project Overview
This project is an industry-standard MLOps application designed to predict whether an employee is likely to look for a career switch. It features a fully modularized codebase, automated data ingestion from **MongoDB Atlas**, robust preprocessing, and a production-ready **CI/CD pipeline** that deploys a **Dockerized FastAPI** application to **AWS EC2**.

### Key Technical Highlights:
- **Modular Design:** Built with separate components for Ingestion, Validation, Transformation, Training, and Evaluation.
- **Model Registry:** Integrated with **AWS S3**.
- **Automation:** End-to-end CI/CD using **GitHub Actions** and self-hosted runners on EC2.

---

## System Architecture
1. **Data Ingestion:** Fetches raw HR analytics data from MongoDB Atlas.
2. **Data Validation:** Validates data drift and schema integrity using `schema.yaml`.
3. **Data Transformation:** Custom cleaning (City/Experience/Company Size mapping), Outlier capping (IQR), and scaling.
4. **Model Training:** The model training is done through decision tree and it was selected after comparing performance of multiple models.
5. **Model Evaluation:** Compares the new model's F1-Score against the production model in S3.
6. **Model Pusher:** Deploys the best-performing model to the S3 model registry.
7. **Prediction Pipeline:** A FastAPI service that pulls the latest model from S3 to serve real-time predictions.

---

##  Project Structure
```text
.
├── .github/workflows       # CI/CD pipeline (aws.yaml)
├── config                  # schema.yaml & model_config.yaml
├── src                     # Main Source Code
│   ├── components          # Ingestion, Validation, Transformation, Trainer, Evaluation
│   ├── configuration       # AWS and MongoDB connections
│   ├── cloud storage       # AWS storage
│   ├── constants           # Project-wide constants
│   ├── data_access         # MongoDB data fetching logic
│   ├── entity              # Config and Artifact entities
│   ├── pipeline            # Training and Prediction logic
│   ├── utils               # Utility functions (save/load objects)
│   └── logger/exception    # Custom logging and error handling
├── static/css              # Frontend styling
├── templates               # Jinja2 HTML templates
├── app.py                  # FastAPI application entry point
├── Dockerfile              # Docker image configuration
├── requirements.txt        # Python dependencies
└── setup.py                # Package installation

```



##  Tech Stack
- **Backend:** FastAPI
- **Database:** MongoDB Atlas
- **Cloud:** AWS (EC2, ECR, S3, IAM)
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Machine Learning:** Scikit-Learn, Pandas, Numpy, Imbalanced-Learn (SMOTEENN)

---

##  Environment Variables
To run this project, you must configure the following secrets in your environment or GitHub Actions Secrets:

| Key | Description |
| :--- | :--- |
| `MONGODB_URL` | MongoDB Atlas connection string |
| `AWS_ACCESS_KEY_ID` | AWS IAM User access key |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM User secret key |
| `AWS_DEFAULT_REGION` | AWS Region (e.g., us-east-1) |
| `ECR_REPO` | Name of your AWS ECR repository |

---

##  Setup & Execution

### 1. Local Installation
```
# Clone the repo
git clone https://github.com/nawrin1/Career-Switch-Prediction-MLOPS
cd Career-Switch-Prediction-MLOPS

# Create Virtual Env
conda create -n career python=3.10 -y
conda activate career

# Install dependencies
pip install -r requirements.txt

```
