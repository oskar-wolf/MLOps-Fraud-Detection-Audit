import mlflow
import os

def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("Fraud Detection Experiment")

if __name__ == '__main__':
    setup_mlflow()

