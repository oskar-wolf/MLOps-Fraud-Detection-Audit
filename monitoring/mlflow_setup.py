import mlflow
import mlflow.sklearn
import os

# Set the tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

# Create an experiment if it doesn't exist
experiment_name = "Fraud Detection Experiment"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)

# Set the experiment
mlflow.set_experiment(experiment_name)

print("MLflow setup complete. You can now start the MLflow server using the command:")
print("mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001")
