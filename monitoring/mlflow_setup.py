import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    # Set the tracking URI to a local file-based store
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Set the experiment name
    mlflow.set_experiment("Fraud Detection Experiment")

    # Start the MLflow server
    mlflow.server()