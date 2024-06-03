'''import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import pandas as pd 
import joblib
import time
from google.cloud import storage
from models import models
import mlflow
import mlflow.sklearn
import numpy as np
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from prometheus_client import Summary, Counter, Gauge, start_http_server

from models import models

google_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if google_credentials_path and os.path.isfile(google_credentials_path):
    storage_client = storage.Client.from_service_account_json(google_credentials_path)
else:
    raise FileNotFoundError("Google Cloud credentials file not found or environment variable not set correctly.")

TRAINING_TIME = Summary('model_training_seconds', 'Time spent training the model', ['model_name'])
MODEL_ACCURACY = Gauge('model_accuracy', 'Accuracy of the model', ['model_name'])
MODEL_PRECISION = Gauge('model_precision', 'Precision of the model', ['model_name'])
MODEL_RECALL = Gauge('model_recall', 'Recall of the model', ['model_name'])
MODEL_F1 = Gauge('model_f1', 'F1 score of the model', ['model_name'])
TRAINED_MODELS = Counter('trained_models_total', 'Total number of models trained')

def download_data_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}.")

def load_data(local_file_path):
    return pd.read_csv(local_file_path)

def save_model(model, model_name):
    model_dir = 'saved_models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{model_name}.joblib')
    joblib.dump(model, model_path)
    print(f"Model {model_name} saved to {model_path}.")
    print('\n')

def feature_selection(df, target_column='Risk', n_features=10):    
    X = df.drop(columns=[target_column])
    y = df[target_column]
  
    print("DataFrame before feature selection:")
    print(X.head()) 

    print("DataFrame before feature selection:")
    print(y.head()) 

    print("Data types of the DataFrame:")
    print(df.dtypes)

    # Select top n_features based on ANOVA F-value between feature and target
    kbest_selector = SelectKBest(score_func=f_classif, k=n_features)
    kbest_selector.fit(X, y)
    
    selected_features = X.columns[kbest_selector.get_support()]

    with open("selected_features.json", "w") as f:
        json.dump(selected_features.tolist(), f)
    
    print(f"Selected features: {selected_features}")

    return df[selected_features], y

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model, params):
    grid_search = GridSearchCV(model,params, cv = 5, scoring = 'accuracy')
    start_time= time.time()
    grid_search.fit(X_train,y_train)
    training_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred,average='weighted')
    recall = recall_score(y_test,y_pred,average='weighted')
    f1 = f1_score(y_test,y_pred,average='weighted')

    MODEL_ACCURACY.labels(model_name=model_name).set(accuracy)
    MODEL_PRECISION.labels(model_name=model_name).set(precision)
    MODEL_RECALL.labels(model_name=model_name).set(recall)
    MODEL_F1.labels(model_name=model_name).set(f1)
    TRAINING_TIME.labels(model_name=model_name).observe(training_time)
    TRAINED_MODELS.inc()

    print('\n')
    print(f"Model: {model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print('\n')

    # Save the predictions for the test set
    prediction_dir = os.path.join('training_predictions')
    os.makedirs(prediction_dir, exist_ok=True)
    predictions_df = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
    predictions_df.to_csv(os.path.join(prediction_dir, f'test_predictions_{model_name}.csv'), index=False)

    time.sleep(2)
    return best_model, accuracy, precision, recall, f1

def main():
    bucket_name = 'mlops-fraud-detect-audit-app'
    source_blob_name = 'cleaned_audit_data.csv'
    local_file_path = 'cleaned_audit_data.csv'

    download_data_from_gcs(bucket_name,source_blob_name,local_file_path)

    df = load_data(local_file_path)
    df = df.drop(columns=['LOCATION_ID'])

    print("Target variable distribution:")
    print(df['Risk'].value_counts())

    X, selected_features = feature_selection(df, target_column='Risk', n_features=10)
    print("Selected features: ", selected_features)
    y = df['Risk']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    results = []

    mlflow.set_tracking_uri("http://localhost:5001")
    for model_name, (model,params) in models.items():
        print(f'Training {model_name}...')
        with mlflow.start_run(run_name=model_name):
            best_model, accuracy, precision, recall, f1 = train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model, params)
            save_model(best_model,model_name)
            mlflow.log_metrics({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
            mlflow.sklearn.log_model(best_model,model_name)
        results.append((model_name, accuracy, precision, recall, f1))

if __name__ == '__main__':
    start_http_server(8001)
    main()'''

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import pandas as pd 
import joblib
import time
from google.cloud import storage
from models import models
import mlflow
import mlflow.sklearn
import numpy as np
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from prometheus_client import Summary, Counter, Gauge, start_http_server

from models import models

google_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if google_credentials_path and os.path.isfile(google_credentials_path):
    storage_client = storage.Client.from_service_account_json(google_credentials_path)
else:
    raise FileNotFoundError("Google Cloud credentials file not found or environment variable not set correctly.")

TRAINING_TIME = Summary('model_training_seconds', 'Time spent training the model', ['model_name'])
MODEL_ACCURACY = Gauge('model_accuracy', 'Accuracy of the model', ['model_name'])
MODEL_PRECISION = Gauge('model_precision', 'Precision of the model', ['model_name'])
MODEL_RECALL = Gauge('model_recall', 'Recall of the model', ['model_name'])
MODEL_F1 = Gauge('model_f1', 'F1 score of the model', ['model_name'])
TRAINED_MODELS = Counter('trained_models_total', 'Total number of models trained')

def download_data_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}.")

def load_data(local_file_path):
    return pd.read_csv(local_file_path)

def save_model(model, model_name):
    model_dir = os.path.join('saved_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{model_name}.joblib')
    joblib.dump(model, model_path)
    print(f"Model {model_name} saved to {model_path}.")

def save_pipeline(pipeline):
    pipeline_dir = 'preprocessing_objects'
    os.makedirs(pipeline_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(pipeline_dir, 'pipeline.joblib'))
    print("Pipeline saved.")

def save_selected_features(features):
    pipeline_dir = 'preprocessing_objects'
    os.makedirs(pipeline_dir, exist_ok=True)
    with open(os.path.join(pipeline_dir, 'selected_features.json'), 'w') as f:
        json.dump(features.tolist(), f)
    print("Selected features saved.")

def feature_selection(df, target_column='Risk', n_features=10):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print("DataFrame before feature selection:")
    print(X.head()) 

    print("DataFrame before feature selection:")
    print(y.head()) 

    print("Data types of the DataFrame:")
    print(df.dtypes)

    kbest_selector = SelectKBest(score_func=f_classif, k=n_features)
    X_selected = kbest_selector.fit_transform(X, y)
    
    selected_features = X.columns[kbest_selector.get_support()]
    print("Selected features:", selected_features)
    
    return X_selected, y, kbest_selector, selected_features

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model, params):
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    MODEL_ACCURACY.labels(model_name=model_name).set(accuracy)
    MODEL_PRECISION.labels(model_name=model_name).set(precision)
    MODEL_RECALL.labels(model_name=model_name).set(recall)
    MODEL_F1.labels(model_name=model_name).set(f1)
    TRAINING_TIME.labels(model_name=model_name).observe(training_time)
    TRAINED_MODELS.inc()

    print('\n')
    print(f"Model: {model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print('\n')

    time.sleep(2)
    return best_model, accuracy, precision, recall, f1

def main():
    bucket_name = 'mlops-fraud-detect-audit-app'
    source_blob_name = 'cleaned_audit_data.csv'
    local_file_path = 'cleaned_audit_data.csv'

    download_data_from_gcs(bucket_name, source_blob_name, local_file_path)

    df = load_data(local_file_path)
    df = df.drop(columns=['LOCATION_ID'])

    print("Target variable distribution:")
    print(df['Risk'].value_counts())

    X, y, selector, selected_features = feature_selection(df, target_column='Risk', n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()

    # Create a pipeline that applies feature selection and scaling
    pipeline = Pipeline([
        ('selector', selector),
        ('scaler', scaler)
    ])

    X_train = pipeline.fit_transform(X_train, y_train)
    X_test = pipeline.transform(X_test)

    save_pipeline(pipeline)
    save_selected_features(selected_features)

    mlflow.set_tracking_uri("http://localhost:5001")
    for model_name, (model, params) in models.items():
        print(f'Training {model_name}...')
        with mlflow.start_run(run_name=model_name):
            best_model, accuracy, precision, recall, f1 = train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model, params)
            save_model(best_model, model_name)
            mlflow.log_metrics({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
            mlflow.sklearn.log_model(best_model, model_name)

if __name__ == '__main__':
    start_http_server(8001)
    main()
