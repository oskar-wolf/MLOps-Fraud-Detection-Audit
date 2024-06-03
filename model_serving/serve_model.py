import os
import joblib
import pandas as pd
import json
from flask import Flask, request, jsonify
from prometheus_client import Counter, Summary, start_http_server

app = Flask(__name__)

# Load trained model
model_path = os.path.join('..', 'model_training', 'saved_models', 'SVM.joblib')
model = joblib.load(model_path)

# Load preprocessing pipeline
pipeline = joblib.load(os.path.join('..', 'model_training', 'preprocessing_objects', 'pipeline.joblib'))

# Load selected features
selected_features_path = os.path.join('..', 'model_training', 'preprocessing_objects', 'selected_features.json')
with open(selected_features_path, 'r') as f:
    selected_features = json.load(f)

# Prometheus Metrics
REQUEST_COUNT = Counter('request_count', 'Total API Requests')
REQUEST_LATENCY = Summary('request_latency_seconds', 'API Request Latency')

@app.route('/predict', methods=['POST'])
def predict():
    REQUEST_COUNT.inc(1)
    with REQUEST_LATENCY.time():
        data = request.json
        features = pd.DataFrame([data['features']], columns=selected_features)
        features_preprocessed = pipeline.transform(features)
        prediction = model.predict(features_preprocessed)
        return jsonify({'prediction': prediction[0]})

@app.route('/predict_from_csv', methods=['POST'])
def predict_from_csv():
    REQUEST_COUNT.inc(1)
    with REQUEST_LATENCY.time():
        file = request.files['file']
        df = pd.read_csv(file)
        
        # Ensure only the required features are selected
        if set(selected_features).issubset(df.columns):
            df = df[selected_features]
        else:
            return jsonify({'error': 'CSV file does not contain the required features'}), 400

        print(f"Shape of selected features dataframe during prediction: {df.shape}")

        df_preprocessed = pipeline.transform(df)
        predictions = model.predict(df_preprocessed)

        # Print value counts of predictions
        predictions_series = pd.Series(predictions)
        print("Prediction value counts:\n", predictions_series.value_counts())
        
        return jsonify({'predictions': predictions.tolist()})

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f'An error occurred: {e}')
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    start_http_server(8003)
    app.run(host='0.0.0.0', port=5002)
