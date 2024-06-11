#!/bin/bash

while [ ! -f /app/saved_models/XGBoost.joblib ] || [ ! -f /app/preprocessing_objects/pipeline.joblib ] || [ ! -f /app/preprocessing_objects/selected_features.json ]; do
    echo "Waiting for model and preprocessing files..."
    sleep 10
done

echo "Model and preprocessing files found. Starting model_serving..."
python /app/model_serving/serve_model.py
