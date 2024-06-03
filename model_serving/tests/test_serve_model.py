import os
import pytest
from flask import Flask
from model_serving.serve_model import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict(client):
    response = client.post('/predict', json={'features': [3.89, 4.18, 0.6, 2.508, 0.2, 0.0, 0.4, 0.5, 0.6, 0.7]})
    assert response.status_code == 200
    assert 'prediction' in response.json

def test_error_handling(client):
    response = client.post('/predict', json={'invalid_key': 'invalid_value'})
    assert response.status_code == 500
    assert 'error' in response.json
