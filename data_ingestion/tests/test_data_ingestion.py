import os
import tempfile
import pytest

# Ensure the PYTHONPATH is set to the project root so that imports work correctly.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_ingestion.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_upload_file(client):
    data = {
        'file': (tempfile.NamedTemporaryFile(delete=False, suffix='.csv'), 'test.csv')
    }
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert response.json == {'message': 'File uploaded successfully'}

def test_upload_file_error(client):
    response = client.post('/upload', data={}, content_type='multipart/form-data')
    assert response.status_code == 500
    assert 'message' in response.json
