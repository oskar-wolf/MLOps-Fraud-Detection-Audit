import os
from google.cloud import storage
from prometheus_client import Counter, generate_latest
from flask import Flask, request, jsonify

# Initialize App
app = Flask(__name__)


#set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\mrosk\OneDrive\Desktop\Google-Cloud\ServiceKey_GoogleCloud.json
#set GRAFANA_API_KEY=your_grafana_api_key
# Set up Google Cloud Storage client using environment variable
google_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if google_credentials_path and os.path.isfile(google_credentials_path):
    storage_client = storage.Client.from_service_account_json(google_credentials_path)
else:
    raise FileNotFoundError("Google Cloud credentials file not found or environment variable not set correctly.")

# Prometheus Metrics
file_upload_counter = Counter('file_uploads_total', 'Total number of file uploads')
file_upload_size = Counter('file_upload_bytes', 'Total size of uploaded files in bytes')
file_upload_errors = Counter('file_upload_errors_total', 'Total number of file upload errors')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        file_size = len(file.read())
        file.seek(0)
        bucket_name = 'mlops-fraud-detect-audit-app'
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file.filename)
        blob.upload_from_file(file)
        file_upload_counter.inc()
        file_upload_size.inc(file_size)
        return jsonify({'message': 'File uploaded successfully'}), 200
    except Exception as e:
        file_upload_errors.inc()
        return jsonify({'message': str(e)}), 500

@app.route('/metrics')
def metrics():
    return generate_latest()

if __name__ == '__main__':
    app.run(debug=True)
