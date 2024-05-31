Grafana : http://localhost:3000/ 
Prometheus : http://localhost:9090/




YML File:
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'flask_app'
    static_configs:
      - targets: ['localhost:5000']

  - job_name: 'data_preprocessing'
    static_configs:
      - targets: ['localhost:8000']


cd C:\Users\mrosk\OneDrive\Desktop\prometheus
prometheus.exe --config.file=prometheus.yml

cd C:\Program Files\GrafanaLabs\grafana\bin
grafana-server.exe

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001


set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\mrosk\OneDrive\Desktop\Google-Cloud\ServiceKey_GoogleCloud.json


Start the Flask app that handles data ingestion:
	cd data_ingestion
	python app.py


curl -F "file=@dataset/audit_data.csv" http://localhost:5000/upload


Execute the cleaning.py script to process the uploaded data:
	cd data_preprocessing
	python cleaning.py

	
cd grafana_dashboards
python create_dashboards.py


