----Local Hosts------
Grafana : http://localhost:3000/ 
Prometheus : http://localhost:9090/

----Prometheus Configuration------
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

  - job_name: 'model_training'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:8001']

  - job_name: 'flask_app_model_serve'
    static_configs:
      - targets: ['localhost:8003']

---Prometheus server -------
	
	cd C:\Users\mrosk\OneDrive\Desktop\prometheus
	prometheus.exe --config.file=prometheus.yml

----Grafana server-----
	
	cd C:\Program Files\GrafanaLabs\grafana\bin
	grafana-server.exe

----mlflow server------
	
	cd C:\Users\mrosk\OneDrive\Desktop\Fraud-Detection-Audit\monitoring
	python mlflow_setup.py
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001

set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\mrosk\OneDrive\Desktop\Google-Cloud\ServiceKey_GoogleCloud.json

--------Data Ingestion----------
	
	cd C:\Users\mrosk\OneDrive\Desktop\Fraud-Detection-Audit\data_ingestion
	python app.py

	cd C:\Users\mrosk\OneDrive\Desktop\Fraud-Detection-Audit
	curl -F "file=@dataset/audit_data.csv" http://localhost:5000/upload

---------Preprocessing--------
	
	cd C:\Users\mrosk\OneDrive\Desktop\Fraud-Detection-Audit\data_preprocessing
	python cleaning.py

---Training------
	
	cd C:\Users\mrosk\OneDrive\Desktop\Fraud-Detection-Audit\model_training
	python train.py

------Model Serving----------

	cd C:\Users\mrosk\OneDrive\Desktop\Fraud-Detection-Audit\model_serving
	python serve_model.py

	cd C:/Users/mrosk/OneDrive/Desktop/Fraud-Detection-Audit/model_training
	curl -X POST http://localhost:5002/predict_from_csv -F "file=@cleaned_audit_data.csv"



----Dashboard-----
cd C:\Users\mrosk\OneDrive\Desktop\Fraud-Detection-Audit\grafana_dashboards
python create_dashboards.py

------Testing-------
	
	cd C:\Users\mrosk\OneDrive\Desktop\Fraud-Detection-Audit\data_ingestion\tests
	pytest test_data_ingestion.py

	cd C:\Users\mrosk\OneDrive\Desktop\Fraud-Detection-Audit\data_preprocessing\tests
	pytest test_cleaning.py

	cd C:\Users\mrosk\OneDrive\Desktop\Fraud-Detection-Audit\model_training\tests
	pytest test_model_training.py