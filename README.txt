Grafana : http://localhost:3000/ 
Prometheus : http://localhost:9090/

YML File:
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'flask_app'
    static_configs:
      - targets: ['localhost:5000']

cd C:\Users\mrosk\OneDrive\Desktop\prometheus
prometheus.exe --config.file=prometheus.yml

cd C:\Program Files\GrafanaLabs\grafana\bin
grafana-server.exe

set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\mrosk\OneDrive\Desktop\Google-Cloud\ServiceKey_GoogleCloud.json
set GRAFANA_API_KEY=your_grafana_api_key

