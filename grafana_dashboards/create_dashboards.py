from grafanalib.core import (
    Dashboard, Graph, GridPos, Target, YAxes, YAxis
)
from grafanalib._gen import DashboardEncoder
import json
import requests

# Define the dashboard with multiple panels
dashboard = Dashboard(
    title="Fraud Detection",
    panels=[
        Graph(
            title="File Uploads Total",
            dataSource='prometheus',
            targets=[
                Target(
                    expr='file_uploads_total',
                    legendFormat="Total Uploads",
                ),
            ],
            gridPos=GridPos(h=9, w=12, x=0, y=0),
            yAxes=YAxes(
                left=YAxis(format="short"),
                right=YAxis(format="short"),
            ),
        ),
        Graph(
            title="File Upload Bytes Total",
            dataSource='prometheus',
            targets=[
                Target(
                    expr='file_upload_bytes_total',
                    legendFormat="Total Bytes",
                ),
            ],
            gridPos=GridPos(h=9, w=12, x=0, y=9),
            yAxes=YAxes(
                left=YAxis(format="bytes"),
                right=YAxis(format="short"),
            ),
        ),
        Graph(
            title="File Upload Errors Total",
            dataSource='prometheus',
            targets=[
                Target(
                    expr='file_upload_errors_total',
                    legendFormat="Upload Errors",
                ),
            ],
            gridPos=GridPos(h=9, w=12, x=0, y=18),
            yAxes=YAxes(
                left=YAxis(format="short"),
                right=YAxis(format="short"),
            ),
        ),
    ],
).auto_panel_ids()

# Convert the dashboard to JSON format
dashboard_json = json.dumps(dashboard.to_json_data(), cls=DashboardEncoder, indent=2)

# Save the dashboard JSON to a file (for debugging purposes)
with open('fraud_detection.json', 'w') as f:
    f.write(dashboard_json)

# Define Grafana API details
GRAFANA_URL = 'http://admin:admin@localhost:3000'  # Using basic auth for simplicity
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# Upload the dashboard to Grafana
response = requests.post(
    f"{GRAFANA_URL}/api/dashboards/db",
    headers=headers,
    data=json.dumps({
        "dashboard": dashboard.to_json_data(),
        "overwrite": True,
    }, cls=DashboardEncoder)
)

# Check the response
if response.status_code == 200:
    print("Dashboard uploaded successfully")
else:
    print(f"Failed to upload dashboard: {response.status_code}")
    print(response.json())
