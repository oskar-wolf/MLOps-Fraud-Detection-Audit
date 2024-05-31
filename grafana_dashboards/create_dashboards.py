from grafanalib.core import (
    Dashboard, GridPos, Target, BarGauge, RowPanel, Stat, Graph, BarGauge
)
from grafanalib._gen import DashboardEncoder
import json
import requests

# Function to generate color for each model based on its index
def get_color(index):
    colors = [
        "green", "blue", "red", "yellow", "purple", "orange", "pink", "cyan",
        "brown", "lime", "magenta", "grey", "maroon", "navy", "olive", "teal", "violet", "indigo"
    ]
    return colors[index % len(colors)]

colors = [
    "#7EB26D", "#EAB839", "#6ED0E0", "#EF843C", "#E24D42", 
    "#1F78C1", "#BA43A9", "#705DA0", "#508642", "#CCA300",
    "#447EBC", "#C15C17", "#890F02", "#0A437C", "#6D1F62",
    "#584477", "#B7DBAB", "#F4D598"
]

def create_dashboard():
    dashboard = Dashboard(
        title="Fraud Detection Dashboard",
        panels=[
            # Data Ingestion Panels
            Stat(
                title="File Uploads Total",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='file_uploads_total',
                        legendFormat="Total Uploads",
                    ),
                ],
                gridPos=GridPos(h=3, w=3, x=0, y=0),
                colorMode="value",
                graphMode="area",
                orientation="horizontal",
            ),
            Stat(
                title="File Upload Bytes Total",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='file_upload_bytes_total',
                        legendFormat="Total Bytes",
                    ),
                ],
                gridPos=GridPos(h=3, w=3, x=3, y=0),
                colorMode="value",
                graphMode="area",
                orientation="horizontal",
            ),
            Stat(
                title="File Upload Errors Total",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='file_upload_errors_total',
                        legendFormat="Upload Errors",
                    ),
                ],
                gridPos=GridPos(h=3, w=3, x=6, y=0),
                colorMode="value",
                graphMode="area",
                orientation="horizontal",
            ),
            # Data Processing Panels
            Stat(
                title="Processed Records Total",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='processed_records_total',
                        legendFormat="Total Processed Records",
                    ),
                ],
                gridPos=GridPos(h=3, w=3, x=9, y=0),
                colorMode="value",
                graphMode="area",
                orientation="horizontal",
            ),
            Stat(
                title="Data Processing Time",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='data_processing_seconds_sum',
                        legendFormat="Processing Time (seconds)",
                    ),
                ],
                gridPos=GridPos(h=3, w=3, x=12, y=0),
                colorMode="value",
                graphMode="area",
                orientation="horizontal",
            ),
            BarGauge(
                title="Missing Values",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='missing_values',
                        legendFormat="Missing Values",
                    ),
                ],
                gridPos=GridPos(h=3, w=3, x=15, y=0),
            ),
            BarGauge(
                title="Duplicate Records",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='duplicate_records',
                        legendFormat="Duplicate Records",
                    ),
                ],
                gridPos=GridPos(h=3, w=3, x=18, y=0),
            ),
            # Model Metrics Panels
            BarGauge(
                title="Model Accuracy",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='model_accuracy',
                        legendFormat="{{model_name}}",
                    ),
                ],
                gridPos=GridPos(h=8, w=20, x=0, y=3),
                orientation="vertical",
                displayMode="gradient",
                thresholds=[
                    {"color": color, "value": None} for color in (get_color(i) for i in range(3))
                ],
                max=1.0,
                min = 0.94
            ),
            BarGauge(
                title="Model Precision",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='model_precision',
                        legendFormat="{{model_name}}",
                    ),
                ],
                gridPos=GridPos(h=8, w=20, x=0, y=6),
                orientation="vertical",
                displayMode="gradient",
                thresholds=[
                    {"color": color, "value": None} for color in (get_color(i) for i in range(8))
                ],
                max=1.0,
                min = 0.94
            ),
            BarGauge(
                title="Model Recall",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='model_recall',
                        legendFormat="{{model_name}}",
                    ),
                ],
                gridPos=GridPos(h=8, w=20, x=0, y=9),
                orientation="vertical",
                displayMode="gradient",
                thresholds=[
                    {"color": color, "value": None} for color in (get_color(i) for i in range(4))
                ],
                max=1.0,
                min = 0.94
            ),
            BarGauge(
                title="Model F1 Score",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='model_f1',
                        legendFormat="{{model_name}}",
                    ),
                ],
                gridPos=GridPos(h=8, w=20, x=0, y=12),
                orientation="vertical",
                displayMode="gradient",
                thresholds=[
                    {"color": color, "value": None} for color in (get_color(i) for i in range(18))
                ],
                decimals=5,
                max=1.0,
                min = 0.94,
            ),
        ],
    ).auto_panel_ids()

    return dashboard

def save_dashboard(dashboard, filename="dashboard.json"):
    with open(filename, 'w') as f:
        json.dump(dashboard.to_json_data(), f, cls=DashboardEncoder, indent=2)

def upload_dashboard(dashboard_json, grafana_url, headers):
    response = requests.post(
        f"{grafana_url}/api/dashboards/db",
        headers=headers,
        data=json.dumps({
            "dashboard": dashboard_json,
            "overwrite": True,
        }, cls=DashboardEncoder)
    )
    return response

if __name__ == '__main__':
    dashboard = create_dashboard()
    dashboard_json = dashboard.to_json_data()

    save_dashboard(dashboard)

    # Define Grafana API details
    GRAFANA_URL = 'http://admin:admin@localhost:3000'  # Using basic auth for simplicity
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Upload the dashboard to Grafana
    response = upload_dashboard(dashboard_json, GRAFANA_URL, headers)

    # Check the response
    if response.status_code == 200:
        print("Dashboard uploaded successfully")
    else:
        print(f"Failed to upload dashboard: {response.status_code}")
        print(response.json())
