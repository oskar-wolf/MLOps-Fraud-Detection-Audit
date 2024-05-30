from grafanalib.core import (
    Dashboard, GridPos, Target, BarGauge, RowPanel, Stat, Graph
)
from grafanalib._gen import DashboardEncoder
import json
import requests

# Define the combined dashboard with panels for both data ingestion and data processing
'''dashboard = Dashboard(
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
            gridPos=GridPos(h=5, w=5, x=0, y=0),
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
            gridPos=GridPos(h=5, w=5, x=5, y=0),
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
            gridPos=GridPos(h=5, w=5, x=10, y=0),
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
            gridPos=GridPos(h=5, w=5, x=0, y=5),
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
            gridPos=GridPos(h=5, w=5, x=5, y=5),
            colorMode="value",
            graphMode="area",
            orientation="horizontal",
        ),
        GaugePanel(
            title="Missing Values",
            dataSource='prometheus',
            targets=[
                Target(
                    expr='missing_values',
                    legendFormat="Missing Values",
                ),
            ],
            gridPos=GridPos(h=5, w=5, x=10, y=5),
        ),
        GaugePanel(
            title="Duplicate Records",
            dataSource='prometheus',
            targets=[
                Target(
                    expr='duplicate_records',
                    legendFormat="Duplicate Records",
                ),
            ],
            gridPos=GridPos(h=5, w=5, x=15, y=5),
        ),
    ],
).auto_panel_ids()'''

'''# Convert the dashboard to JSON format
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
    print(response.json())'''


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
                gridPos=GridPos(h=5, w=5, x=0, y=0),
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
                gridPos=GridPos(h=5, w=5, x=5, y=0),
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
                gridPos=GridPos(h=5, w=5, x=10, y=0),
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
                gridPos=GridPos(h=5, w=5, x=0, y=5),
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
                gridPos=GridPos(h=5, w=5, x=5, y=5),
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
                gridPos=GridPos(h=5, w=5, x=10, y=5),
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
                gridPos=GridPos(h=5, w=5, x=15, y=5),
            ),
            # Model Metrics Panels
            Graph(
                title="Model Accuracy",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='model_accuracy',
                        legendFormat="{{model_name}}",
                    ),
                ],
                gridPos=GridPos(h=8, w=12, x=0, y=10),
            ),
            Graph(
                title="Model Precision",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='model_precision',
                        legendFormat="{{model_name}}",
                    ),
                ],
                gridPos=GridPos(h=8, w=12, x=12, y=10),
            ),
            Graph(
                title="Model Recall",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='model_recall',
                        legendFormat="{{model_name}}",
                    ),
                ],
                gridPos=GridPos(h=8, w=12, x=0, y=18),
            ),
            Graph(
                title="Model F1 Score",
                dataSource='prometheus',
                targets=[
                    Target(
                        expr='model_f1',
                        legendFormat="{{model_name}}",
                    ),
                ],
                gridPos=GridPos(h=8, w=12, x=12, y=18),
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
