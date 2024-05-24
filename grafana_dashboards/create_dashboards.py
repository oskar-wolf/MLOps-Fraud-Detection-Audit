from grafanalib.core import (
    Dashboard, Graph, Row, GridPos, YAxes, YAxis
)

from grafanalib._gen import DashboardEncoder
import json
import requests

dashboard = Dashboard(
    title="Fraud Detection Metrics",
    rows=[
        Row(panels=[
            Graph(
                title="Number of Files Uploaded",
                dataSource='Prometheus',
                targets=[
                    Target(
                        expr='file_uploads_total',
                        legendFormat="File Uploads",
                    ),
                ],
                gridPos=GridPos(h=9, w=12, x=0, y=0),
                yAxes=YAxes(
                    left=YAxis(format="short"),
                    right=YAxis(format="short"),
                ),
            ),
            Graph(
                title ="Size of Files Uploaded",
                dataSource='Prometheus',
                targets=[
                    Target(
                        expr='file+uploaded_size_bytes',
                        legendFormat="File Size",
                    ),
                ],
                gridpos=GridPos(h=9,w=12,x=12,y=0),
                yAxes=YAxes(
                    left = YAxis(format='bytes'),
                    right= YAxis(format = 'short'),
                ),
            ),
            Graph(
                tile= 'File Upload Errors',
                DataSource = 'Prometheus',
                targets=[
                    Target(
                        expr = 'file_upload_errors_total',
                        legendFormat='Upload Errors',
                    ),
                ],
                gridPos = GridPos(h=9,w=12,x=24,y=0),
                yAxes= YAxes(
                    left = YAxis(format='short'),
                    right = YAxis(format='short'),
                ),
            ),
        ]),
    ],
).auto_panel_ids()

dashboard_json = json.dumps(dashboard.to_json_data(), cls = DashboardEncoder, indent=2)

with open('dashboard.json','w') as f:
    f.write(dashboard_json)

#Define Grafana API Details
GRAFANA_URL = 'http://localhost:3000'
API_KEY = 'glsa_iJxPD21aSAQVbmtP8aZfqFpk2YLHasEU_7c67913c'
headers= {
    "Content-Type" : "application/json",
    "Authorization" : f"Bearer  {API_KEY}"
}

#Upload Dashboard to Grafana
response = request.post(
    f"{GRAFANA_URL}/api/dashboards/db",
    headers= headers,
    data=json.dumps({
        "dashboard" : dashboard.to_json_data(),
        "overwrite" : True,
    })
)

#Check the respone
if response.status_code == 200:
    print("Dashboard uploaded siccessfully")
else:
    print(f"Failed to upload dashboard: {response.status_code}")
    print(respons.json())
