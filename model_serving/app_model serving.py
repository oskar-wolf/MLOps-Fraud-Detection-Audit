import os 
import joblib
from flask import Flask, request, jsonify
from prometheus_client_client import Counter, Summary, start_http_server
app = Flask(__name__)

model