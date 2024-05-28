import pandas as pd 
from google.cloud import storage
import os
import time
from prometheus_client import start_http_server, Summary, Counter, Gauge

REQUEST_TIME = Summary('data_processing_seconds', 'Time spent processing data')
PROCESSED_RECORDS = Counter('processed_records_total', 'Total number of records processed')
MISSING_VALUES = Gauge('missing_values', 'Total number of missing values in the dataset')
DUPLICATES = Gauge('duplicate_records', 'Total number of duplicate records in dataset')

# Function to download data from GCS
def download_data_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}.")

# Function to save data to GCS
def save_data_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to bucket {bucket_name} as {destination_blob_name}.")

def load_data(local_file_path):
    return pd.read_csv(local_file_path)

def save_data(df,local_file_path):
    df.to_csv(local_file_path,index=False)

def clean_data(df):
    df = df.drop_duplicates()
    df = df.fillna(0)
    return df

@REQUEST_TIME.time()
def process_data(bucket_name,source_file_name,destination_file_name, temp_local_file):
    download_data_from_gcs(bucket_name,source_file_name,temp_local_file)

    df = load_data(temp_local_file)

    initial_record_count = len(df)
    initial_missing_values = df.isnull().sum().sum()

    df_cleaned = clean_data(df)

    cleaned_record_count = len(df_cleaned)
    cleaned_missing_values = df_cleaned.isnull().sum().sum()
    duplicate_count = df_cleaned.duplicated().sum()

    PROCESSED_RECORDS.inc(cleaned_record_count)
    MISSING_VALUES.set(cleaned_missing_values)
    DUPLICATES.set(duplicate_count)

    save_data(df_cleaned,temp_local_file)

    save_data_to_gcs(bucket_name,temp_local_file,destination_file_name)

if __name__ == '__main__':
    start_http_server(8000)
    process_data('mlops-fraud-detect-audit-app', 'audit_data.csv', 'cleaned_audit_data.csv', 'temp.csv')
