import os
import pytest
import pandas as pd
from data_preprocessing.cleaning import clean_data, load_data, save_data, process_data

@pytest.fixture
def test_dataset():
    data = {
        'col1': [1, 2, 2, 4],
        'col2': [None, 2, 3, None]
    }
    df = pd.DataFrame(data)
    return df

def test_clean_data(test_dataset):
    df_cleaned = clean_data(test_dataset)
    assert df_cleaned.duplicated().sum() == 0
    assert df_cleaned.isnull().sum().sum() == 0

def test_load_save_data(tmp_path, test_dataset):
    file_path = tmp_path / "test_data.csv"
    save_data(test_dataset, file_path)
    df_loaded = load_data(file_path)
    pd.testing.assert_frame_equal(test_dataset, df_loaded)

def test_process_data(monkeypatch, tmp_path):
    temp_file = tmp_path / "temp.csv"

    # Mock GCS methods
    def mock_download_data_from_gcs(bucket_name, source_blob_name, destination_file_name):
        test_dataset().to_csv(destination_file_name, index=False)

    def mock_save_data_to_gcs(bucket_name, source_file_name, destination_blob_name):
        pass

    monkeypatch.setattr('data_preprocessing.cleaning.download_data_from_gcs', mock_download_data_from_gcs)
    monkeypatch.setattr('data_preprocessing.cleaning.save_data_to_gcs', mock_save_data_to_gcs)

    process_data('mlops-fraud-detect-audit-app', 'source.csv', 'destination.csv', temp_file)

    df_cleaned = load_data(temp_file)
    assert df_cleaned.duplicated().sum() == 0
    assert df_cleaned.isnull().sum().sum() == 0