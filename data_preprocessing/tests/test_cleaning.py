import os
import pytest
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_preprocessing.cleaning import clean_data, load_data, save_data, process_data



@pytest.fixture
def test_dataset():
    data = {
        'Sector_score': [3.89, 3.89, 3.89, 3.89, 3.89],
        'PARA_A': [4.18, 0.00, 0.51, 0.00, 0.00],
        'Score_A': [0.6, 0.2, 0.2, 0.2, 0.2],
        'Risk_A': [2.508, 0.000, 0.102, 0.000, 0.000],
        'PARA_B': [5.18, 1.00, 1.51, 1.00, 1.00],
        'Score_B': [0.7, 0.3, 0.3, 0.3, 0.3],
        'Risk_B': [3.508, 0.100, 0.202, 0.100, 0.100],
        'TOTAL': [9.36, 1.00, 2.02, 1.00, 1.00],
        'numbers': [12, 8, 9, 5, 7],
        'Score_B.1': [1.7, 0.3, 0.3, 0.3, 0.3],
        'Risk_C': [4.508, 0.200, 0.302, 0.200, 0.200],
        'Money_Value': [1.0, None, 3.0, 4.0, None],
        'Score_MV': [2.7, 0.3, 0.3, 0.3, 0.3],
        'Risk_D': [1.508, 0.300, 0.402, 0.300, 0.300],
        'District_Loss': [0, 1, 0, 0, 0],
        'PROB': [0.1, 0.1, 0.1, 0.1, 0.1],
        'RiSk_E': [0.508, 0.400, 0.502, 0.400, 0.400],
        'History': [2, 1, 2, 2, 1],
        'Prob': [0.2, 0.2, 0.2, 0.2, 0.2],
        'Risk_F': [2.508, 0.500, 0.602, 0.500, 0.500],
        'Score': [1.8, 0.2, 0.2, 0.2, 0.2],
        'Inherent_Risk': [0.508, 0.300, 0.202, 0.200, 0.200],
        'CONTROL_RISK': [0.400, 0.400, 0.400, 0.400, 0.400],
        'Detection_Risk': [0.500, 0.500, 0.500, 0.500, 0.500],
        'Audit_Risk': [1.7148, 0.5108, 0.3096, 3.5060, 0.2832],
        'Risk': [1, 0, 0, 1, 0]
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

def test_process_data(monkeypatch, tmp_path, test_dataset):
    def mock_download_data_from_gcs(bucket_name, source_blob_name, destination_file_name):
        test_dataset.to_csv(destination_file_name, index=False)

    def mock_save_data_to_gcs(bucket_name, source_file_name, destination_blob_name):
        pass

    monkeypatch.setattr('data_preprocessing.cleaning.download_data_from_gcs', mock_download_data_from_gcs)
    monkeypatch.setattr('data_preprocessing.cleaning.save_data_to_gcs', mock_save_data_to_gcs)

    temp_file = tmp_path / "temp.csv"
    process_data('mlops-fraud-detect-audit-app', 'source.csv', 'destination.csv', temp_file)

    df_cleaned = load_data(temp_file)
    assert df_cleaned.duplicated().sum() == 0
    assert df_cleaned.isnull().sum().sum() == 0