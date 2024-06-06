import sys
import os
import pytest
import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

# Append the root directory of the project to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from model_training.train import feature_selection, train_and_evaluate

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df['Risk'] = y
    return df

def test_feature_selection(sample_data):
    X_selected, y, kbest_selector, selected_features = feature_selection(sample_data, target_column='Risk', n_features=10)
    assert X_selected.shape[1] == 10
    assert len(selected_features) == 10

def test_train_and_evaluate(sample_data, tmpdir):
    X = sample_data.drop(columns=['Risk'])
    y = sample_data['Risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.linear_model import LogisticRegression
    model_name = 'LogisticRegression'
    model = LogisticRegression()
    params = {'C': [0.1, 1, 10]}

    best_model, accuracy, precision, recall, f1 = train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model, params)

    assert accuracy > 0
    assert precision > 0
    assert recall > 0
    assert f1 > 0

    # Save and load model to check if the saving functionality works correctly
    model_path = os.path.join(tmpdir, f'{model_name}.joblib')
    joblib.dump(best_model, model_path)
    loaded_model = joblib.load(model_path)

    assert isinstance(loaded_model, LogisticRegression)

if __name__ == '__main__':
    pytest.main()
