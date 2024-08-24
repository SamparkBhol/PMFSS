# integration_tests.py

import unittest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from scripts.data_preprocessing import load_data, clean_data, encode_labels, normalize_features, split_data
from scripts.feature_importance import load_model
import yaml

# Load configuration
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.data_file = config["data"]["raw_data_path"]
        self.target_column = config["data"]["target_column"]
        self.feature_columns = config["model"]["input_features"]
        self.model_path = config["model"]["path"]

    def test_full_pipeline(self):
        # Step 1: Load and preprocess data
        data = load_data(self.data_file)
        cleaned_data = clean_data(data)
        data_encoded, _ = encode_labels(cleaned_data, self.target_column)
        data_normalized, _ = normalize_features(data_encoded, self.feature_columns)
        X_train, X_test, y_train, y_test = split_data(data_normalized, self.target_column)

        # Step 2: Load model and perform predictions
        model = load_model(self.model_path)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Step 3: Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        self.assertGreater(accuracy, 0.7, "Model accuracy should be greater than 0.7")
        self.assertGreater(auc, 0.75, "Model AUC should be greater than 0.75")

        print(f"Integration test passed with Accuracy: {accuracy:.4f} and AUC: {auc:.4f}")

if __name__ == "__main__":
    unittest.main()
