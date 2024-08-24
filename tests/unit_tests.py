# unit_tests.py

import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import joblib
import yaml

from scripts.data_preprocessing import load_data, clean_data, encode_labels, normalize_features, split_data
from scripts.feature_importance import load_model, calculate_feature_importance

# Load configuration
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.data_file = config["data"]["raw_data_path"]
        self.target_column = config["data"]["target_column"]
        self.feature_columns = config["model"]["input_features"]
        self.data = load_data(self.data_file)

    def test_load_data(self):
        self.assertIsInstance(self.data, pd.DataFrame, "Data should be loaded as a DataFrame")

    def test_clean_data(self):
        cleaned_data = clean_data(self.data)
        self.assertFalse(cleaned_data.isnull().values.any(), "Cleaned data should not have any missing values")

    def test_encode_labels(self):
        data_encoded, label_encoder = encode_labels(self.data, self.target_column)
        self.assertIsInstance(label_encoder, LabelEncoder, "LabelEncoder should be used for encoding")
        self.assertTrue(data_encoded[self.target_column].dtype == 'int32' or 'int64', "Target column should be integer after encoding")

    def test_normalize_features(self):
        data_normalized, scaler = normalize_features(self.data, self.feature_columns)
        self.assertIsInstance(scaler, StandardScaler, "StandardScaler should be used for normalization")
        self.assertTrue((data_normalized[self.feature_columns].mean().round(6) == 0).all(), "Normalized features should have mean 0")

    def test_split_data(self):
        X_train, X_test, y_train, y_test = split_data(self.data, self.target_column)
        self.assertEqual(len(X_train) + len(X_test), len(self.data), "Total samples after splitting should equal original data")
        self.assertEqual(len(y_train) + len(y_test), len(self.data), "Total labels after splitting should equal original data")

class TestFeatureImportance(unittest.TestCase):

    def setUp(self):
        self.model_path = config["model"]["path"]
        self.feature_columns = config["model"]["input_features"]

    def test_load_model(self):
        model = load_model(self.model_path)
        self.assertIsInstance(model, XGBClassifier, "Model should be an instance of XGBClassifier")

    def test_calculate_feature_importance(self):
        model = load_model(self.model_path)
        importance_df = calculate_feature_importance(model, self.feature_columns)
        self.assertEqual(len(importance_df), len(self.feature_columns), "Importance DataFrame should have an entry for each feature")
        self.assertIn("Importance", importance_df.columns, "Importance DataFrame should contain an 'Importance' column")

if __name__ == "__main__":
    unittest.main()
