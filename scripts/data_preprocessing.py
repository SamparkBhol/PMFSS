# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml
import logging

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_preprocessing")

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_data(data):
    # Drop any rows with missing values
    data_cleaned = data.dropna()
    logger.info(f"Data cleaned. Removed rows with missing values: {len(data) - len(data_cleaned)} rows dropped")
    return data_cleaned

def encode_labels(data, target_column):
    le = LabelEncoder()
    data[target_column] = le.fit_transform(data[target_column])
    logger.info(f"Target column '{target_column}' encoded with labels: {list(le.classes_)}")
    return data, le

def normalize_features(data, feature_columns):
    scaler = StandardScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    logger.info(f"Feature columns '{feature_columns}' normalized")
    return data, scaler

def split_data(data, target_column, test_size=0.2, random_state=42):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info(f"Data split into training and test sets with test size {test_size}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load raw data
    data_file = config["data"]["raw_data_path"]
    data = load_data(data_file)

    # Clean the data
    data_cleaned = clean_data(data)

    # Encode the target labels
    target_column = config["data"]["target_column"]
    data_encoded, label_encoder = encode_labels(data_cleaned, target_column)

    # Normalize the features
    feature_columns = config["model"]["input_features"]
    data_normalized, scaler = normalize_features(data_encoded, feature_columns)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = split_data(data_normalized, target_column)

    # Save the processed data
    processed_data_path = config["data"]["processed_data_path"]
    X_train.to_csv(f"{processed_data_path}/X_train.csv", index=False)
    X_test.to_csv(f"{processed_data_path}/X_test.csv", index=False)
    y_train.to_csv(f"{processed_data_path}/y_train.csv", index=False)
    y_test.to_csv(f"{processed_data_path}/y_test.csv", index=False)

    logger.info("Data preprocessing completed and saved to processed data directory")
