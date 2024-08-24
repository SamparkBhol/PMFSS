# feature_importance.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
from xgboost import plot_importance
import numpy as np

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("feature_importance")

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def calculate_feature_importance(model, feature_names):
    importance = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    importance_df.sort_values(by="Importance", ascending=False, inplace=True)
    logger.info("Feature importance calculated")
    return importance_df

def plot_feature_importance(importance_df, top_n=10):
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(top_n))
    plt.title(f"Top {top_n} Most Important Features")
    plt.show()
    logger.info(f"Feature importance plot displayed for top {top_n} features")

if __name__ == "__main__":
    # Load the trained model
    model_path = config["model"]["path"]
    model = load_model(model_path)

    # Calculate feature importance
    feature_names = config["model"]["input_features"]
    importance_df = calculate_feature_importance(model, feature_names)

    # Plot the top N important features
    plot_feature_importance(importance_df, top_n=10)

    # Save the feature importance to a CSV file
    importance_path = config["model"]["importance_output_path"]
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to {importance_path}")
