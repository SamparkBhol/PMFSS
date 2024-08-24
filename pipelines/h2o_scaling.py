# h2o_scaling.py

import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import joblib
import yaml
import os

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Initialize H2O cluster
h2o.init()

# Load and preprocess data
data_path = config["data"]["processed_data_path"]
logger.info(f"Loading data from {data_path}...")
data = pd.read_csv(data_path)

# Convert data to H2OFrame
logger.info("Converting data to H2OFrame...")
hf = h2o.H2OFrame(data)

# Split data into training and test sets
train, test = hf.split_frame(ratios=[.8], seed=config["h2o"]["seed"])

# Set predictors and response variable
x = train.columns[:-1]
y = "failure"

# Train the model using H2O AutoML
logger.info("Training model using H2O AutoML...")
aml = H2OAutoML(max_models=20, seed=config["h2o"]["seed"], stopping_metric="AUC")
aml.train(x=x, y=y, training_frame=train)

# Get the best model
best_model = aml.leader

# Save the model
model_save_path = "models/h2o_model"
h2o.save_model(model=best_model, path=model_save_path, force=True)
logger.info(f"Best model saved at {model_save_path}")

# Evaluate the model on the test set
logger.info("Evaluating the model on the test set...")
perf = best_model.model_performance(test)
logger.info(f"Model performance:\n{perf}")

# Export the model as a standalone model
standalone_model_path = "models/standalone_h2o_model"
best_model.download_mojo(standalone_model_path)
logger.info(f"Standalone H2O model saved at {standalone_model_path}")

# Stop the H2O cluster
h2o.shutdown(prompt=False)
logger.info("H2O cluster shut down.")

