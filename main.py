# main.py

import os
import logging
import yaml
import pandas as pd
import xgboost as xgb
import h2o
from h2o.automl import H2OAutoML
from kubeflow import kfp
from kubeflow.kfp import Client
from kubeflow.kfp.compiler import Compiler
from gradio import Interface, inputs, outputs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set up logging
logging.config.dictConfig(config["logging"])
logger = logging.getLogger("main")

# Initialize H2O cluster
logger.info("Initializing H2O cluster...")
h2o.init(nthreads=config["h2o"]["nthreads"],
         max_mem_size=config["h2o"]["max_mem_size"],
         min_mem_size=config["h2o"]["min_mem_size"],
         port=54321)

# Load and preprocess data
logger.info("Loading and preprocessing data...")
raw_data_path = config["data"]["raw_data_path"]
processed_data_path = config["data"]["processed_data_path"]

if not os.path.exists(processed_data_path):
    data = pd.read_csv(raw_data_path)
    # Preprocessing steps
    data.dropna(inplace=True)
    data.to_csv(processed_data_path, index=False)
else:
    data = pd.read_csv(processed_data_path)

# Split data for XGBoost
X = data.drop(columns=["failure"])
y = data["failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config["xgboost"]["random_state"])

# Train XGBoost model
logger.info("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=config["xgboost"]["n_estimators"],
    learning_rate=config["xgboost"]["learning_rate"],
    max_depth=config["xgboost"]["max_depth"],
    subsample=config["xgboost"]["subsample"],
    colsample_bytree=config["xgboost"]["colsample_bytree"],
    random_state=config["xgboost"]["random_state"],
    objective=config["xgboost"]["objective"],
    eval_metric=config["xgboost"]["eval_metric"]
)

xgb_model.fit(X_train, y_train, early_stopping_rounds=config["xgboost"]["early_stopping_rounds"],
              eval_set=[(X_test, y_test)], verbose=True)

# Evaluate XGBoost model
logger.info("Evaluating XGBoost model...")
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
logger.info(f"XGBoost Model Accuracy: {accuracy}")
logger.info(f"XGBoost Model ROC-AUC: {roc_auc}")

# Deploy model using Kubeflow
logger.info("Deploying model using Kubeflow...")
client = Client()
compiler = Compiler()
pipeline_path = "pipelines/kubeflow_pipeline.py"
pipeline_func = lambda: None  # Placeholder for actual pipeline function
compiler.compile(pipeline_func, pipeline_path)

experiment_name = config["kubeflow"]["experiment_name"]
run_name = config["kubeflow"]["run_name"]
client.create_run_from_pipeline_func(pipeline_func, experiment_name=experiment_name, run_name=run_name)

# Gradio Dashboard Interface
logger.info("Starting Gradio dashboard...")
def predict(input_file):
    df = pd.read_csv(input_file)
    predictions = xgb_model.predict(df)
    return predictions

iface = Interface(fn=predict,
                  inputs=inputs.File(label="Upload Space Station Data"),
                  outputs=outputs.Textbox(label="Predictions"),
                  live=config["gradio"]["live"])

iface.launch(server_port=config["gradio"]["port"])

# Email Notification for Alerts
if config["email_notifications"]["enabled"]:
    import smtplib
    from email.mime.text import MIMEText

    logger.info("Sending alert notification...")
    smtp_server = config["email_notifications"]["smtp_server"]
    port = config["email_notifications"]["port"]
    sender_email = config["email_notifications"]["auth"]["username"]
    password = config["email_notifications"]["auth"]["password"]
    recipient_email = config["email_notifications"]["recipient_email"]
    subject = config["email_notifications"]["subject"]
    body = config["email_notifications"]["body"]

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, recipient_email, msg.as_string())

logger.info("All tasks completed successfully.")
