# kubeflow_pipeline.py

import kfp
from kfp import dsl
from kubernetes import client as k8s_client
import yaml
import os

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def load_data_op():
    return dsl.ContainerOp(
        name="Load Data",
        image="python:3.8-slim",
        command=["python", "-c"],
        arguments=[
            """
            import pandas as pd
            import os

            # Load and preprocess data
            data_path = 'data/raw_data.csv'
            processed_data_path = 'data/processed_data.csv'
            df = pd.read_csv(data_path)

            # Add any preprocessing steps here
            df.to_csv(processed_data_path, index=False)
            """
        ],
        file_outputs={
            'processed_data': 'data/processed_data.csv'
        }
    )

def train_model_op(processed_data):
    return dsl.ContainerOp(
        name="Train XGBoost Model",
        image="python:3.8-slim",
        command=["python", "-c"],
        arguments=[
            """
            import pandas as pd
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            import joblib

            data_path = 'data/processed_data.csv'
            model_save_path = 'models/xgboost_model.joblib'

            data = pd.read_csv(data_path)
            X = data.drop(columns=['failure'])
            y = data['failure']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
            model.fit(X_train, y_train)

            joblib.dump(model, model_save_path)
            """
        ],
        file_outputs={
            'model': 'models/xgboost_model.joblib'
        }
    ).add_pvolumes({"/mnt": k8s_client.V1VolumeMount(mount_path="/mnt")})

def evaluate_model_op(model):
    return dsl.ContainerOp(
        name="Evaluate Model",
        image="python:3.8-slim",
        command=["python", "-c"],
        arguments=[
            """
            import pandas as pd
            import joblib
            from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

            model_path = 'models/xgboost_model.joblib'
            data_path = 'data/processed_data.csv'

            data = pd.read_csv(data_path)
            X = data.drop(columns=['failure'])
            y = data['failure']

            model = joblib.load(model_path)
            y_pred = model.predict(X)

            accuracy = accuracy_score(y, y_pred)
            roc_auc = roc_auc_score(y, y_pred)

            with open('metrics.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\\nROC AUC: {roc_auc}\\n")
            """
        ],
        file_outputs={
            'metrics': 'metrics.txt'
        }
    )

def deploy_model_op(model):
    return dsl.ContainerOp(
        name="Deploy Model",
        image="python:3.8-slim",
        command=["python", "-c"],
        arguments=[
            """
            import joblib
            import os

            model_path = 'models/xgboost_model.joblib'
            deployment_path = '/mnt/deployment/model.joblib'

            os.makedirs('/mnt/deployment', exist_ok=True)
            joblib.copy(model_path, deployment_path)
            """
        ],
        pvolumes={"/mnt": k8s_client.V1VolumeMount(mount_path="/mnt")}
    )

@dsl.pipeline(
    name="Space Station Predictive Maintenance Pipeline",
    description="A pipeline that trains, evaluates, and deploys a predictive maintenance model for space stations."
)
def predictive_maintenance_pipeline():
    load_data = load_data_op()
    train_model = train_model_op(load_data.output)
    evaluate_model = evaluate_model_op(train_model.output)
    deploy_model = deploy_model_op(train_model.output)

    # Sequential dependencies
    train_model.after(load_data)
    evaluate_model.after(train_model)
    deploy_model.after(evaluate_model)

if __name__ == "__main__":
    # Compile the pipeline
    kfp.compiler.Compiler().compile(predictive_maintenance_pipeline, 'kubeflow_pipeline.yaml')

    # Optionally, you can run the pipeline
    # client = kfp.Client()
    # client.create_run_from_pipeline_func(predictive_maintenance_pipeline, arguments={})
