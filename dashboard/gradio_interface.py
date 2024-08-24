# gradio_interface.py

import gradio as gr
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load the model
model_path = config["model"]["path"]
model = joblib.load(model_path)

# Define a function for making predictions
def predict_failure(input_data):
    # Input data preprocessing
    df = pd.DataFrame([input_data], columns=config["model"]["input_features"])
    
    # Make predictions
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {"Prediction": int(prediction), "Failure Probability": float(probability)}

# Define a function to generate model evaluation reports
def generate_reports(input_data, true_labels):
    # Input data preprocessing
    df = pd.DataFrame(input_data, columns=config["model"]["input_features"])
    
    # Make predictions
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    # Calculate metrics
    auc = roc_auc_score(true_labels, probabilities)
    report = classification_report(true_labels, predictions, output_dict=True)

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return {"AUC Score": auc, "Classification Report": report}

# Define the Gradio interface for prediction
def gradio_predict_interface():
    # Define input fields
    inputs = []
    for feature in config["model"]["input_features"]:
        inputs.append(gr.inputs.Number(label=feature))

    # Define output fields
    outputs = [gr.outputs.Textbox(label="Prediction"), gr.outputs.Textbox(label="Failure Probability")]

    # Create the Gradio interface
    interface = gr.Interface(
        fn=predict_failure,
        inputs=inputs,
        outputs=outputs,
        title="Space Station Predictive Maintenance",
        description="Input sensor data to predict potential failures in space station equipment."
    )
    return interface

# Define the Gradio interface for generating reports
def gradio_report_interface():
    # Define input fields
    data_input = gr.inputs.Dataframe(
        headers=config["model"]["input_features"] + ["True Label"],
        datatype="number",
        col_count=(len(config["model"]["input_features"]) + 1)
    )

    # Define output fields
    outputs = [gr.outputs.Textbox(label="AUC Score"), gr.outputs.JSON(label="Classification Report")]

    # Create the Gradio interface
    interface = gr.Interface(
        fn=generate_reports,
        inputs=data_input,
        outputs=outputs,
        title="Model Evaluation Reports",
        description="Upload sensor data and true labels to generate model evaluation reports."
    )
    return interface

if __name__ == "__main__":
    # Launch both interfaces
    gradio_predict_interface().launch()
    gradio_report_interface().launch()
