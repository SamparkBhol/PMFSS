# model_evaluation.py

import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
import os

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set up logging
logging.config.dictConfig(config["logging"])
logger = logging.getLogger("model_evaluation")

# Load the trained model
model_path = "models/xgboost_model.joblib"
if not os.path.exists(model_path):
    logger.error(f"Model not found at {model_path}. Please run xgboost_model.py first.")
    raise FileNotFoundError(f"Model not found at {model_path}")

logger.info(f"Loading the trained model from {model_path}...")
model = joblib.load(model_path)

# Load the test data
data_path = config["data"]["processed_data_path"]
logger.info(f"Loading data from {data_path}...")
data = pd.read_csv(data_path)

# Splitting the data into features and target
X = data.drop(columns=["failure"])
y = data["failure"]

# Make predictions on the test data
logger.info("Making predictions on the test data...")
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Calculate performance metrics
logger.info("Calculating performance metrics...")
accuracy = accuracy_score(y, y_pred)
roc_auc = roc_auc_score(y, y_prob)
conf_matrix = confusion_matrix(y, y_pred)
class_report = classification_report(y, y_pred)

logger.info(f"Accuracy: {accuracy}")
logger.info(f"ROC-AUC: {roc_auc}")
logger.info("Confusion Matrix:\n%s", conf_matrix)
logger.info("Classification Report:\n%s", class_report)

# Save performance metrics
performance_metrics = {
    "Accuracy": accuracy,
    "ROC-AUC": roc_auc,
    "Confusion Matrix": conf_matrix.tolist(),
    "Classification Report": class_report
}

metrics_save_path = "models/performance_metrics.yaml"
with open(metrics_save_path, "w") as file:
    yaml.dump(performance_metrics, file)
logger.info(f"Performance metrics saved at {metrics_save_path}")

# Plot ROC Curve
logger.info("Plotting ROC Curve...")
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
roc_curve_plot_path = "models/roc_curve.png"
plt.savefig(roc_curve_plot_path)
logger.info(f"ROC Curve plot saved at {roc_curve_plot_path}")

# Plot confusion matrix
logger.info("Plotting confusion matrix...")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=["No Failure", "Failure"], 
            yticklabels=["No Failure", "Failure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
conf_matrix_plot_path = "models/confusion_matrix.png"
plt.savefig(conf_matrix_plot_path)
logger.info(f"Confusion Matrix plot saved at {conf_matrix_plot_path}")

# Plot feature importance
logger.info("Plotting feature importance...")
feature_importance_path = "models/feature_importance.csv"
feature_importance = pd.read_csv(feature_importance_path)

plt.figure(figsize=(10, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
feature_importance_plot_path = "models/feature_importance_evaluation.png"
plt.savefig(feature_importance_plot_path)
logger.info(f"Feature importance plot (evaluation) saved at {feature_importance_plot_path}")

# Generate classification report as a plot
logger.info("Generating classification report plot...")
report_data = []
for line in class_report.split("\n")[2:6]:
    row = {}
    row_data = line.split()
    row['class'] = row_data[0]
    row['precision'] = float(row_data[1])
    row['recall'] = float(row_data[2])
    row['f1_score'] = float(row_data[3])
    report_data.append(row)
df = pd.DataFrame.from_dict(report_data)

plt.figure(figsize=(8, 6))
df.plot(kind='bar', x='class', y=['precision', 'recall', 'f1_score'], legend=True)
plt.ylim(0, 1)
plt.title('Classification Report Metrics')
plt.xlabel('Class')
plt.ylabel('Score')
plt.tight_layout()
classification_report_plot_path = "models/classification_report.png"
plt.savefig(classification_report_plot_path)
logger.info(f"Classification report plot saved at {classification_report_plot_path}")

logger.info("Model evaluation completed successfully.")
