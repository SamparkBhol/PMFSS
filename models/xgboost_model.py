# xgboost_model.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import logging
import yaml
import joblib
import os

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set up logging
logging.config.dictConfig(config["logging"])
logger = logging.getLogger("xgboost_model")

# Load and preprocess data
logger.info("Loading and preprocessing data...")
data_path = config["data"]["processed_data_path"]
data = pd.read_csv(data_path)

# Splitting the data into features and target
X = data.drop(columns=["failure"])
y = data["failure"]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config["xgboost"]["random_state"])

# Initialize the XGBoost model
logger.info("Initializing the XGBoost model...")
xgb_model = xgb.XGBClassifier(
    objective=config["xgboost"]["objective"],
    random_state=config["xgboost"]["random_state"],
    eval_metric=config["xgboost"]["eval_metric"]
)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Perform grid search for hyperparameter tuning
logger.info("Starting hyperparameter tuning with GridSearchCV...")
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_xgb_model = grid_search.best_estimator_

# Save the model
model_save_path = "models/xgboost_model.joblib"
joblib.dump(best_xgb_model, model_save_path)
logger.info(f"Model saved at {model_save_path}")

# Evaluate the model on the test set
logger.info("Evaluating the model on the test set...")
y_pred = best_xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

logger.info(f"Accuracy: {accuracy}")
logger.info(f"ROC-AUC: {roc_auc}")
logger.info("Confusion Matrix:\n%s", conf_matrix)
logger.info("Classification Report:\n%s", class_report)

# Feature importance
logger.info("Calculating feature importance...")
importance = best_xgb_model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

feature_importance_save_path = "models/feature_importance.csv"
feature_importance.to_csv(feature_importance_save_path, index=False)
logger.info(f"Feature importance saved at {feature_importance_save_path}")

# Visualize the feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
feature_importance_plot_path = "models/feature_importance.png"
plt.savefig(feature_importance_plot_path)
logger.info(f"Feature importance plot saved at {feature_importance_plot_path}")

# Log the best parameters
best_params_save_path = "models/best_params.yaml"
with open(best_params_save_path, "w") as file:
    yaml.dump(grid_search.best_params_, file)
logger.info(f"Best model parameters saved at {best_params_save_path}")

# Function to make predictions on new data
def predict_new_data(new_data_path):
    new_data = pd.read_csv(new_data_path)
    predictions = best_xgb_model.predict(new_data)
    return predictions

# Example prediction on new data
simulation_data_path = config["data"]["simulation_data_path"]
if os.path.exists(simulation_data_path):
    logger.info(f"Making predictions on simulated data from {simulation_data_path}")
    predictions = predict_new_data(simulation_data_path)
    predictions_save_path = "models/simulated_predictions.csv"
    pd.DataFrame(predictions, columns=["Predicted_Failure"]).to_csv(predictions_save_path, index=False)
    logger.info(f"Simulated data predictions saved at {predictions_save_path}")

logger.info("XGBoost model training and evaluation completed successfully.")
