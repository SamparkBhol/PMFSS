# alerts.py

import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml
import logging
from datetime import datetime

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alerts")

# Load the model
model_path = config["model"]["path"]
model = joblib.load(model_path)

# Email setup
smtp_server = config["alerts"]["smtp_server"]
smtp_port = config["alerts"]["smtp_port"]
smtp_user = config["alerts"]["smtp_user"]
smtp_password = config["alerts"]["smtp_password"]

def send_email_alert(subject, body, to_email):
    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_email, msg.as_string())
        logger.info(f"Email alert sent to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

def check_for_alerts(sensor_data):
    df = pd.DataFrame([sensor_data], columns=config["model"]["input_features"])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    logger.info(f"Prediction: {prediction}, Probability: {probability}")

    if prediction == 1 and probability > config["alerts"]["probability_threshold"]:
        logger.warning("Potential failure detected!")
        
        # Send an email alert
        subject = "Space Station Alert: Potential Failure Detected"
        body = (f"Potential failure detected in space station equipment. "
                f"Failure probability: {probability:.2f}\n"
                f"Sensor data: {sensor_data}\n"
                f"Timestamp: {datetime.now()}")
        send_email_alert(subject, body, config["alerts"]["notification_email"])

        return "Alert triggered", probability
    else:
        return "No alert", probability

if __name__ == "__main__":
    # Example sensor data
    sensor_data = {
        "sensor1": 0.85,
        "sensor2": 0.76,
        "sensor3": 0.94,
        "sensor4": 0.55,
        "sensor5": 0.67
    }

    # Check for alerts
    result, prob = check_for_alerts(sensor_data)
    logger.info(f"Alert check result: {result}, Failure probability: {prob:.2f}")
