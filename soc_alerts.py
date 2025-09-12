# soc_alerts.py
import requests
import smtplib
from email.mime.text import MIMEText

def send_slack_alert(message, webhook_url="YOUR_SLACK_WEBHOOK"):
    try:
        requests.post(webhook_url, json={"text": message})
    except Exception as e:
        print(f"Slack alert failed: {e}")

def send_email_alert(subject, body, to_email, from_email, smtp_server="smtp.gmail.com", smtp_port=587, password=""):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, [to_email], msg.as_string())
        server.quit()
    except Exception as e:
        print(f"Email alert failed: {e}")

def send_webhook_alert(url, payload):
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Webhook alert failed: {e}")
