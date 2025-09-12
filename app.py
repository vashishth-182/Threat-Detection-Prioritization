import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import shap
from sklearn.ensemble import IsolationForest
import requests
from real_time_ingest import start_watching_in_thread

st.set_page_config(page_title="NLP Threat Detection", layout="wide")

ARTIFACT_PATH = "artifacts/model_bundle.joblib"
SAMPLE_DIR = "sample_files"
LOG_FILE = "processed_logs.csv"  # persistent logs

COLUMN_MAPPING = {
    "Cleaned Threat Description": ["Threat Description", "Description", "Text"],
    "Attack Vector": ["AttackVector", "Vector", "Attack_Type"],
    "Threat Actor": ["Actor", "ThreatActor", "Actor_Name"]
}

# -----------------------------
# Load model artifacts
# -----------------------------
@st.cache_resource
def load_artifacts(path=ARTIFACT_PATH):
    bundle = joblib.load(path)
    model = bundle['model']
    vectorizer = bundle['vectorizer']
    label_encoder = bundle.get('label_encoder')
    text_col = bundle.get('text_col', 'text')
    classes = bundle.get('classes', label_encoder.classes_ if label_encoder else model.classes_)
    return model, vectorizer, label_encoder, text_col, classes

artifacts_ready = False
if os.path.exists(ARTIFACT_PATH):
    try:
        model, vectorizer, label_encoder, text_col_default, classes = load_artifacts()
        artifacts_ready = True
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
else:
    st.warning("Model artifacts not found. Run train_model.py first.")

st.title("🛡️ NLP-based Threat Detection & Prioritization")
st.markdown("Upload a CSV with text and threat columns. Column names can vary; the app will try to auto-detect them.")

# -----------------------------
# Initialize session log storage
# -----------------------------
if 'logs_df' not in st.session_state:
    if os.path.exists(LOG_FILE):
        st.session_state.logs_df = pd.read_csv(LOG_FILE, low_memory=False)
    else:
        st.session_state.logs_df = pd.DataFrame()

# -----------------------------
# Function to process new logs
# -----------------------------
def process_new_logs(df_new):
    # Detect columns ignoring case
    def detect_column(df, candidates):
        df_cols = [c.strip().lower() for c in df.columns]
        for cand in candidates:
            if cand.lower() in df_cols:
                return df.columns[df_cols.index(cand.lower())]
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        return obj_cols[0] if obj_cols else df.columns[0]

    text_col = detect_column(df_new, COLUMN_MAPPING["Cleaned Threat Description"])
    vector_col = detect_column(df_new, COLUMN_MAPPING["Attack Vector"])
    actor_col = detect_column(df_new, COLUMN_MAPPING["Threat Actor"])

    # Combine text
    df_new["combined_text"] = df_new[text_col].astype(str) + " " + \
                              df_new[vector_col].astype(str) + " " + \
                              df_new[actor_col].astype(str)

    # Numeric feature
    if 'event_type' in df_new.columns:
        df_new['Failed_Login'] = df_new['event_type'].apply(lambda x: 1 if 'failed login' in str(x).lower() else 0)
    else:
        df_new['Failed_Login'] = 0

    # -----------------------------
    # Use only TF-IDF features
    # -----------------------------
    X_tfidf = vectorizer.transform(df_new["combined_text"])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_tfidf)
        best_idx = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
    else:
        best_idx = model.predict(X_tfidf)
        confs = np.ones(len(best_idx))

    preds = np.array(classes)[best_idx]
    df_new['Predicted'] = preds
    df_new['Confidence'] = np.round(confs, 3)

    # Severity mapping
    DEFAULT_SEVERITY = {
        'malware': 1.0,
        'ransomware': 1.0,
        'intrusion': 0.9,
        'dos': 0.9,
        'phishing': 0.8,
        'spam': 0.3,
        'benign': 0.0,
        'normal': 0.0
    }
    df_new['Severity'] = df_new['Predicted'].apply(lambda x: DEFAULT_SEVERITY.get(str(x).lower(), 0.5))

    # Anomaly detection on TF-IDF
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_tfidf)
    df_new['Anomaly_Score'] = iso_forest.decision_function(X_tfidf)
    df_new['Is_Anomaly'] = iso_forest.predict(X_tfidf) == -1

    # PriorityScore
    df_new['PriorityScore'] = (df_new['Severity'] * df_new['Confidence'] + df_new['Is_Anomaly'] * 0.3).round(3)

    # SHAP explanation (TF-IDF only)
    explainer = shap.LinearExplainer(model, X_tfidf)
    shap_values = explainer.shap_values(X_tfidf)
    top_features = []
    for i in range(X_tfidf.shape[0]):
        row_shap = shap_values[0][i] if isinstance(shap_values, list) else shap_values[i]
        indices = row_shap.argsort()[-5:][::-1]
        words = [str(vectorizer.get_feature_names_out()[idx]) for idx in indices]
        top_features.append(", ".join(words))
    df_new['Top_Features'] = top_features

    # Slack alerts
    def send_slack_alert(text, webhook_url="YOUR_SLACK_WEBHOOK"):
        try:
            requests.post(webhook_url, json={"text": text})
        except:
            pass

    for _, row in df_new.iterrows():
        if row['PriorityScore'] > 0.8:
            send_slack_alert(f"⚠️ High priority threat detected: {row['combined_text']} ({row['Predicted']})")

    # Append to session logs
    st.session_state.logs_df = pd.concat([st.session_state.logs_df, df_new], ignore_index=True)
    st.session_state.logs_df.to_csv(LOG_FILE, index=False)

    return df_new

# -----------------------------
# Start real-time log watcher
# -----------------------------
if artifacts_ready:
    start_watching_in_thread(process_new_logs)

# -----------------------------
# Sidebar sample files
# -----------------------------
st.sidebar.header("Sample files")
if os.path.exists(SAMPLE_DIR):
    for f in sorted(os.listdir(SAMPLE_DIR)):
        if f.endswith(".csv"):
            with open(os.path.join(SAMPLE_DIR, f), "rb") as fh:
                st.sidebar.download_button(f"Download {f}", fh, file_name=f, mime="text/csv")

# -----------------------------
# File upload
# -----------------------------
uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"])
use_sample = st.checkbox("Use sample file", value=False)

df = None
if use_sample and os.path.exists(SAMPLE_DIR):
    sample_files = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(".csv")]
    if sample_files:
        chosen = st.selectbox("Select sample file", sample_files)
        if chosen:
            df = pd.read_csv(os.path.join(SAMPLE_DIR, chosen))
elif uploaded:
    df = pd.read_csv(uploaded, low_memory=False)

if df is not None:
    df = process_new_logs(df)

    # Display Streamlit
    st.subheader("Top predictions")
    st.dataframe(df.head(50))

    st.subheader("Prediction distribution")
    st.bar_chart(df['Predicted'].value_counts())

    st.subheader("Top priority alerts")
    st.dataframe(df[['combined_text','Predicted','Confidence','Severity','PriorityScore']].head(20))

    # WordCloud
    top_text = " ".join(df.loc[df['PriorityScore']>0,'combined_text'].astype(str).tolist())
    if top_text.strip() == "":
        top_text = " ".join(df['combined_text'].astype(str).tolist())
    st.subheader("Word Cloud (Top priority / all)")
    wc = WordCloud(width=800, height=300, background_color='white').generate(top_text)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Download predictions
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download predictions CSV", csv, file_name="predictions.csv", mime="text/csv")

# -----------------------------
# Display persistent logs
# -----------------------------
if not st.session_state.logs_df.empty:
    st.subheader("All Processed Logs")
    st.dataframe(st.session_state.logs_df.tail(50))
    with open(LOG_FILE, "rb") as f:
        st.download_button("Download All Processed Logs", f, file_name="processed_logs.csv", mime="text/csv")

st.success("✅ Done — review top priority alerts.")
