# app.py
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Threat Detection Dashboard", layout="wide")

# column names (same as training)
COLUMN_NAMES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty_level"
]

@st.cache_resource
def load_artifacts():
    bundle = joblib.load("artifacts/model_bundle.joblib")
    return bundle['model'], bundle['scaler'], bundle['encoders'], bundle['feature_cols']

model, scaler, encoders, feature_cols = load_artifacts()

st.title("AI-driven Threat Detection (NSL-KDD Multi-class)")

uploaded = st.file_uploader("Upload NSL-KDD formatted test file", type=['txt','csv'])

def preprocess_df(df):
    df.columns = COLUMN_NAMES
    if "difficulty_level" in df.columns:
        df.drop("difficulty_level", axis=1, inplace=True)
    # encode categorical
    for c, le in encoders.items():
        if c in df.columns:
            df[c] = le.transform(df[c])
    X = scaler.transform(df[feature_cols])
    return pd.DataFrame(X, columns=feature_cols), df

if uploaded:
    raw = pd.read_csv(uploaded, header=None)
    X_scaled, df_with_labels = preprocess_df(raw)

    preds = model.predict(X_scaled)
    df_with_labels['Predicted'] = preds

    st.subheader("Sample Predictions")
    st.dataframe(df_with_labels.head(20))

    st.subheader("Prediction Distribution")
    st.bar_chart(df_with_labels['Predicted'].value_counts())

    st.subheader("Confusion Matrix (From Training)")
    st.image("artifacts/confusion_matrix.png")

else:
    st.info("Upload NSL-KDD Test+ file to see predictions and charts.")
