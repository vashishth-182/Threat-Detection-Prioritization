# make_mini_test.py
import os
import pandas as pd
import random

SAMPLE_DIR = "sample_files"
DATASET_PATH = "dataset/cybersecurity.csv"

os.makedirs(SAMPLE_DIR, exist_ok=True)

# Column mapping for auto-detection
COLUMN_MAPPING = {
    "Cleaned Threat Description": ["Threat Description", "Description", "Text"],
    "Attack Vector": ["AttackVector", "Vector", "Attack_Type"],
    "Threat Actor": ["Actor", "ThreatActor", "Actor_Name"],
    "Label": ["Threat Category", "Category", "Label", "Type"]
}

def detect_column(df, candidates):
    df_cols = [c.strip().lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in df_cols:
            return df.columns[df_cols.index(cand.lower())]
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    return obj_cols[0] if obj_cols else df.columns[0]

def create_from_dataset(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)

    text_col = detect_column(df, COLUMN_MAPPING["Cleaned Threat Description"])
    vector_col = detect_column(df, COLUMN_MAPPING["Attack Vector"])
    actor_col = detect_column(df, COLUMN_MAPPING["Threat Actor"])
    label_col = detect_column(df, COLUMN_MAPPING["Label"])

    # Combine text columns
    df["combined_text"] = df[text_col].astype(str) + " " + \
                          df.get(vector_col, "").astype(str) + " " + \
                          df.get(actor_col, "").astype(str)

    # Labeled sample
    labeled = df[[text_col, vector_col, actor_col, label_col, "combined_text"]].dropna().head(200)
    labeled.to_csv(os.path.join(SAMPLE_DIR, "sample_labeled.csv"), index=False)

    # Unlabeled sample
    unlabeled = labeled[["combined_text"]].copy()
    unlabeled.to_csv(os.path.join(SAMPLE_DIR, "sample_unlabeled.csv"), index=False)

    print(f"Created sample files in {SAMPLE_DIR}")

def create_synthetic():
    texts = [
        "Subject: Your account will be suspended. Verify now at http://fake.example",
        "Please find attached invoice for your recent purchase",
        "Detected ransomware encrypting multiple files on host 10.0.0.12",
        "Login successful for user alice from IP 203.0.113.10",
        "Large number of failed SSH logins from 192.0.2.5",
        "Click the link to reset your password: http://bad.example",
        "Monthly newsletter: updates and offers",
        "Unusual outbound traffic detected on port 4444",
        "Payment failed for order ID 12345",
        "Brute force attempt detected on admin account"
    ]
    labels = ["phishing","spam","malware","benign","intrusion","phishing","benign","malware","spam","intrusion"]

    rows = []
    for i in range(200):
        idx = random.randrange(len(texts))
        rows.append({
            "Cleaned Threat Description": texts[idx],
            "Attack Vector": "",
            "Threat Actor": "",
            "Threat Category": labels[idx],
            "combined_text": texts[idx]  # same for combined_text
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAMPLE_DIR, "sample_labeled.csv"), index=False)
    df[["combined_text"]].to_csv(os.path.join(SAMPLE_DIR, "sample_unlabeled.csv"), index=False)
    print(f"Created synthetic sample files in {SAMPLE_DIR}")

if __name__ == "__main__":
    if os.path.exists(DATASET_PATH):
        create_from_dataset(DATASET_PATH)
    else:
        print("Dataset not found at dataset/cybersecurity.csv — creating synthetic sample files.")
        create_synthetic()
