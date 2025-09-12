# train_model.py
import os
import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

KAGGLE_URL = "https://www.kaggle.com/datasets/hussainsheikh03/nlp-based-cyber-security-dataset"

# Column mapping for auto-detection
COLUMN_MAPPING = {
    "Cleaned Threat Description": ["Threat Description", "Description", "Text"],
    "Attack Vector": ["AttackVector", "Vector", "Attack_Type"],
    "Threat Actor": ["Actor", "ThreatActor", "Actor_Name"],
    "Label": ["Threat Category", "Category", "Label", "Type"]
}

def detect_column(df, candidates):
    """Detect column in df from list of candidate names"""
    df_cols = [c.strip().lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in df_cols:
            return df.columns[df_cols.index(cand.lower())]
    # fallback: first object column
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    return obj_cols[0] if obj_cols else df.columns[0]

def prepare_dataset(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    
    text_col = detect_column(df, COLUMN_MAPPING["Cleaned Threat Description"])
    vector_col = detect_column(df, COLUMN_MAPPING["Attack Vector"])
    actor_col = detect_column(df, COLUMN_MAPPING["Threat Actor"])
    label_col = detect_column(df, COLUMN_MAPPING["Label"])
    
    # Combine text features
    df["combined_text"] = df[text_col].astype(str) + " " + \
                          df.get(vector_col, "").astype(str) + " " + \
                          df.get(actor_col, "").astype(str)
    return df, "combined_text", label_col

def main(csv=None, out_dir="artifacts", max_features=10000):
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset
    if csv is None:
        local_path = "dataset/cybersecurity.csv"
        if os.path.exists(local_path):
            csv = local_path
        else:
            try:
                import opendatasets as od
                print("Dataset not found locally — attempting Kaggle download...")
                od.download(KAGGLE_URL, data_dir="dataset")
                candidate = "dataset/nlp-based-cyber-security-dataset/cybersecurity.csv"
                if os.path.exists(candidate):
                    csv = candidate
                else:
                    raise FileNotFoundError("CSV not found even after Kaggle download.")
            except Exception as e:
                raise RuntimeError(f"Failed to auto-download dataset. Please place CSV in dataset/. Error: {e}")

    df, text_col, label_col = prepare_dataset(csv)
    print(f"✅ Using text column: '{text_col}', label column: '{label_col}'")

    X = df[text_col].astype(str).values
    y = df[label_col].astype(str).values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF vectorization
    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")
    X_train_t = vect.fit_transform(X_train)
    X_test_t = vect.transform(X_test)

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Classifier
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    print("🔄 Training classifier...")
    clf.fit(X_train_t, y_train_enc)

    # Evaluate
    preds = clf.predict(X_test_t)
    print("\n=== Classification Report ===")
    print(classification_report(y_test_enc, preds, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test_enc, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - NLP Cyber Security")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    # Save artifacts
    bundle_path = os.path.join(out_dir, "model_bundle.joblib")
    joblib.dump({
        "model": clf,
        "vectorizer": vect,
        "label_encoder": le,
        "classes": le.classes_,
        "text_col": text_col,
        "label_col": label_col,
    }, bundle_path)
    print(f"💾 Saved artifacts to {bundle_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=False, help="Path to CSV file (optional).")
    parser.add_argument("--out", default="artifacts", help="Artifacts output directory")
    parser.add_argument("--max_features", type=int, default=10000, help="TF-IDF max features")
    args = parser.parse_args()
    main(csv=args.csv, out_dir=args.out, max_features=args.max_features)
