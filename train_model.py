# train_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# --- dataset URLs
TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
TEST_URL  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

# --- column names
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

# --- map attack types to categories
def map_attack_category(name):
    name = name.strip()
    if name == 'normal':
        return 'Normal'
    DOS = {"back","land","neptune","pod","smurf","teardrop","apache2","udpstorm","processtable","worm"}
    PROBE = {"satan","ipsweep","nmap","portsweep","mscan","saint"}
    R2L = {"ftp_write","guess_passwd","imap","phf","multihop","warezmaster","warezclient","spy","xlock","xsnoop","snmpgetattack","snmpguess","httptunnel","sendmail","named"}
    U2R = {"buffer_overflow","loadmodule","perl","rootkit","sqlattack","xterm","ps"}
    if name in DOS: return 'DoS'
    if name in PROBE: return 'Probe'
    if name in R2L: return 'R2L'
    if name in U2R: return 'U2R'
    return 'Other'

def load_df(url):
    return pd.read_csv(url, header=None, names=COLUMN_NAMES)

def preprocess(train_df, test_df):
    # drop difficulty if present
    for df in (train_df, test_df):
        if "difficulty_level" in df.columns:
            df.drop("difficulty_level", axis=1, inplace=True)

    # add attack_category
    for df in (train_df, test_df):
        df['attack_category'] = df['label'].apply(map_attack_category)

    # categorical encode
    cat_cols = ['protocol_type', 'service', 'flag']
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([train_df[c], test_df[c]]))
        train_df[c] = le.transform(train_df[c])
        test_df[c] = le.transform(test_df[c])
        encoders[c] = le

    drop_cols = ['label']
    feature_cols = [c for c in train_df.columns if c not in drop_cols + ['attack_category']]

    # scale
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train_df[feature_cols]), columns=feature_cols)
    X_test  = pd.DataFrame(scaler.transform(test_df[feature_cols]), columns=feature_cols)

    y_train = train_df['attack_category']
    y_test  = test_df['attack_category']

    return X_train, X_test, y_train, y_test, encoders, scaler, feature_cols

def train_and_save(X_train, y_train, X_test, y_test, encoders, scaler, feature_cols):
    clf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))

    # confusion matrix
    cm = confusion_matrix(y_test, preds, labels=clf.classes_)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=clf.classes_, yticklabels=clf.classes_, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - NSL-KDD Multi-class")
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/confusion_matrix.png")
    plt.close()

    # save artifacts
    joblib.dump({
        'model': clf,
        'scaler': scaler,
        'encoders': encoders,
        'feature_cols': feature_cols
    }, 'artifacts/model_bundle.joblib')
    print("Artifacts saved in artifacts/")

if __name__ == "__main__":
    print("Loading dataset...")
    train_df = load_df(TRAIN_URL)
    test_df  = load_df(TEST_URL)
    X_train, X_test, y_train, y_test, encoders, scaler, feature_cols = preprocess(train_df, test_df)
    train_and_save(X_train, y_train, X_test, y_test, encoders, scaler, feature_cols)
