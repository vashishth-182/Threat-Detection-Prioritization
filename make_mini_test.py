# make_mini_test.py
import pandas as pd

TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

# NSL-KDD has 43 features + label + difficulty level
col_names = [
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

print("Downloading NSL-KDD Test+...")
df = pd.read_csv(TEST_URL, header=None, names=col_names)

# take only first 200 rows for quick demo
mini_df = df.head(200)

# save as txt (same format as original, no header)
mini_df.to_csv("mini_test.txt", index=False, header=False)

print("mini_test.txt created with 200 rows ✅")
