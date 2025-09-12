# # anomaly.py
# import numpy as np
# from sklearn.ensemble import IsolationForest
# from sklearn.svm import OneClassSVM

# class AnomalyDetector:
#     def __init__(self, method="isolation_forest"):
#         if method == "isolation_forest":
#             self.model = IsolationForest(contamination=0.05, random_state=42)
#         elif method == "ocsvm":
#             self.model = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
#         else:
#             raise ValueError("Unsupported method: choose 'isolation_forest' or 'ocsvm'")

#     def fit(self, X):
#         self.model.fit(X)

#     def score(self, X):
#         if hasattr(self.model, "decision_function"):
#             return self.model.decision_function(X)
#         return -self.model.score_samples(X)

#     def predict(self, X):
#         return self.model.predict(X) == -1  # True if anomaly
