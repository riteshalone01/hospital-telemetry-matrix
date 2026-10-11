"""Train an Isolation Forest on hospital telemetry and export model.pkl.

The backend loads the pickle like this:

    import joblib
    bundle = joblib.load("model.pkl")
    model, features = bundle["model"], bundle["features"]
    X = [[voltage, pressure, flood]]
    is_anomaly = model.predict(X)[0] == -1
    score = model.decision_function(X)[0]   # lower = more anomalous
"""
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

FEATURES = ["voltage_v", "water_pressure_psi", "flood_index"]

df = pd.read_csv("hospital_telemetry.csv")
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.is_anomaly)

# Unsupervised: fit on the training features only (labels are never used for fitting)
model = IsolationForest(
    n_estimators=200,
    contamination=0.04,
    random_state=42,
    n_jobs=-1,
)
model.fit(train[FEATURES])

pred = (model.predict(test[FEATURES]) == -1).astype(int)
print(confusion_matrix(test.is_anomaly, pred))
print(classification_report(test.is_anomaly, pred, target_names=["normal", "anomaly"]))

joblib.dump({"model": model, "features": FEATURES}, "model.pkl")
print("Saved model.pkl")
