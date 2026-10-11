import joblib
import pandas as pd

# 1. Load the dictionary and extract the actual model object
model_data = joblib.load("model.pkl")

if isinstance(model_data, dict):
    model = model_data.get("model", model_data)
else:
    model = model_data

print("--- Telemetry Anomaly Detector Ready ---")

def check_telemetry(voltage, pressure, flood):
    new_data = pd.DataFrame([{
        "voltage_v": voltage,
        "water_pressure_psi": pressure,
        "flood_index": flood
    }])
    
    # Predict (Isolation Forest outputs 1 for normal, -1 for anomaly)
    prediction = model.predict(new_data)[0]
    
    if prediction == 1:
        return "✅ STATUS: Normal Operating Conditions"
    else:
        return "🚨 ALERT: Critical Anomaly Detected!"

# 2. Test the detector with sample metrics
sample_voltage = 220.5
sample_pressure = 45.2
sample_flood = 0.01

status = check_telemetry(sample_voltage, sample_pressure, sample_flood)
print(f"Testing metrics: Voltage={sample_voltage}V, Pressure={sample_pressure}psi, Flood={sample_flood}")
print(status)
