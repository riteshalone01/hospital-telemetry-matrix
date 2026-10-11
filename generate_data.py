"""Generate mock hospital telemetry data.

Columns:
  timestamp           - 1 reading per 5 minutes
  hospital_id         - H01..H05
  voltage_v           - mains voltage (nominal ~230 V)
  water_pressure_psi  - supply pressure (nominal ~60 psi)
  flood_index         - satellite flood index, 0 (dry) .. 1 (flooded)
  is_anomaly          - ground-truth label (1 = injected anomaly), for evaluation only
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_HOSPITALS = 5
DAYS = 30
STEPS_PER_DAY = 24 * 12  # 5-minute readings
N = DAYS * STEPS_PER_DAY

frames = []
for h in range(1, N_HOSPITALS + 1):
    ts = pd.date_range("2026-09-01", periods=N, freq="5min")
    hours = np.asarray(ts.hour + ts.minute / 60)

    # Normal behaviour: small daily load cycle + noise
    voltage = 230 + 2.0 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 1.5, N)
    pressure = 60 + 3.0 * np.sin(2 * np.pi * (hours - 6) / 24) + rng.normal(0, 1.2, N)
    flood = np.clip(0.08 + rng.normal(0, 0.03, N), 0, 1)
    label = np.zeros(N, dtype=int)

    # Inject ~4% anomalies in short bursts of different types
    n_events = int(N * 0.04 / 8)
    starts = rng.choice(N - 20, n_events, replace=False)
    for s in starts:
        length = rng.integers(4, 13)
        e = s + length
        kind = rng.choice(["brownout", "surge", "pipe_burst", "flood", "combined"])
        if kind == "brownout":
            voltage[s:e] -= rng.uniform(25, 50)
        elif kind == "surge":
            voltage[s:e] += rng.uniform(20, 40)
        elif kind == "pipe_burst":
            pressure[s:e] -= rng.uniform(20, 35)
        elif kind == "flood":
            flood[s:e] = np.clip(flood[s:e] + rng.uniform(0.4, 0.8), 0, 1)
        else:  # flood causing power + pressure problems
            flood[s:e] = np.clip(flood[s:e] + rng.uniform(0.5, 0.9), 0, 1)
            voltage[s:e] -= rng.uniform(15, 35)
            pressure[s:e] -= rng.uniform(10, 25)
        label[s:e] = 1

    frames.append(pd.DataFrame({
        "timestamp": ts,
        "hospital_id": f"H{h:02d}",
        "voltage_v": voltage.round(2),
        "water_pressure_psi": pressure.round(2),
        "flood_index": flood.round(3),
        "is_anomaly": label,
    }))

df = pd.concat(frames, ignore_index=True)
df.to_csv("hospital_telemetry.csv", index=False)
print(f"Wrote hospital_telemetry.csv: {len(df)} rows, anomaly rate {df.is_anomaly.mean():.2%}")
