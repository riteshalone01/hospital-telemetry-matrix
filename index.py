import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Load Model
# ---------------------------------------------------------
try:
    model_data = joblib.load("model.pkl")
    model = model_data.get("model", model_data) if isinstance(model_data, dict) else model_data
except FileNotFoundError:
    st.error("🚨 Error: 'model.pkl' was not found in the app folder.")
    st.stop()

# ---------------------------------------------------------
# 2. Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(page_title="Hospital Anomaly Monitoring Dashboard", layout="wide", initial_sidebar_state="expanded")

# Dark theme overrides
st.markdown("""
    <style>
    .stApp { background-color: #060913; color: #f1f5f9; }
    [data-testid="stSidebar"] { background-color: #0a0f24 !important; border-right: 1px solid #1e293b; }

    .telemetry-box {
        background: linear-gradient(135deg, #0d1527 0%, #070a14 100%);
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #1e293b;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        text-align: center;
    }
    .metric-title { font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: #64748b; font-weight: bold; }
    .metric-display { font-size: 36px; font-weight: 800; font-family: 'Consolas', monospace; margin: 8px 0; }

    .status-panel {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-weight: 800;
        font-size: 22px;
        letter-spacing: 1px;
        margin-top: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .log-terminal {
        background-color: #030712;
        border: 1px solid #1e293b;
        font-family: 'Consolas', monospace;
        padding: 15px;
        border-radius: 8px;
        height: 380px;
        overflow-y: auto;
        color: #38bdf8;
        font-size: 13px;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# Page header
st.markdown("<h1 style='color:#00f2fe; font-weight:900; letter-spacing:-0.5px; margin-bottom:0;'>🏥 HOSPITAL INFRASTRUCTURE MONITORING DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#475569; font-size:14px; margin-bottom:25px;'>Live telemetry simulation with machine-learning anomaly detection for hospital power, water, and flood sensors.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Session State
# ---------------------------------------------------------
if 'history_list' not in st.session_state:
    st.session_state.history_list = []
if 'console_logs' not in st.session_state:
    st.session_state.console_logs = [f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring started."]

# ---------------------------------------------------------
# 4. Sidebar Controls
# ---------------------------------------------------------
LIVE_MODE = "⚡ Live Streaming"
MANUAL_MODE = "📊 Manual Input"

st.sidebar.markdown("<h3 style='color:#00f2fe;'>🕹️ CONTROLS</h3>", unsafe_allow_html=True)
operation_mode = st.sidebar.selectbox("Simulation Mode", [LIVE_MODE, MANUAL_MODE])

if operation_mode == MANUAL_MODE:
    voltage = st.sidebar.slider("Line Voltage (V)", 160.0, 280.0, 220.0, 0.5)
    pressure = st.sidebar.slider("Water Pressure (PSI)", 15.0, 95.0, 45.0, 0.5)
    flood = st.sidebar.slider("Flood Index", 0.0, 1.0, 0.0, 0.01)
    refresh_rate = 1.0
else:
    refresh_rate = st.sidebar.select_slider("Refresh Rate (Hz)", options=[0.2, 0.5, 1.0, 2.0], value=0.5)

    # Simulate gradual drift in live equipment readings
    seed_time = time.time()
    voltage = round(float(220.0 + np.sin(seed_time / 6) * 15 + np.random.normal(0, 2)), 1)
    pressure = round(float(45.0 + np.cos(seed_time / 4) * 8 + np.random.normal(0, 1.2)), 1)

    # Inject an occasional anomaly (roughly 8% of readings)
    if np.random.rand() > 0.92:
        flood = round(float(0.65 + np.random.rand() * 0.3), 2)
        voltage = round(voltage - np.random.uniform(30, 50), 1)
    else:
        flood = round(float(0.01 + np.random.rand() * 0.02), 2)

# ---------------------------------------------------------
# 5. Anomaly Detection
# ---------------------------------------------------------
# Build the input row with the same column names used in training
input_df = pd.DataFrame([{"voltage_v": voltage, "water_pressure_psi": pressure, "flood_index": flood}])
prediction = model.predict(input_df)[0]  # 1 = normal, -1 = anomaly

current_time_str = datetime.now().strftime('%H:%M:%S')

# Keep a rolling history of the last 30 readings for the chart
st.session_state.history_list.append({
    'Timestamp': current_time_str,
    'Voltage': float(voltage),
    'Pressure': float(pressure),
    'Flood': float(flood)
})
if len(st.session_state.history_list) > 30:
    st.session_state.history_list.pop(0)

# Rolling event log
if len(st.session_state.console_logs) > 20:
    st.session_state.console_logs.pop(1)

if prediction == 1:
    st.session_state.console_logs.append(f"[{current_time_str}] 🟩 NORMAL: Voltage={voltage}V | Pressure={pressure}PSI | Flood={flood}")
else:
    st.session_state.console_logs.append(f"[{current_time_str}] 🟥 ANOMALY ALERT: Abnormal sensor readings detected!")

# ---------------------------------------------------------
# 6. Live Metric Cards
# ---------------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    v_color = "#ff0055" if voltage > 245 or voltage < 195 else "#00f2fe"
    st.markdown(f'<div class="telemetry-box" style="border-top: 3px solid {v_color};"><div class="metric-title">⚡ Line Voltage</div><div class="metric-display" style="color: {v_color};">{voltage} V</div></div>', unsafe_allow_html=True)
with c2:
    p_color = "#ff0055" if pressure > 65 or pressure < 25 else "#00f2fe"
    st.markdown(f'<div class="telemetry-box" style="border-top: 3px solid {p_color};"><div class="metric-title">💧 Water Pressure</div><div class="metric-display" style="color: {p_color};">{pressure} PSI</div></div>', unsafe_allow_html=True)
with c3:
    f_color = "#ff0055" if flood > 0.2 else "#00f2fe"
    st.markdown(f'<div class="telemetry-box" style="border-top: 3px solid {f_color};"><div class="metric-title">🌊 Flood Index</div><div class="metric-display" style="color: {f_color};">{flood} idx</div></div>', unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# 7. System Status Banner
# ---------------------------------------------------------
if prediction == 1:
    st.markdown('<div class="status-panel" style="background: linear-gradient(135deg, #064e3b 0%, #022c22 100%); color: #34d399; border: 1px solid #059669;">🛡️ SYSTEM STATUS: NORMAL</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-panel" style="background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%); color: #f87171; border: 1px solid #dc2626;">🚨 ALERT: ANOMALY DETECTED</div>', unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# 8. Live Chart & Event Log
# ---------------------------------------------------------
left_panel, right_panel = st.columns(2)

with left_panel:
    st.markdown("<h4 style='color: #64748b; font-size:13px; text-transform:uppercase;'>📊 Live Sensor Readings</h4>", unsafe_allow_html=True)

    df_plot = pd.DataFrame(st.session_state.history_list)

    fig = go.Figure()
    if not df_plot.empty:
        fig.add_trace(go.Scatter(x=df_plot['Timestamp'], y=df_plot['Voltage'], mode='lines+markers', name='Voltage (V)', line=dict(color='#00f2fe', width=3)))
        fig.add_trace(go.Scatter(x=df_plot['Timestamp'], y=df_plot['Pressure'], mode='lines+markers', name='Pressure (PSI)', line=dict(color='#a855f7', width=3)))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13, 21, 39, 0.4)',
        margin=dict(l=20, r=20, t=10, b=10), height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#f1f5f9")),
        xaxis=dict(showgrid=True, gridcolor='#1e293b', tickfont=dict(color='#64748b')),
        yaxis=dict(showgrid=True, gridcolor='#1e293b', tickfont=dict(color='#64748b'))
    )
    st.plotly_chart(fig, use_container_width=True)

with right_panel:
    st.markdown("<h4 style='color: #64748b; font-size:13px; text-transform:uppercase;'>📟 Event Log</h4>", unsafe_allow_html=True)
    log_content = "<br>".join(st.session_state.console_logs[::-1])
    st.markdown(f'<div class="log-terminal">{log_content}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 9. Auto-Refresh (Live Mode Only)
# ---------------------------------------------------------
if operation_mode == LIVE_MODE:
    time.sleep(refresh_rate)
    st.rerun()
