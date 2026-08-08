import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import serial
import csv

st.set_page_config(page_title="EcoScout", layout="wide")
st.title("EcoScout — Fungal Activity Screening")

st.header("1. Plot Setup")
col1, col2 = st.columns(2)
with col1:
    plot_width_ft = st.number_input("Plot width (ft)", min_value=1, value=20)
with col2:
    plot_height_ft = st.number_input("Plot height (ft)", min_value=1, value=20)
grid_size_ft = st.slider("Grid box size (ft)", 2, 10, 5)

cols = math.ceil(plot_width_ft / grid_size_ft)
rows = math.ceil(plot_height_ft / grid_size_ft)
total_boxes = cols * rows
st.write(f"Scanning in {total_boxes} boxes ({cols} across, {rows} down).")

os.makedirs("scan_images", exist_ok=True)

if "current_box" not in st.session_state:
    st.session_state.current_box = 0

current = st.session_state.current_box
if current < total_boxes:
    row, col = current // cols, current % cols
    st.subheader(f"Box {current + 1} of {total_boxes} — (row {row+1}, col {col+1})")
    st.info("Stand directly over this grid square, hold the camera straight down, take the photo.")
    uploaded = st.file_uploader(f"Upload photo for box {current+1}", key=f"upload_{current}")
    if uploaded is not None:
        save_path = f"scan_images/box_{row}_{col}.jpg"
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved box ({row},{col}).")
        st.session_state.current_box += 1
        st.rerun()
else:
    st.success("All grid boxes photographed.")

st.header("2. Conductivity Readings")

DRY_RAW = 200
WET_RAW = 900

def normalize_score(raw):
    score = (raw - DRY_RAW) / (WET_RAW - DRY_RAW) * 10
    return max(0, min(10, score))

def read_average(ser, n=5):
    readings = []
    for _ in range(n):
        line = ser.readline().decode().strip()
        if line.isdigit():
            readings.append(int(line))
    return sum(readings) / len(readings) if readings else None

serial_port = st.text_input("Arduino serial port", value="/dev/tty.usbmodem1101")
if st.button("Read conductivity for current box"):
    try:
        ser = serial.Serial(serial_port, 9600, timeout=1)
        raw = read_average(ser)
        ser.close()
        if raw is not None:
            score = normalize_score(raw)
            row, col = current // cols, current % cols
            file_exists = os.path.exists("conductivity_log.csv")
            with open("conductivity_log.csv", "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["row", "col", "conductivity_score"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({"row": row, "col": col, "conductivity_score": score})
            st.success(f"Logged conductivity {score:.1f} for box ({row},{col})")
    except Exception as e:
        st.error(f"Serial read failed: {e}")

st.header("3. Fungal Activity Heatmap")

def mc_dropout_predict(model, image, n_passes=25):
    model.train()
    preds = [model(image).item() for _ in range(n_passes)]
    mean = float(np.mean(preds))
    std = float(np.std(preds))
    return {"mean": mean, "std": std}

def dummy_predict(image):
    # stub for testing the UI before Isha's checkpoint exists
    return {"mean": np.random.uniform(0, 1), "std": np.random.uniform(0, 0.3)}

def build_heatmap(grid_results, rows, cols):
    fig, ax = plt.subplots()
    cmap = plt.cm.RdYlGn_r
    for r in range(rows):
        for c in range(cols):
            result = grid_results.get((r, c))
            if result is None:
                continue
            color = cmap(result["mean"])
            alpha = max(0.2, 1 - result["std"])
            ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, facecolor=color, alpha=alpha, edgecolor="black"))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig

if st.button("Run inference on captured boxes"):
    grid_results = {}
    for r in range(rows):
        for c in range(cols):
            path = f"scan_images/box_{r}_{c}.jpg"
            if os.path.exists(path):
                # swap dummy_predict for mc_dropout_predict(model, load_image(path)) once model exists
                grid_results[(r, c)] = dummy_predict(path)
    fig = build_heatmap(grid_results, rows, cols)
    st.pyplot(fig)
    st.caption("Darker/redder = higher estimated fungal-network likelihood. Faded = lower confidence.")


st.header("4. Environmental Context (Satellite)")
if os.path.exists("satellite_features.csv"):
    sat_df = pd.read_csv("satellite_features.csv")
    st.dataframe(sat_df)
    st.caption("Plot-level NDVI / soil moisture / land surface temp — ~10m resolution, shown as context, not blended into the tile heatmap.")
else:
    st.write("Run satellite_features.py first to populate this panel.")