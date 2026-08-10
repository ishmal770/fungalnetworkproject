import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import serial
import csv
import re
import json
import time
from datetime import datetime

st.set_page_config(page_title="EcoScout", layout="wide")
st.title("EcoScout — Fungal Activity Screening")

# ============================================================
# 0. CONDUCTIVITY CALIBRATION + SAMPLING (Saira's — used only from section 4)
# ============================================================
CALIBRATION_FILE = "calibration.json"
SITE_SAMPLE_LOG = "site_samples_log.csv"
SAMPLE_READINGS_N = 5  # sample = average of this many quick readings


def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            return json.load(f)
    return None


def parse_raw_line(line):
    # matches the Arduino's "Raw: 123" format (case-insensitive — sketches vary)
    match = re.search(r"raw:\s*(\d+)", line, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if line.strip().isdigit():
        return int(line.strip())
    return None


def score_and_level(raw_avg, cal):
    dry, wet, mid = cal["dry_raw"], cal["wet_raw"], cal["midpoint"]
    score = (raw_avg - dry) / (wet - dry) * 10
    score = max(0.0, min(10.0, score))
    level = "HIGH" if raw_avg >= mid else "LOW"
    return score, level


def sample_soil(serial_port, site_label, n=SAMPLE_READINGS_N, status_ph=None):
    """
    Opens the serial port, grabs n quick valid readings, logs each parsed
    raw value (timestamped, tagged with site_label) to SITE_SAMPLE_LOG,
    and returns the list of raw readings collected.
    """
    readings = []
    ser = serial.Serial(serial_port, 9600, timeout=1)
    time.sleep(1.5)  # Arduino resets when the port opens — give it a moment to reboot
    file_exists = os.path.exists(SITE_SAMPLE_LOG)
    f = open(SITE_SAMPLE_LOG, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=["timestamp", "site", "raw"])
    if not file_exists:
        writer.writeheader()
    try:
        ser.reset_input_buffer()
        attempts = 0
        while len(readings) < n and attempts < n * 10:
            attempts += 1
            line = ser.readline().decode("utf-8", errors="replace")
            raw = parse_raw_line(line)
            if raw is None:
                continue
            readings.append(raw)
            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "site": site_label,
                "raw": raw,
            })
            f.flush()
            if status_ph is not None:
                status_ph.write(f"Sampling {site_label}... {len(readings)}/{n} readings")
    finally:
        f.close()
        ser.close()
    if status_ph is not None:
        status_ph.write(f"Done — {len(readings)} readings logged for {site_label}")
    return readings


def run_sample_and_report(serial_port, site_label):
    """
    Runs a sample, averages it, compares to calibration.json's range/midpoint,
    and returns (avg_raw, score, level) or (None, None, None) on failure.
    Renders result directly to the Streamlit page.
    """
    cal = load_calibration()
    if cal is None:
        st.error(f"No {CALIBRATION_FILE} found. Run `python log_moisture.py --calibrate` first.")
        return None, None, None

    status_ph = st.empty()
    try:
        with st.spinner(f"Sampling {site_label}..."):
            readings = sample_soil(serial_port, site_label, status_ph=status_ph)
    except Exception as e:
        st.error(f"Serial read failed: {e}")
        return None, None, None

    if not readings:
        st.error("No valid readings received. Check the port and that the probe is connected.")
        return None, None, None

    avg_raw = sum(readings) / len(readings)
    score, level = score_and_level(avg_raw, cal)
    st.write(f"Range: {cal['dry_raw']}–{cal['wet_raw']}  |  Midpoint: {cal['midpoint']:.0f}  "
             f"|  Average raw: {avg_raw:.0f}")
    if level == "HIGH":
        st.success(f"HIGH conductivity ({score:.1f}/10) — moist, ion-rich soil, "
                    f"favorable conditions for a fungal network.")
    else:
        st.warning(f"LOW conductivity ({score:.1f}/10) — drier soil, lower fungal-network likelihood.")
    return avg_raw, score, level


# ============================================================
# 1. PLOT SETUP + GRID-GUIDED IMAGE CAPTURE
# ============================================================
def draw_capture_grid(rows, cols, current_row, current_col, captured_set):
    fig, ax = plt.subplots(figsize=(min(cols, 8), min(rows, 8)))
    for r in range(rows):
        for c in range(cols):
            if (r, c) in captured_set:
                color = "#8fd19e"       # green — already captured
            elif (r, c) == (current_row, current_col):
                color = "#ffe066"       # yellow — capture this one now
            else:
                color = "#e0e0e0"       # gray — not yet reached
            edge = "black" if (r, c) != (current_row, current_col) else "red"
            lw = 1 if (r, c) != (current_row, current_col) else 3
            ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, facecolor=color, edgecolor=edge, linewidth=lw))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def get_box_coordinates(center_lat, center_lon, row, col, rows, cols, width_ft, height_ft):
    width_m = width_ft * 0.3048
    height_m = height_ft * 0.3048
    box_w, box_h = width_m / cols, height_m / rows
    x_offset = (col + 0.5) * box_w - width_m / 2
    y_offset = (rows - row - 0.5) * box_h - height_m / 2
    lat_offset = y_offset / 111320
    lon_offset = x_offset / (111320 * math.cos(math.radians(center_lat)))
    return center_lat + lat_offset, center_lon + lon_offset


st.header("1. Plot Setup")
col1, col2 = st.columns(2)
with col1:
    plot_width_ft = st.number_input("Plot width (ft)", min_value=1, value=20)
with col2:
    plot_height_ft = st.number_input("Plot height (ft)", min_value=1, value=20)
grid_size_ft = st.slider("Grid box size (ft)", 2, 10, 5)

cols = math.ceil(plot_width_ft / grid_size_ft)
rows = math.ceil(plot_height_ft / grid_size_ft)

if "current_box" not in st.session_state:
    st.session_state.current_box = 0

if st.session_state.current_box == 0:
    st.session_state.locked_rows, st.session_state.locked_cols = rows, cols
elif (rows, cols) != (st.session_state.locked_rows, st.session_state.locked_cols):
    st.warning(f"Grid changed from {st.session_state.locked_rows}x{st.session_state.locked_cols} to {rows}x{cols} — "
               f"previously captured images no longer match. Clear scan_images/ or keep the grid size fixed.")

rows, cols = st.session_state.locked_rows, st.session_state.locked_cols
total_boxes = rows * cols
st.write(f"Scanning in {total_boxes} boxes ({cols} across, {rows} down).")

os.makedirs("scan_images", exist_ok=True)

if "box_coords" not in st.session_state:
    st.session_state.box_coords = {}  # {(row, col): (lat, lon)} — real GPS per photo, when provided

current = st.session_state.current_box
if current < total_boxes:
    row, col = current // cols, current % cols
    captured_set = {(r, c) for r in range(rows) for c in range(cols) if os.path.exists(f"scan_images/box_{r}_{c}.jpg")}
    st.pyplot(draw_capture_grid(rows, cols, row, col, captured_set))
    st.subheader(f"Box {current + 1} of {total_boxes} — (row {row+1}, col {col+1})")
    st.info("Stand directly over this grid square, hold the camera straight down, take the photo.")

    with_gps = st.checkbox("I have the GPS coordinates for this exact spot (from Maps)", key=f"has_gps_{current}")
    if with_gps:
        gps_col1, gps_col2 = st.columns(2)
        with gps_col1:
            box_lat = st.number_input("Latitude", value=37.4275, format="%.6f", key=f"lat_{current}")
        with gps_col2:
            box_lon = st.number_input("Longitude", value=-122.1697, format="%.6f", key=f"lon_{current}")

    uploaded = st.file_uploader(f"Upload photo for box {current+1}", key=f"upload_{current}")
    if uploaded is not None:
        save_path = f"scan_images/box_{row}_{col}.jpg"
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        if with_gps:
            st.session_state.box_coords[(row, col)] = (box_lat, box_lon)
        st.image(save_path, caption=f"Saved: box ({row},{col})", width=200)
        st.success(f"Saved box ({row},{col}).")
        st.session_state.current_box += 1
        st.rerun()
else:
    st.success("All grid boxes photographed.")

# ============================================================
# 2. INFERENCE + HEATMAP + SKY GUARD
# ============================================================
st.header("2. Fungal Activity Heatmap")

def looks_like_sky(image_path, brightness_thresh=200, sat_thresh=40, edge_thresh=8):
    """Fast, non-ML safety net: sky / blown-out frames are bright, flat, and
    low-texture (or uniformly blue) in a way real ground/fungi photos aren't.
    Catches this failure mode instantly without retraining the CNN."""
    from PIL import Image, ImageStat
    img = Image.open(image_path).convert("RGB").resize((100, 100))
    stat = ImageStat.Stat(img)
    r, g, b = stat.mean
    brightness = (r + g + b) / 3
    saturation = max(r, g, b) - min(r, g, b)
    texture = sum(stat.stddev) / 3
    is_bright_flat = brightness > brightness_thresh and texture < edge_thresh
    is_blue_dominant = b > r + 15 and b > g + 5
    return is_bright_flat or (is_blue_dominant and saturation < sat_thresh)


def dummy_predict(image):
    # stub for testing the UI before Isha's checkpoint exists
    return {"mean": np.random.uniform(0, 1), "std": np.random.uniform(0, 0.3),
            "class": "none", "abstain": False}

try:
    from fungal_model import load_model, predict_image
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

use_real_model = st.checkbox("Use trained model (fungal_model.pth) instead of random stub",
                              value=False, disabled=not MODEL_AVAILABLE)
if use_real_model and "loaded_model" not in st.session_state:
    try:
        st.session_state.loaded_model = load_model()
        st.success("Model checkpoint loaded.")
    except Exception as e:
        st.error(f"Couldn't load fungal_model.pth: {e}")
        use_real_model = False

def run_predict(image_path):
    # Rule-based sky/blown-out guard runs first, regardless of which model path
    # is active below — this is what stops sky from being scored as fungal.
    if looks_like_sky(image_path):
        return {"mean": 0.0, "std": 0.0, "class": "background", "abstain": True}
    if use_real_model and "loaded_model" in st.session_state:
        # tiled: a grid photo covers ~5ft, so a fruiting body is a small fraction
        # of the frame. Scoring overlapping crops keeps it at trained scale.
        return predict_image(st.session_state.loaded_model, image_path, min_confidence=0.65)
    return dummy_predict(image_path)

def build_photo_mosaic(rows, cols):
    import matplotlib.image as mpimg
    fig, ax = plt.subplots()
    for r in range(rows):
        for c in range(cols):
            path = f"scan_images/box_{r}_{c}.jpg"
            if os.path.exists(path):
                img = mpimg.imread(path)
                ax.imshow(img, extent=(c, c + 1, rows - r - 1, rows - r))
            else:
                ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, facecolor="lightgray", edgecolor="black"))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def compute_vari(img_array):
    img_array = img_array.astype(float) / 255.0
    R, G, B = img_array[..., 0], img_array[..., 1], img_array[..., 2]
    return (G - R) / (G + R - B + 1e-6)


def build_pixel_heatmap(rows, cols, tile_px=150):
    from PIL import Image
    canvas = np.full((rows * tile_px, cols * tile_px), np.nan)
    for r in range(rows):
        for c in range(cols):
            path = f"scan_images/box_{r}_{c}.jpg"
            if os.path.exists(path):
                img = Image.open(path).convert("RGB").resize((tile_px, tile_px))
                vari = compute_vari(np.array(img))
                canvas[r*tile_px:(r+1)*tile_px, c*tile_px:(c+1)*tile_px] = vari

    fig, ax = plt.subplots()
    im = ax.imshow(canvas, cmap="RdYlGn", vmin=-0.3, vmax=0.3)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("VARI (greenness index)")
    cbar.set_ticks([-0.3, 0, 0.3])
    cbar.set_ticklabels(["Stressed/bare", "Neutral", "Healthy/green"])
    fig.tight_layout()
    return fig


def build_heatmap(grid_results, rows, cols):
    import matplotlib.image as mpimg
    from matplotlib.patches import Patch
    fig, ax = plt.subplots()
    cmap = plt.cm.RdYlGn_r
    for r in range(rows):
        for c in range(cols):
            result = grid_results.get((r, c))
            if result is None:
                continue
            if result.get("abstain"):
                color, confidence_alpha = (0.6, 0.6, 0.6, 1.0), 0.5
            else:
                color = cmap(result["mean"])
                confidence_alpha = max(0.2, 1 - result["std"])

            path = f"scan_images/box_{r}_{c}.jpg"
            if os.path.exists(path):
                img = mpimg.imread(path)
                ax.imshow(img, extent=(c, c + 1, rows - r - 1, rows - r))
                ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, facecolor=color,
                                            alpha=confidence_alpha * 0.55, edgecolor="black"))
            else:
                ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1,
                                            facecolor=color, alpha=confidence_alpha, edgecolor="black"))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    # legend: colorbar for the continuous risk gradient, plus a swatch for the
    # one thing a colorbar can't show — "abstained" is a separate category, not
    # a point on the risk scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fungal risk (CNN)")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    abstain_patch = Patch(facecolor=(0.6, 0.6, 0.6, 1.0), edgecolor="black", label="Abstained (unclear/sky)")
    ax.legend(handles=[abstain_patch], loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.tight_layout()
    return fig

if st.button("Run inference on captured boxes"):
    grid_results = {}
    for r in range(rows):
        for c in range(cols):
            path = f"scan_images/box_{r}_{c}.jpg"
            if os.path.exists(path):
                grid_results[(r, c)] = run_predict(path)
    if not grid_results:
        st.error(f"No images found in scan_images/ matching the current {rows}x{cols} grid. "
                 f"Check the folder actually has files named box_R_C.jpg, and that the grid size hasn't changed since capture.")
    else:
        st.session_state.grid_results = grid_results
        st.write(f"Found {len(grid_results)} of {rows*cols} boxes captured.")
        st.session_state.grid_rows, st.session_state.grid_cols = rows, cols

        st.write("**Model risk heatmap — CNN prediction per box:**")
        st.pyplot(build_heatmap(grid_results, rows, cols))
        st.caption("Green = low risk, red = high risk, grey = abstained (no confident read)")

        st.write("**Before / after — exactly what was analyzed:**")
        before_col, after_col = st.columns(2)
        with before_col:
            st.pyplot(build_photo_mosaic(rows, cols))
            st.caption("Before: your actual captured photos, in grid position")
        with after_col:
            st.pyplot(build_pixel_heatmap(rows, cols))
            st.caption("After: continuous per-pixel likelihood gradient (VARI-based, same style as the satellite NDVI map)")

# ============================================================
# 3. SATELLITE CONTEXT PANEL (plot-level, not per-tile)
# ============================================================
st.header("3. Environmental Context (Satellite)")
try:
    from satellite import init_ee, get_satellite_features, get_ndvi_map_url, get_truecolor_map_url, build_region
    SATELLITE_AVAILABLE = True
except ImportError:
    SATELLITE_AVAILABLE = False
    st.info("Satellite module unavailable (missing the `ee` package). This section is optional and doesn't "
            "affect sections 1, 2, or 4. Run `python3 -m pip install earthengine-api` to enable it.")

if SATELLITE_AVAILABLE:
    st.write("Enter the plot's coordinates (from your phone's Maps app — drop a pin on the plot and copy the lat/lon shown).")
    sat_col1, sat_col2 = st.columns(2)
    with sat_col1:
        plot_lat = st.number_input("Latitude", value=37.4275, format="%.6f")
    with sat_col2:
        plot_lon = st.number_input("Longitude", value=-122.1697, format="%.6f")

    satellite_radius_m = st.slider("Satellite context radius (meters)", 200, 2000, 500)
    width_m = height_m = satellite_radius_m * 2
    ee_project = st.text_input("Earth Engine project ID", value="")

    if st.button("Fetch satellite data for this plot"):
        try:
            init_ee(project=ee_project if ee_project else None)
            result = get_satellite_features(plot_lat, plot_lon, plot_id="current_plot")
            result["ndvi_map_url"] = get_ndvi_map_url(plot_lat, plot_lon, width_m, height_m)
            result["truecolor_map_url"] = get_truecolor_map_url(plot_lat, plot_lon, width_m, height_m)
            st.session_state.sat_result = result
            st.success("Satellite data fetched.")
        except Exception as e:
            st.error(f"Earth Engine call failed: {e}")

    if "sat_result" in st.session_state:
        r = st.session_state.sat_result
        metrics_df = pd.DataFrame([
            {"Metric": "NDVI (vegetation health)", "Value": f"{r['ndvi']:.3f}" if r["ndvi"] is not None else "N/A"},
            {"Metric": "Soil moisture", "Value": f"{r['soil_moisture']:.3f}" if r["soil_moisture"] is not None else "N/A"},
            {"Metric": "Land surface temp (K)", "Value": f"{r['land_surface_temp_k']:.1f}" if r["land_surface_temp_k"] is not None else "N/A"},
        ])
        st.table(metrics_df)
        st.write("**Before / after — the exact area being analyzed:**")
        before_col, after_col = st.columns(2)
        with before_col:
            st.image(r["truecolor_map_url"], caption="Before: true-color satellite photo")
        with after_col:
            st.image(r["ndvi_map_url"], caption="After: NDVI (green = healthy, red = stressed)")
        st.caption("Plot-level, ~10m resolution — shown as context, not blended into the tile heatmap.")
else:
    # fall back to plain number inputs so section 4's coordinate math still works
    sat_col1, sat_col2 = st.columns(2)
    with sat_col1:
        plot_lat = st.number_input("Latitude", value=37.4275, format="%.6f")
    with sat_col2:
        plot_lon = st.number_input("Longitude", value=-122.1697, format="%.6f")

# ============================================================
# 4. RED-ZONE ALERT + SAMPLING PROMPT — CNN flags risk, Arduino verifies on demand
# ============================================================
def build_sampling_sites_map(rows, cols, labeled_sites):
    import matplotlib.image as mpimg
    fig, ax = plt.subplots()
    for r in range(rows):
        for c in range(cols):
            path = f"scan_images/box_{r}_{c}.jpg"
            if os.path.exists(path):
                img = mpimg.imread(path)
                ax.imshow(img, extent=(c, c + 1, rows - r - 1, rows - r))
            else:
                ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, facecolor="lightgray", edgecolor="black"))

    for (r, c, num) in labeled_sites:
        ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, facecolor="none", edgecolor="red", linewidth=4))
        ax.text(c + 0.5, rows - r - 0.5, f"#{num}", color="white", fontsize=14, fontweight="bold",
                ha="center", va="center", bbox=dict(facecolor="red", alpha=0.85, boxstyle="circle"))

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


st.header("4. Recommended Sampling Sites")
st.write("Whenever the CNN flags a red (high-risk) tile, you'll be prompted here to go test it with the probe. "
         "There's no general-purpose conductivity reader elsewhere in the app — sampling only happens here.")

RISK_THRESHOLD = 0.6  # tiles scoring above this get flagged red / sampling candidates

try:
    from serial.tools import list_ports
    available_ports = [p.device for p in list_ports.comports()]
    # On macOS, prefer /dev/cu.* over /dev/tty.* for the same physical device.
    # tty.* waits for a carrier-detect signal and can hang or silently drop
    # data in a way cu.* doesn't — the classic "works in the Serial Monitor
    # but not here" symptom on Mac. Sort cu.* first so it's the default pick.
    available_ports.sort(key=lambda p: (0 if "/cu." in p else 1, p))
except Exception:
    available_ports = []

if available_ports:
    serial_port = st.selectbox("Arduino serial port", available_ports)
else:
    st.warning("No serial ports detected — is the Arduino plugged in? Falling back to manual entry.")
    serial_port = st.text_input("Arduino serial port", value="/dev/cu.usbmodem1101")

st.caption("⚠️ Only one program can read a serial port at a time. Close the Arduino IDE's Serial Monitor, "
           "VS Code's serial extension, and any running `log_moisture.py` session before sampling here — "
           "otherwise this app can silently get zero readings while the other program keeps working fine.")

with st.expander("🔧 Debug: show raw serial output (use this if samples come back with 0 readings)"):
    if st.button("Read raw lines for 5 seconds"):
        try:
            ser = serial.Serial(serial_port, 9600, timeout=1)
            time.sleep(1.5)
            lines = []
            start = time.time()
            while time.time() - start < 5:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if line:
                    lines.append(line)
            ser.close()
            if not lines:
                st.error("Nothing received in 5 seconds — check the port, that nothing else has it open, "
                          "and that the sketch is actually looping and printing.")
            else:
                st.code("\n".join(lines))
                if not any(parse_raw_line(l) is not None for l in lines):
                    st.warning("Got data, but none of it matched the expected 'Raw: <number>' format — "
                               "compare the lines above to what parse_raw_line() is looking for.")
                else:
                    st.success("At least one line parsed successfully — sampling should work now.")
        except Exception as e:
            st.error(f"Couldn't open port: {e}")

cal_preview = load_calibration()
if cal_preview:
    st.caption(f"Using calibration.json — range {cal_preview['dry_raw']}–{cal_preview['wet_raw']}, "
               f"midpoint {cal_preview['midpoint']:.0f} (raw ≥ midpoint = HIGH)")
else:
    st.caption("No calibration.json found — run `python log_moisture.py --calibrate` before sampling.")

if "sampling_sites" not in st.session_state:
    st.session_state.sampling_sites = {}  # {(r,c): {"sensor_status": ..., "conductivity_score": ..., "level": ...}}

if "grid_results" in st.session_state:
    gr = st.session_state.grid_results
    g_rows, g_cols = st.session_state.grid_rows, st.session_state.grid_cols

    # abstained tiles (sky, unclear frames) are excluded outright — the model
    # declined to assess them, so they must not become sampling candidates
    high_risk = sorted(
        [(pos, res) for pos, res in gr.items()
         if not res.get("abstain") and res["mean"] >= RISK_THRESHOLD],
        key=lambda x: x[1]["mean"], reverse=True
    )
    n_abstained = sum(1 for res in gr.values() if res.get("abstain"))
    if n_abstained:
        st.caption(f"{n_abstained} tile(s) read as bare ground / sky / unclear — shown grey, not scored.")

    if not high_risk:
        st.write(f"No red zones — nothing scored above the {int(RISK_THRESHOLD*100)}% risk threshold.")
    else:
        n_unsampled = sum(
            1 for (r, c), _ in high_risk
            if st.session_state.sampling_sites.get((r, c), {}).get("sensor_status", "Not sampled") == "Not sampled"
        )
        if n_unsampled > 0:
            st.error(f"🔴 Red zone detected: {n_unsampled} area(s) flagged high-risk by the CNN. "
                     f"Go to each numbered site below and test the soil with the probe.")
        else:
            st.success("All red zones have been probe-tested.")

        labeled = [(r, c, i) for i, ((r, c), res) in enumerate(high_risk, start=1)]
        st.pyplot(build_sampling_sites_map(g_rows, g_cols, labeled))
        st.caption("🔴 Red-numbered boxes = high-risk sites flagged by the CNN — go test these with the probe.")

        for i, ((r, c), res) in enumerate(high_risk, start=1):
            site_key = (r, c)
            if site_key not in st.session_state.sampling_sites:
                st.session_state.sampling_sites[site_key] = {"sensor_status": "Not sampled",
                                                               "conductivity_score": None, "level": None}
            site = st.session_state.sampling_sites[site_key]

            if (r, c) in st.session_state.box_coords:
                lat, lon = st.session_state.box_coords[(r, c)]
                coord_source = "GPS entered at capture"
            else:
                lat, lon = get_box_coordinates(plot_lat, plot_lon, r, c, g_rows, g_cols, plot_width_ft, plot_height_ft)
                coord_source = "estimated from plot center"
            priority = "High priority" if res["mean"] >= 0.8 else "Medium priority"

            with st.container(border=True):
                if site["sensor_status"] == "Not sampled":
                    st.markdown(f"### 🔴 Site #{i} — needs testing")
                else:
                    st.markdown(f"### ✅ Site #{i} — tested")
                st.write(f"CNN risk: {res['mean']*100:.0f}% ± {res.get('std', 0)*100:.0f}%  "
                         f"(class: {res.get('class', 'n/a')})  |  Coordinates: {lat:.4f}, {lon:.4f} ({coord_source})")
                st.write(f"Sensor status: {site['sensor_status']}  |  Recommendation: {priority}")

                if st.button(f"Take Sample — Site #{i}", key=f"sample_btn_{r}_{c}"):
                    avg_raw, score, level = run_sample_and_report(
                        serial_port, site_label=f"site_{i}_r{r}_c{c}"
                    )
                    if avg_raw is not None:
                        site["conductivity_score"] = score
                        site["level"] = level
                        site["sensor_status"] = f"Sampled — {level} conductivity ({score:.1f}/10)"

                        conductivity_high = level == "HIGH"
                        cnn_high = res["mean"] >= RISK_THRESHOLD
                        if conductivity_high and cnn_high:
                            st.success("Verified high risk — both CNN and conductivity agree. Strong candidate for real sampling.")
                        elif not conductivity_high and cnn_high:
                            st.warning("Discrepancy — CNN flagged this as high risk but conductivity reads LOW/dry. Worth a second look before prioritizing.")
                        st.rerun()
else:
    st.write("Run inference in section 2 first — any red zones will trigger a testing prompt here.")
