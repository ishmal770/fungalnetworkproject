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
    # approximates each grid box's real-world coordinates, assuming plot_lat/plot_lon
    # entered in the satellite section is the CENTER of the plot
    width_m = width_ft * 0.3048
    height_m = height_ft * 0.3048
    box_w, box_h = width_m / cols, height_m / rows
    x_offset = (col + 0.5) * box_w - width_m / 2
    y_offset = (rows - row - 0.5) * box_h - height_m / 2  # row 0 = top of grid
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

# only freeze the grid once you've actually captured something — before that,
# the sliders should update freely (this was the bug: it was freezing on page load instead)
if st.session_state.current_box == 0:
    st.session_state.locked_rows, st.session_state.locked_cols = rows, cols
elif (rows, cols) != (st.session_state.locked_rows, st.session_state.locked_cols):
    st.warning(f"Grid changed from {st.session_state.locked_rows}x{st.session_state.locked_cols} to {rows}x{cols} — "
               f"previously captured images no longer match. Clear scan_images/ or keep the grid size fixed.")

rows, cols = st.session_state.locked_rows, st.session_state.locked_cols
total_boxes = rows * cols
st.write(f"Scanning in {total_boxes} boxes ({cols} across, {rows} down).")

os.makedirs("scan_images", exist_ok=True)

current = st.session_state.current_box
if current < total_boxes:
    row, col = current // cols, current % cols
    captured_set = {(r, c) for r in range(rows) for c in range(cols) if os.path.exists(f"scan_images/box_{r}_{c}.jpg")}
    st.pyplot(draw_capture_grid(rows, cols, row, col, captured_set))
    st.subheader(f"Box {current + 1} of {total_boxes} — (row {row+1}, col {col+1})")
    st.info("Stand directly over this grid square, hold the camera straight down, take the photo.")
    uploaded = st.file_uploader(f"Upload photo for box {current+1}", key=f"upload_{current}")
    if uploaded is not None:
        save_path = f"scan_images/box_{row}_{col}.jpg"
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.image(save_path, caption=f"Saved: box ({row},{col})", width=200)
        st.success(f"Saved box ({row},{col}).")
        st.session_state.current_box += 1
        st.rerun()
else:
    st.success("All grid boxes photographed.")

# ============================================================
# 2. CONDUCTIVITY LOGGING
# ============================================================
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

# ============================================================
# 3. INFERENCE + HEATMAP + UNCERTAINTY
# ============================================================
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

try:
    from fungal_model import load_model, preprocess_image, mc_dropout_predict as real_mc_dropout_predict
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
    if use_real_model and "loaded_model" in st.session_state:
        tensor = preprocess_image(image_path)
        return real_mc_dropout_predict(st.session_state.loaded_model, tensor)
    return dummy_predict(image_path)

def build_photo_mosaic(rows, cols):
    # "before" image — stitches your actual captured photos into their grid positions,
    # so you can see literally what was analyzed, tile by tile
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
    # VARI (Visible Atmospherically Resistant Index) — an NDVI-style formula that only
    # needs ordinary RGB, no infrared. Per-pixel, so it gives a real continuous gradient
    # instead of one flat score per photo.
    img_array = img_array.astype(float) / 255.0
    R, G, B = img_array[..., 0], img_array[..., 1], img_array[..., 2]
    return (G - R) / (G + R - B + 1e-6)


def build_pixel_heatmap(rows, cols, tile_px=150):
    # stitches every captured photo into one canvas and colors it pixel-by-pixel by VARI —
    # same visual style as the satellite NDVI map, driven by real image data, no CNN needed
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
    ax.imshow(canvas, cmap="RdYlGn", vmin=-0.3, vmax=0.3)
    ax.axis("off")
    return fig


def build_heatmap(grid_results, rows, cols):
    import matplotlib.image as mpimg
    fig, ax = plt.subplots()
    cmap = plt.cm.RdYlGn_r
    for r in range(rows):
        for c in range(cols):
            result = grid_results.get((r, c))
            if result is None:
                continue
            color = cmap(result["mean"])
            confidence_alpha = max(0.2, 1 - result["std"])  # low confidence = more transparent tint

            path = f"scan_images/box_{r}_{c}.jpg"
            if os.path.exists(path):
                # show the real photo, then lay a translucent color tint on top —
                # this keeps actual texture visible instead of a flat solid block
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
        st.session_state.grid_results = grid_results  # needed by the sampling-sites section below
        st.session_state.grid_rows, st.session_state.grid_cols = rows, cols
        st.write("**Before / after — exactly what was analyzed:**")
        before_col, after_col = st.columns(2)
        with before_col:
            st.pyplot(build_photo_mosaic(rows, cols))
            st.caption("Before: your actual captured photos, in grid position")
        with after_col:
            st.pyplot(build_pixel_heatmap(rows, cols))
            st.caption("After: continuous per-pixel likelihood gradient (VARI-based, same style as the satellite NDVI map)")

# ============================================================
# 4. SATELLITE CONTEXT PANEL (plot-level, not per-tile)
# ============================================================
st.header("4. Environmental Context (Satellite)")
from satellite import init_ee, get_satellite_features, get_ndvi_map_url, get_truecolor_map_url, build_region

st.write("Enter the plot's coordinates (from your phone's Maps app — drop a pin on the plot and copy the lat/lon shown).")
sat_col1, sat_col2 = st.columns(2)
with sat_col1:
    plot_lat = st.number_input("Latitude", value=37.4275, format="%.6f")
with sat_col2:
    plot_lon = st.number_input("Longitude", value=-122.1697, format="%.6f")

satellite_radius_m = st.slider("Satellite context radius (meters)", 200, 2000, 500,
                                help="Real-world area shown around your plot. Sentinel-2 pixels are ~10m, "
                                     "so this needs to stay well above your plot's actual size to show real detail.")
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

    # clean table instead of a raw dict dump
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

# ============================================================
# 5. RECOMMENDED SAMPLING SITES — two-stage verification workflow
# ============================================================
def build_sampling_sites_map(rows, cols, labeled_sites):
    # labeled_sites: list of (row, col, label_number) — draws the actual photo grid
    # with a highlighted numbered box over each recommended sampling location
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


st.header("5. Recommended Sampling Sites")
st.write("High-risk tiles from the CNN heatmap, with an option to verify each one with the Arduino conductivity probe.")

RISK_THRESHOLD = 0.6  # tiles scoring above this get flagged as sampling candidates

if "sampling_sites" not in st.session_state:
    st.session_state.sampling_sites = {}  # {(r,c): {"sensor_status": ..., "conductivity_score": ...}}

if "grid_results" in st.session_state:
    gr = st.session_state.grid_results
    g_rows, g_cols = st.session_state.grid_rows, st.session_state.grid_cols

    high_risk = sorted(
        [(pos, res) for pos, res in gr.items() if res["mean"] >= RISK_THRESHOLD],
        key=lambda x: x[1]["mean"], reverse=True
    )

    if not high_risk:
        st.write(f"No tiles above the {int(RISK_THRESHOLD*100)}% risk threshold yet.")
    else:
        labeled = [(r, c, i) for i, ((r, c), res) in enumerate(high_risk, start=1)]
        st.pyplot(build_sampling_sites_map(g_rows, g_cols, labeled))
        st.caption("Red-numbered boxes mark exactly which physical grid square to go sample.")

        for i, ((r, c), res) in enumerate(high_risk, start=1):
            site_key = (r, c)
            if site_key not in st.session_state.sampling_sites:
                st.session_state.sampling_sites[site_key] = {"sensor_status": "Not sampled", "conductivity_score": None}
            site = st.session_state.sampling_sites[site_key]

            lat, lon = get_box_coordinates(plot_lat, plot_lon, r, c, g_rows, g_cols, plot_width_ft, plot_height_ft)
            priority = "High priority" if res["mean"] >= 0.8 else "Medium priority"

            with st.container(border=True):
                st.write(f"**Sampling Site #{i}**")
                st.write(f"CNN risk: {res['mean']*100:.0f}%  |  Coordinates: {lat:.4f}, {lon:.4f}")
                st.write(f"Sensor status: {site['sensor_status']}  |  Recommendation: {priority}")

                if st.button(f"Take Arduino reading — Site #{i}", key=f"sample_btn_{r}_{c}"):
                    try:
                        ser = serial.Serial(serial_port, 9600, timeout=1)
                        raw = read_average(ser)
                        ser.close()
                        if raw is not None:
                            score = normalize_score(raw)
                            site["conductivity_score"] = score
                            site["sensor_status"] = f"Sampled — conductivity {score:.1f}/10"

                            # cross-reference: do the two proxies agree?
                            conductivity_high = score >= 6
                            cnn_high = res["mean"] >= RISK_THRESHOLD
                            if conductivity_high and cnn_high:
                                st.success("Verified high risk — both CNN and conductivity agree. Strong candidate for real sampling.")
                            elif not conductivity_high and cnn_high:
                                st.warning("Discrepancy — CNN flagged this as high risk but conductivity reads low/dry. Worth a second look before prioritizing.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Serial read failed: {e}")
else:
    st.write("Run inference in section 3 first to populate sampling sites.")