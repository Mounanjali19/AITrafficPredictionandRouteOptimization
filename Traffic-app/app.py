# app.py (corrected)

import os
import json
import torch
import numpy as np
import random
import pandas as pd                         # <<--- added
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
import folium
from datetime import datetime

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------
# UTILS
# ------------------------------------------------
from utils.preprocess import generate_traffic_sequence
from utils.graph_utils import build_edge_index
from utils.hybrid_model import HybridGAT_LSTM

# ------------------------------------------------
# MODEL PATHS
# ------------------------------------------------
HYBRID_PATH = os.path.join(BASE_DIR, "models", "hybrid_model_final.pt")
PPO_PATH = os.path.join(BASE_DIR, "models", "ppo_traffic_agent_fixed.pt")
YOLO_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
EDGE_FILE = os.path.join(BASE_DIR, "data", "ecity_edges.csv")

device = "cpu"

# -------------------------
# Globals to hold last hybrid output
# -------------------------
last_hybrid_prediction = None     # numpy array shape (N,)
last_hybrid_meta = None           # dict with keys: date, time, scenario, timestamp

# ======================================================
# 1. LOAD HYBRID MODEL
# ======================================================
try:
    hybrid_model = HybridGAT_LSTM(
        in_dim=8,
        gat_hidden=64,
        gat_heads=4,
        lstm_hidden=128,
        fusion_hidden=128
    ).to(device)

    hybrid_model.load_state_dict(torch.load(HYBRID_PATH, map_location=device), strict=True)
    hybrid_model.eval()
    print("✅ Hybrid GAT-LSTM model loaded.")
except Exception as e:
    print("❌ Error loading Hybrid model:", e)
    hybrid_model = None

# ======================================================
# 2. LOAD EDGE INDEX
# ======================================================
try:
    SEL_EDGES, EDGE_ID_MAP, EDGE_INDEX = build_edge_index(EDGE_FILE)
    print("✅ Edge index loaded. SEL_EDGES length:", len(SEL_EDGES) if SEL_EDGES is not None else "None")
except Exception as e:
    print("❌ Error loading edge_index:", e)
    SEL_EDGES, EDGE_ID_MAP, EDGE_INDEX = None, None, None

# ======================================================
# 3. LOAD PPO AGENT (optional, kept but not used for demo selection)
# ======================================================
try:
    from stable_baselines3 import PPO
    ppo_agent = PPO.load(PPO_PATH, device=device)
    print("✅ PPO agent loaded.")
except Exception as e:
    print("❌ PPO load error (continuing without PPO):", e)
    ppo_agent = None

# ======================================================
# 4. LOAD YOLO MODEL
# ======================================================
try:
    yolo_model = YOLO(YOLO_PATH)
    print("✅ YOLO model loaded.")
except Exception as e:
    print("❌ YOLO load error:", e)
    yolo_model = None

# ======================================================
# ROUTES (Frontend Pages)
# ======================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/map")
def map_page():
    return render_template("map.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/influence")
def influence_page():
    return render_template("influence.html")

# ======================================================
# Helper: format timestamp safely
# ======================================================
def make_timestamp(date_str, time_str):
    """
    date_str: 'YYYY-MM-DD' or None
    time_str: 'HH:MM' or 'HH:MM:SS' or None
    returns 'YYYY-MM-DDTHH:MM'
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    if time_str is None or time_str == "":
        time_str = "10:00"
    parts = str(time_str).split(":")
    if len(parts) >= 2:
        time_short = f"{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    else:
        time_short = "10:00"
    return f"{date_str}T{time_short}"

# ======================================================
# API — HYBRID MULTI-ROAD PREDICTION
# ======================================================
@app.route("/api/hybrid_predict", methods=["POST"])
def hybrid_predict():
    global last_hybrid_prediction, last_hybrid_meta

    if hybrid_model is None:
        return jsonify({"error": "Hybrid model not loaded on server."}), 500

    try:
        data = request.json or {}
        date = data.get("date")
        time = data.get("time", "10:00")
        scenario = data.get("scenario", "normal")

        timestamp = make_timestamp(date, time)

        # Generate sequence and run model
        seq = generate_traffic_sequence(timestamp, scenario)  # expected shape (12, N, 8)
        x = torch.tensor(seq).unsqueeze(0).float()            # (1, 12, N, 8) or the shape your model expects

        with torch.no_grad():
            preds = hybrid_model(x, EDGE_INDEX).detach().cpu().numpy().flatten()

        # Save to global store so PPO endpoint can reuse exactly this set
        last_hybrid_prediction = preds.copy()
        last_hybrid_meta = {
            "date": date,
            "time": time,
            "scenario": scenario,
            "timestamp": timestamp
        }

        roads = [f"R{i}" for i in range(len(preds))]

        return jsonify({
            "roads": roads,
            "speeds": preds.tolist(),
            "unit": "km/h",
            "date": date,
            "time": time,
            "scenario": scenario
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================================================
# API — PPO ROUTE RECOMMENDATION (range-limited & uses last hybrid)
# ======================================================
@app.route("/api/ppo_route", methods=["POST"])
def ppo_route():
    global last_hybrid_prediction, last_hybrid_meta

    if hybrid_model is None:
        return jsonify({"error": "Hybrid model not loaded on server."}), 500

    try:
        data = request.json or {}
        # validate start/end provided
        if "start" not in data or "end" not in data:
            return jsonify({"error": "Please provide 'start' and 'end' indices."}), 400

        start = int(data.get("start", 0))
        end = int(data.get("end", 0))
        date = data.get("date")
        time = data.get("time", "10:00")
        scenario = data.get("scenario", "normal")

        timestamp = make_timestamp(date, time)

        # Use stored hybrid preds if they exactly match the requested timestamp & scenario
        use_preds = None
        if last_hybrid_prediction is not None and last_hybrid_meta is not None:
            if last_hybrid_meta.get("timestamp") == timestamp and last_hybrid_meta.get("scenario") == scenario:
                use_preds = last_hybrid_prediction.copy()

        # Otherwise generate fresh preds and store them
        if use_preds is None:
            seq = generate_traffic_sequence(timestamp, scenario)
            x = torch.tensor(seq).unsqueeze(0).float()
            with torch.no_grad():
                use_preds = hybrid_model(x, EDGE_INDEX).detach().cpu().numpy().flatten()
            last_hybrid_prediction = use_preds.copy()
            last_hybrid_meta = {
                "date": date,
                "time": time,
                "scenario": scenario,
                "timestamp": timestamp
            }

        # Validate start/end bounds with number of predicted roads
        n_roads = len(use_preds)
        if start < 0 or end < 0 or start >= n_roads or end >= n_roads:
            return jsonify({"error": f"start/end must be in range 0..{n_roads-1}"}), 400

        # Restrict to corridor [low, high] inclusive
        low = min(start, end)
        high = max(start, end)

        subset = use_preds[low : high + 1]  # inclusive
        if subset.size == 0:
            return jsonify({"error": "No roads in the selected start-end range."}), 400

        best_local_idx = int(np.argmax(subset))
        best_global_idx = low + best_local_idx
        best_speed = float(use_preds[best_global_idx])

        # Also provide actual edge_id if SEL_EDGES mapping exists
        real_edge_id = None
        if SEL_EDGES is not None and 0 <= best_global_idx < len(SEL_EDGES):
            real_edge_id = SEL_EDGES[best_global_idx]

        return jsonify({
            "start": start,
            "end": end,
            "date": date,
            "time": time,
            "scenario": scenario,
            "recommended_route_index": best_global_idx,
            "recommended_edge_id": real_edge_id,
            "predicted_speed": best_speed,
            "note": f"Best road between R{start} and R{end} based on predicted speed."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================================================
@app.route("/api/route_map_full", methods=["POST"])
def route_map_full():
    global last_hybrid_prediction

    try:
        if last_hybrid_prediction is None:
            return jsonify({"error": "Run Hybrid Prediction first"}), 400

        start = int(request.json["start"])
        end = int(request.json["end"])

        speeds = last_hybrid_prediction  # shape (N,)
        df = pd.read_csv(EDGE_FILE)

        # create map centered around mean coords if geometry exists, otherwise fallback
        m = folium.Map(location=[12.8450, 77.6600], zoom_start=14)

        for i in range(len(speeds)):
            row = df[df["edge_id"] == (i+1)]
            if row.empty:
                continue
            row = row.iloc[0]
            geom = str(row["geometry"])
            coords = []

            # Parse LINESTRING (simple parser; expects "LINESTRING (lon lat, lon lat, ...)")
            try:
                geom_str = geom.replace("LINESTRING (", "").replace(")", "")
                for pair in geom_str.split(","):
                    lon, lat = pair.strip().split()
                    coords.append([float(lat), float(lon)])
            except Exception:
                # skip malformed geometry
                continue

            speed = float(speeds[i])

            # Color scale
            if speed >= 25:
                color = "green"
            elif speed >= 18:
                color = "orange"
            else:
                color = "red"

            folium.PolyLine(
                coords,
                color=color,
                weight=5,
                tooltip=f"Edge R{i} | Speed: {speed:.2f} km/h"
            ).add_to(m)

        # Highlight best road
        best_idx = int(np.argmax(speeds))
        best_row = df[df["edge_id"] == (best_idx+1)]
        if not best_row.empty:
            best_row = best_row.iloc[0]
            best_geom = str(best_row["geometry"]).replace("LINESTRING (", "").replace(")", "")
            best_coords = []
            try:
                for pair in best_geom.split(","):
                    lon, lat = pair.strip().split()
                    best_coords.append([float(lat), float(lon)])
                folium.PolyLine(
                    best_coords,
                    color="blue",
                    weight=7,
                    tooltip=f"BEST ROAD: R{best_idx} | {speeds[best_idx]:.2f} km/h"
                ).add_to(m)
            except Exception:
                pass

        # Save map
        out_path = os.path.join(BASE_DIR, "static", "maps", "traffic_full.html")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        m.save(out_path)

        return jsonify({"map_url": "/static/maps/traffic_full.html"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================================================
# API — YOLO DETECTION
# ======================================================
@app.route("/api/yolo_detect", methods=["POST"])
def yolo_detect():
    try:
        file = request.files["image"]
        save_path = os.path.join(BASE_DIR, "static", "uploads", file.filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)

        results = yolo_model(save_path)[0]
        count = len(results.boxes)

        return jsonify({
            "vehicle_count": int(count),
            "filename": file.filename
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================================================
# API — MAP GENERATION (simple start/end)
# ======================================================
@app.route("/api/route_map", methods=["POST"])
def route_map():
    try:
        start = request.json["start"]
        end = request.json["end"]

        m = folium.Map(location=start, zoom_start=14)
        folium.Marker(start, popup="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(end, popup="End", icon=folium.Icon(color="red")).add_to(m)
        folium.PolyLine([start, end], color="blue").add_to(m)

        map_path = os.path.join(BASE_DIR, "static", "maps", "route_map.html")
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        m.save(map_path)

        return jsonify({"map_url": "/static/maps/route_map.html"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    # optional: set seeds for deterministic behavior globally (uncomment if desired)
    # np.random.seed(42); random.seed(42); torch.manual_seed(42)
    app.run(debug=True)
