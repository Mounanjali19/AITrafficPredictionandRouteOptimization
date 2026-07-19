import os
import torch
import numpy as np
import pandas as pd
import folium

from flask import Flask, render_template, request, jsonify
from datetime import datetime

from utils.preprocess import generate_traffic_sequence
from utils.graph_utils import build_edge_index
from utils.hybrid_model import HybridGAT_LSTM


# ======================================================
# APP SETUP
# ======================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HYBRID_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hybrid_model_final.pt"
)

YOLO_PATH = os.path.join(
    BASE_DIR,
    "models",
    "yolov8n.pt"
)

EDGE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "ecity_edges.csv"
)

device = "cpu"


# ======================================================
# GLOBAL VARIABLES
# ======================================================

last_hybrid_prediction = None
last_hybrid_meta = None

# YOLO is NOT loaded during server startup.
yolo_model = None


# ======================================================
# LOAD HYBRID MODEL
# ======================================================

try:

    hybrid_model = HybridGAT_LSTM(
        in_dim=8,
        gat_hidden=64,
        gat_heads=4,
        lstm_hidden=128,
        fusion_hidden=128
    ).to(device)

    hybrid_model.load_state_dict(
        torch.load(
            HYBRID_PATH,
            map_location=device
        ),
        strict=True
    )

    hybrid_model.eval()

    print("Hybrid GAT-LSTM model loaded successfully.")

except Exception as e:

    print(
        "Error loading Hybrid model:",
        e
    )

    hybrid_model = None


# ======================================================
# LOAD EDGE INDEX
# ======================================================

try:

    SEL_EDGES, EDGE_ID_MAP, EDGE_INDEX = build_edge_index(
        EDGE_FILE
    )

    print(
        "Edge index loaded successfully.",
        "Number of edges:",
        len(SEL_EDGES)
        if SEL_EDGES is not None
        else "None"
    )

except Exception as e:

    print(
        "Error loading edge index:",
        e
    )

    SEL_EDGES = None
    EDGE_ID_MAP = None
    EDGE_INDEX = None


# ======================================================
# FRONTEND ROUTES
# ======================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/predict")
def predict_page():

    return render_template(
        "predict.html"
    )


@app.route("/map")
def map_page():

    return render_template(
        "map.html"
    )


@app.route("/upload")
def upload_page():

    return render_template(
        "upload.html"
    )


@app.route("/influence")
def influence_page():

    return render_template(
        "influence.html"
    )


# ======================================================
# TIMESTAMP HELPER
# ======================================================

def make_timestamp(
    date_str,
    time_str
):

    if date_str is None:

        date_str = datetime.now().strftime(
            "%Y-%m-%d"
        )

    if time_str is None or time_str == "":

        time_str = "10:00"

    parts = str(
        time_str
    ).split(":")

    if len(parts) >= 2:

        time_short = (
            f"{parts[0].zfill(2)}:"
            f"{parts[1].zfill(2)}"
        )

    else:

        time_short = "10:00"

    return (
        f"{date_str}T{time_short}"
    )


# ======================================================
# HYBRID TRAFFIC PREDICTION
# ======================================================

@app.route(
    "/api/hybrid_predict",
    methods=["POST"]
)
def hybrid_predict():

    global last_hybrid_prediction
    global last_hybrid_meta

    if hybrid_model is None:

        return jsonify({
            "error":
            "Hybrid model not loaded on server."
        }), 500

    try:

        data = request.json or {}

        date = data.get(
            "date"
        )

        time = data.get(
            "time",
            "10:00"
        )

        scenario = data.get(
            "scenario",
            "normal"
        )

        timestamp = make_timestamp(
            date,
            time
        )

        sequence = generate_traffic_sequence(
            timestamp,
            scenario
        )

        x = torch.tensor(
            sequence
        ).unsqueeze(
            0
        ).float()

        with torch.inference_mode():

            predictions = hybrid_model(
                x,
                EDGE_INDEX
            )

            predictions = (
                predictions
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )

        last_hybrid_prediction = (
            predictions.copy()
        )

        last_hybrid_meta = {

            "date":
            date,

            "time":
            time,

            "scenario":
            scenario,

            "timestamp":
            timestamp

        }

        roads = [

            f"R{i}"

            for i
            in range(
                len(predictions)
            )

        ]

        return jsonify({

            "roads":
            roads,

            "speeds":
            predictions.tolist(),

            "unit":
            "km/h",

            "date":
            date,

            "time":
            time,

            "scenario":
            scenario

        })

    except Exception as e:

        return jsonify({

            "error":
            str(e)

        }), 500


# ======================================================
# ROUTE RECOMMENDATION
# ======================================================

@app.route(
    "/api/ppo_route",
    methods=["POST"]
)
def ppo_route():

    global last_hybrid_prediction
    global last_hybrid_meta

    if hybrid_model is None:

        return jsonify({

            "error":
            "Hybrid model not loaded on server."

        }), 500

    try:

        data = request.json or {}

        if (
            "start" not in data
            or
            "end" not in data
        ):

            return jsonify({

                "error":
                "Please provide start and end indices."

            }), 400

        start = int(
            data.get(
                "start",
                0
            )
        )

        end = int(
            data.get(
                "end",
                0
            )
        )

        date = data.get(
            "date"
        )

        time = data.get(
            "time",
            "10:00"
        )

        scenario = data.get(
            "scenario",
            "normal"
        )

        timestamp = make_timestamp(
            date,
            time
        )

        use_predictions = None


        # ----------------------------------------------
        # Reuse previous prediction when possible
        # ----------------------------------------------

        if (
            last_hybrid_prediction
            is not None
            and
            last_hybrid_meta
            is not None
        ):

            same_timestamp = (

                last_hybrid_meta.get(
                    "timestamp"
                )
                ==
                timestamp

            )

            same_scenario = (

                last_hybrid_meta.get(
                    "scenario"
                )
                ==
                scenario

            )

            if (
                same_timestamp
                and
                same_scenario
            ):

                use_predictions = (
                    last_hybrid_prediction.copy()
                )


        # ----------------------------------------------
        # Generate prediction if required
        # ----------------------------------------------

        if use_predictions is None:

            sequence = (
                generate_traffic_sequence(
                    timestamp,
                    scenario
                )
            )

            x = torch.tensor(
                sequence
            ).unsqueeze(
                0
            ).float()

            with torch.inference_mode():

                use_predictions = (
                    hybrid_model(
                        x,
                        EDGE_INDEX
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )

            last_hybrid_prediction = (
                use_predictions.copy()
            )

            last_hybrid_meta = {

                "date":
                date,

                "time":
                time,

                "scenario":
                scenario,

                "timestamp":
                timestamp

            }


        # ----------------------------------------------
        # Validate range
        # ----------------------------------------------

        number_of_roads = len(
            use_predictions
        )

        if (

            start < 0
            or
            end < 0
            or
            start >= number_of_roads
            or
            end >= number_of_roads

        ):

            return jsonify({

                "error":
                (
                    "start/end must be "
                    f"in range 0.."
                    f"{number_of_roads - 1}"
                )

            }), 400


        low = min(
            start,
            end
        )

        high = max(
            start,
            end
        )


        subset = use_predictions[
            low:
            high + 1
        ]


        if subset.size == 0:

            return jsonify({

                "error":
                "No roads in selected range."

            }), 400


        best_local_index = int(
            np.argmax(
                subset
            )
        )


        best_global_index = (

            low
            +
            best_local_index

        )


        best_speed = float(

            use_predictions[
                best_global_index
            ]

        )


        real_edge_id = None


        if (

            SEL_EDGES
            is not None
            and
            0
            <=
            best_global_index
            <
            len(
                SEL_EDGES
            )

        ):

            real_edge_id = (

                SEL_EDGES[
                    best_global_index
                ]

            )


        return jsonify({

            "start":
            start,

            "end":
            end,

            "date":
            date,

            "time":
            time,

            "scenario":
            scenario,

            "recommended_route_index":
            best_global_index,

            "recommended_edge_id":
            real_edge_id,

            "predicted_speed":
            best_speed,

            "note":
            (
                f"Best road between "
                f"R{start} and R{end} "
                "based on predicted speed."
            )

        })


    except Exception as e:

        return jsonify({

            "error":
            str(e)

        }), 500


# ======================================================
# FULL TRAFFIC MAP
# ======================================================

@app.route(
    "/api/route_map_full",
    methods=["POST"]
)
def route_map_full():

    global last_hybrid_prediction

    try:

        if (
            last_hybrid_prediction
            is None
        ):

            return jsonify({

                "error":
                "Run Hybrid Prediction first"

            }), 400


        speeds = (
            last_hybrid_prediction
        )


        dataframe = pd.read_csv(
            EDGE_FILE
        )


        traffic_map = folium.Map(

            location=[
                12.8450,
                77.6600
            ],

            zoom_start=14

        )


        for i in range(
            len(speeds)
        ):

            row = dataframe[

                dataframe[
                    "edge_id"
                ]
                ==
                (
                    i + 1
                )

            ]


            if row.empty:

                continue


            row = row.iloc[0]


            geometry = str(

                row[
                    "geometry"
                ]

            )


            coordinates = []


            try:

                geometry_string = (

                    geometry
                    .replace(
                        "LINESTRING (",
                        ""
                    )
                    .replace(
                        ")",
                        ""
                    )

                )


                for pair in (

                    geometry_string
                    .split(",")

                ):

                    longitude, latitude = (

                        pair
                        .strip()
                        .split()

                    )


                    coordinates.append([

                        float(latitude),

                        float(longitude)

                    ])


            except Exception:

                continue


            speed = float(
                speeds[i]
            )


            if speed >= 25:

                color = "green"

            elif speed >= 18:

                color = "orange"

            else:

                color = "red"


            folium.PolyLine(

                coordinates,

                color=color,

                weight=5,

                tooltip=(
                    f"Edge R{i} | "
                    f"Speed: "
                    f"{speed:.2f} km/h"
                )

            ).add_to(
                traffic_map
            )


        # ----------------------------------------------
        # Highlight best road
        # ----------------------------------------------

        best_index = int(

            np.argmax(
                speeds
            )

        )


        best_row = dataframe[

            dataframe[
                "edge_id"
            ]
            ==
            (
                best_index + 1
            )

        ]


        if not best_row.empty:

            best_row = (
                best_row.iloc[0]
            )


            best_geometry = str(

                best_row[
                    "geometry"
                ]

            ).replace(

                "LINESTRING (",
                ""

            ).replace(

                ")",
                ""

            )


            best_coordinates = []


            try:

                for pair in (

                    best_geometry
                    .split(",")

                ):

                    longitude, latitude = (

                        pair
                        .strip()
                        .split()

                    )


                    best_coordinates.append([

                        float(latitude),

                        float(longitude)

                    ])


                folium.PolyLine(

                    best_coordinates,

                    color="blue",

                    weight=7,

                    tooltip=(

                        f"BEST ROAD: "
                        f"R{best_index} | "
                        f"{speeds[best_index]:.2f} "
                        "km/h"

                    )

                ).add_to(
                    traffic_map
                )


            except Exception:

                pass


        output_path = os.path.join(

            BASE_DIR,

            "static",

            "maps",

            "traffic_full.html"

        )


        os.makedirs(

            os.path.dirname(
                output_path
            ),

            exist_ok=True

        )


        traffic_map.save(
            output_path
        )


        return jsonify({

            "map_url":
            "/static/maps/traffic_full.html"

        })


    except Exception as e:

        return jsonify({

            "error":
            str(e)

        }), 500


# ======================================================
# YOLO VEHICLE DETECTION
#
# IMPORTANT:
# YOLO is loaded ONLY when this API is first called.
# This reduces startup RAM usage.
# ======================================================

@app.route(
    "/api/yolo_detect",
    methods=["POST"]
)
def yolo_detect():

    global yolo_model

    try:

        # Import Ultralytics only when YOLO is actually used.
        if yolo_model is None:

            print(
                "Loading YOLO model..."
            )

            from ultralytics import YOLO

            yolo_model = YOLO(
                YOLO_PATH
            )

            print(
                "YOLO model loaded successfully."
            )


        if "image" not in request.files:

            return jsonify({

                "error":
                "No image uploaded."

            }), 400


        uploaded_file = (
            request.files[
                "image"
            ]
        )


        save_path = os.path.join(

            BASE_DIR,

            "static",

            "uploads",

            uploaded_file.filename

        )


        os.makedirs(

            os.path.dirname(
                save_path
            ),

            exist_ok=True

        )


        uploaded_file.save(
            save_path
        )


        results = (

            yolo_model(
                save_path
            )[0]

        )


        vehicle_count = len(
            results.boxes
        )


        return jsonify({

            "vehicle_count":
            int(
                vehicle_count
            ),

            "filename":
            uploaded_file.filename

        })


    except Exception as e:

        return jsonify({

            "error":
            str(e)

        }), 500


# ======================================================
# SIMPLE ROUTE MAP
# ======================================================

@app.route(
    "/api/route_map",
    methods=["POST"]
)
def route_map():

    try:

        data = request.json or {}

        start = data[
            "start"
        ]

        end = data[
            "end"
        ]


        route_map_object = folium.Map(

            location=start,

            zoom_start=14

        )


        folium.Marker(

            start,

            popup="Start",

            icon=folium.Icon(
                color="green"
            )

        ).add_to(
            route_map_object
        )


        folium.Marker(

            end,

            popup="End",

            icon=folium.Icon(
                color="red"
            )

        ).add_to(
            route_map_object
        )


        folium.PolyLine(

            [
                start,
                end
            ],

            color="blue"

        ).add_to(
            route_map_object
        )


        map_path = os.path.join(

            BASE_DIR,

            "static",

            "maps",

            "route_map.html"

        )


        os.makedirs(

            os.path.dirname(
                map_path
            ),

            exist_ok=True

        )


        route_map_object.save(
            map_path
        )


        return jsonify({

            "map_url":
            "/static/maps/route_map.html"

        })


    except Exception as e:

        return jsonify({

            "error":
            str(e)

        }), 500


# ======================================================
# RUN LOCALLY
# ======================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )
