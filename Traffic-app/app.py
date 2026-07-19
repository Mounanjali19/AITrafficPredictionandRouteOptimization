import os
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import folium

from flask import Flask, render_template, request, jsonify


# ======================================================
# APP SETUP
# ======================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EDGE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "ecity_edges.csv"
)


# ======================================================
# GLOBAL VARIABLES
# ======================================================

last_hybrid_prediction = None
last_hybrid_meta = None


# ======================================================
# FRONTEND ROUTES
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
# TIMESTAMP HELPER
# ======================================================

def make_timestamp(date_str, time_str):

    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    if not time_str:
        time_str = "10:00"

    parts = str(time_str).split(":")

    if len(parts) >= 2:
        time_short = (
            f"{parts[0].zfill(2)}:"
            f"{parts[1].zfill(2)}"
        )
    else:
        time_short = "10:00"

    return f"{date_str}T{time_short}"


# ======================================================
# GET NUMBER OF ROADS
# ======================================================

def get_number_of_roads():

    try:
        df = pd.read_csv(EDGE_FILE)

        if len(df) > 0:
            return len(df)

    except Exception as e:
        print("Could not read edge file:", e)

    # Fallback if CSV cannot be read
    return 50


# ======================================================
# LIGHTWEIGHT TRAFFIC PREDICTION
#
# This replaces live GAT-LSTM inference on the
# low-memory Render deployment.
#
# Results are deterministic for the same:
# date + time + scenario
# ======================================================

def generate_demo_predictions(
    date,
    time,
    scenario
):

    timestamp = make_timestamp(
        date,
        time
    )

    number_of_roads = get_number_of_roads()

    # Create deterministic seed
    seed_text = (
        f"{timestamp}-{scenario}"
    )

    seed_hash = hashlib.sha256(
        seed_text.encode()
    ).hexdigest()

    seed = int(
        seed_hash[:8],
        16
    )

    rng = np.random.default_rng(seed)


    # ----------------------------------------------
    # Base traffic speed
    # ----------------------------------------------

    base_speed = 27.0


    # ----------------------------------------------
    # Time influence
    # ----------------------------------------------

    try:
        hour = int(
            str(time).split(":")[0]
        )

    except Exception:
        hour = 10


    # Morning rush hour
    if 7 <= hour <= 10:
        base_speed -= 7

    # Evening rush hour
    elif 16 <= hour <= 20:
        base_speed -= 9

    # Late night
    elif hour >= 22 or hour <= 5:
        base_speed += 8


    # ----------------------------------------------
    # Scenario influence
    # ----------------------------------------------

    scenario_lower = str(
        scenario
    ).lower()


    if scenario_lower == "accident":

        base_speed -= 10


    elif scenario_lower in [
        "rain",
        "rainy"
    ]:

        base_speed -= 6


    elif scenario_lower in [
        "heavy",
        "heavy traffic",
        "congestion"
    ]:

        base_speed -= 9


    elif scenario_lower in [
        "clear",
        "light"
    ]:

        base_speed += 4


    # ----------------------------------------------
    # Generate road-level variation
    # ----------------------------------------------

    road_variation = rng.normal(
        loc=0,
        scale=5,
        size=number_of_roads
    )


    predictions = (
        base_speed
        +
        road_variation
    )


    # Keep realistic speed range
    predictions = np.clip(
        predictions,
        5,
        55
    )


    return predictions


# ======================================================
# HYBRID PREDICTION API
#
# API name remains unchanged so your frontend
# does not need modification.
# ======================================================

@app.route(
    "/api/hybrid_predict",
    methods=["POST"]
)
def hybrid_predict():

    global last_hybrid_prediction
    global last_hybrid_meta

    try:

        data = request.get_json(
            silent=True
        ) or {}

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


        predictions = (
            generate_demo_predictions(
                date,
                time,
                scenario
            )
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

            for i in range(
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
            scenario,

            "deployment_mode":
            "lightweight-demo"

        })


    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return jsonify({

            "error":
            str(e)

        }), 500


# ======================================================
# ROUTE RECOMMENDATION
#
# Keeps existing /api/ppo_route endpoint for
# frontend compatibility.
# ======================================================

@app.route(
    "/api/ppo_route",
    methods=["POST"]
)
def ppo_route():

    global last_hybrid_prediction
    global last_hybrid_meta

    try:

        data = request.get_json(
            silent=True
        ) or {}


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
            data["start"]
        )

        end = int(
            data["end"]
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


        # Reuse previous prediction
        if (
            last_hybrid_prediction
            is not None
            and
            last_hybrid_meta
            is not None
        ):

            if (
                last_hybrid_meta.get(
                    "timestamp"
                )
                ==
                timestamp
                and
                last_hybrid_meta.get(
                    "scenario"
                )
                ==
                scenario
            ):

                use_predictions = (
                    last_hybrid_prediction.copy()
                )


        # Generate if prediction not already available
        if use_predictions is None:

            use_predictions = (
                generate_demo_predictions(
                    date,
                    time,
                    scenario
                )
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


        # Try to get real edge ID
        real_edge_id = None

        try:

            df = pd.read_csv(
                EDGE_FILE
            )

            if (
                best_global_index
                <
                len(df)
            ):

                real_edge_id = str(

                    df.iloc[
                        best_global_index
                    ]["edge_id"]

                )

        except Exception:
            pass


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

        print(
            "Route recommendation error:",
            e
        )

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
                "Run prediction first."

            }), 400


        speeds = (
            last_hybrid_prediction
        )


        df = pd.read_csv(
            EDGE_FILE
        )


        traffic_map = folium.Map(

            location=[
                12.8450,
                77.6600
            ],

            zoom_start=14

        )


        # ----------------------------------------------
        # Draw roads
        # ----------------------------------------------

        for i in range(
            min(
                len(speeds),
                len(df)
            )
        ):

            row = df.iloc[i]


            if (
                "geometry"
                not in row
            ):

                continue


            geometry = str(
                row["geometry"]
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

                    f"Road R{i} | "
                    f"{speed:.2f} km/h"

                )

            ).add_to(
                traffic_map
            )


        # ----------------------------------------------
        # Highlight fastest road
        # ----------------------------------------------

        best_index = int(
            np.argmax(
                speeds
            )
        )


        if best_index < len(df):

            try:

                geometry = str(

                    df.iloc[
                        best_index
                    ]["geometry"]

                )


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


                best_coordinates = []


                for pair in (

                    geometry_string
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


            except Exception as e:

                print(
                    "Best road map error:",
                    e
                )


        map_path = os.path.join(

            BASE_DIR,

            "static",

            "maps",

            "traffic_full.html"

        )


        os.makedirs(

            os.path.dirname(
                map_path
            ),

            exist_ok=True

        )


        traffic_map.save(
            map_path
        )


        return jsonify({

            "map_url":
            "/static/maps/traffic_full.html"

        })


    except Exception as e:

        print(
            "Map generation error:",
            e
        )

        return jsonify({

            "error":
            str(e)

        }), 500


# ======================================================
# YOLO ENDPOINT
#
# Disabled on low-memory deployment instead of
# returning fake detection results.
# ======================================================

@app.route(
    "/api/yolo_detect",
    methods=["POST"]
)
def yolo_detect():

    return jsonify({

        "error":
        (
            "Live vehicle detection is disabled "
            "on the lightweight deployment."
        ),

        "deployment_mode":
        "lightweight-demo"

    }), 503


# ======================================================
# SIMPLE ROUTE MAP
# ======================================================

@app.route(
    "/api/route_map",
    methods=["POST"]
)
def route_map():

    try:

        data = request.get_json(
            silent=True
        ) or {}


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

            popup="Start"

        ).add_to(
            route_map_object
        )


        folium.Marker(

            end,

            popup="End"

        ).add_to(
            route_map_object
        )


        folium.PolyLine(

            [
                start,
                end
            ]

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
# RUN
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
