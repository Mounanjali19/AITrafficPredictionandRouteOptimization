import numpy as np
from datetime import datetime, timedelta


# Pre-generate static junction factors for 30 roads
np.random.seed(42)
JUNCTION_FACTORS = np.random.uniform(0.8, 1.3, size=30)


def generate_traffic_sequence(timestamp_str, scenario="normal"):
    """
    Generates a realistic 12×30×8 feature tensor for the Hybrid GAT-LSTM model.
    """

    # ----------------------
    # Convert timestamp
    # ----------------------
    dt = datetime.fromisoformat(timestamp_str)   # "2025-01-01T10:00"
    
    # Create the last 12 time points at 5-minute intervals
    time_steps = [dt - timedelta(minutes=5 * i) for i in reversed(range(12))]

    sequence = []

    for t in time_steps:
        minute_of_day = t.hour * 60 + t.minute
        day_of_week = t.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        # Scenario settings
        is_rain = 1 if scenario.lower() == "rain" else 0
        is_event = 1 if scenario.lower() == "event" else 0

        # ----------- BASE SPEED LOGIC -----------
        base_speed = 35  # base speed (km/h)

        # Rush hours (slower)
        if 7 <= t.hour <= 10 or 17 <= t.hour <= 20:
            base_speed -= 10

        # Late night (faster)
        if 22 <= t.hour or t.hour <= 5:
            base_speed += 5

        # Weekend morning (less traffic)
        if is_weekend and 6 <= t.hour <= 10:
            base_speed += 4

        # Rain reduces speed
        if is_rain:
            base_speed *= 0.75

        # Event reduces speed more
        if is_event:
            base_speed *= 0.60

        # Vehicle count logic
        vehicle_count = int(50 + (200 - base_speed * 2))

        # For each of 30 roads
        features_30 = []

        for r in range(30):

            avg_speed = base_speed * np.random.uniform(0.85, 1.15)

            jf = JUNCTION_FACTORS[r]  # static per road

            row = [
                avg_speed,             # avg_speed_kmph
                vehicle_count,         # vehicle_count
                is_weekend,            # is_weekend
                is_rain,               # is_rain
                is_event,              # is_event
                minute_of_day,         # minute_of_day
                day_of_week,           # day_of_week
                jf                     # junction_factor
            ]

            features_30.append(row)

        sequence.append(features_30)

    sequence = np.array(sequence, dtype=np.float32)  # (12,30,8)
    return sequence
