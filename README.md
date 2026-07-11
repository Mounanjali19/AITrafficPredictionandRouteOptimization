# 🚦 AI Traffic Prediction & Adaptive Route Optimization

> **A Hybrid Deep Learning Framework for Predictive Traffic Congestion Management and Adaptive Route Optimization using LSTM, Graph Attention Networks (GAT), YOLOv8, and Proximal Policy Optimization (PPO).**

---

## 📖 Overview

Traffic congestion remains one of the biggest challenges in modern urban transportation systems, leading to increased travel time, fuel consumption, environmental pollution, and economic losses. Traditional navigation systems primarily rely on current traffic conditions, making them reactive rather than predictive.

This project presents a **hybrid Artificial Intelligence framework** that combines **spatio-temporal traffic prediction** with **adaptive route optimization** to proactively manage urban traffic. The framework predicts future congestion using deep learning and dynamically recommends optimal routes using reinforcement learning.

The proposed system integrates:

- 🧠 Long Short-Term Memory (LSTM) for temporal traffic forecasting
- 🌐 Graph Attention Networks (GAT) for learning spatial road dependencies
- 🚗 YOLOv8 for vehicle detection and traffic density estimation
- 🎯 Proximal Policy Optimization (PPO) for adaptive route optimization

The framework was developed and evaluated on the **Electronic City road network in Bengaluru**, consisting of **250 road intersections** and **598 road segments**, with traffic simulated over **60 days at 5-minute intervals**.

---

# ✨ Features

- Hybrid LSTM-GAT traffic prediction model
- Spatio-temporal traffic forecasting
- Real-time vehicle detection using YOLOv8
- Adaptive route optimization using PPO Reinforcement Learning
- Dynamic congestion-aware routing
- Interactive traffic visualization dashboard
- Flask REST API backend
- Intelligent route recommendation
- Modular and scalable architecture

---

# 🏗️ System Architecture

```
                     Historical Traffic Data
                               │
                               ▼
                    Data Acquisition & Preprocessing
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
      LSTM Network                    YOLOv8 Vehicle Detection
 (Temporal Feature Learning)         (Traffic Density Estimation)
              │
              ▼
      Graph Attention Network
   (Spatial Dependency Learning)
              │
              ▼
     Predicted Traffic Speeds
              │
      Dynamic Edge Weight Update
              │
              ▼
 PPO Reinforcement Learning Agent
              │
              ▼
   Optimal Route Recommendation
              │
              ▼
      Interactive Web Dashboard
```

---

# 🧠 Methodology

## 1. Data Acquisition

The road network was extracted from **OpenStreetMap (OSM)** using the **OSMnx** library.

### Dataset

- 📍 Location: Electronic City, Bengaluru
- 🛣️ 250 Nodes
- 🛣️ 598 Road Segments
- 📅 60 Days of Traffic Data
- ⏱️ 5-Minute Sampling Interval
- 📊 Over 10 Million Traffic Observations

Traffic features include:

- Vehicle speed
- Traffic flow
- Congestion level
- Weather conditions
- Vehicle density

---

## 2. Data Preprocessing

Before training, the collected data undergoes:

- Missing value handling
- Outlier interpolation
- Time synchronization
- Min-Max Normalization
- Graph index mapping
- Feature engineering

---

## 3. Traffic Prediction

The prediction module combines temporal and spatial learning.

### LSTM

The LSTM network learns:

- Historical traffic flow
- Time-dependent congestion patterns
- Long-term temporal dependencies

### Graph Attention Network (GAT)

The GAT model captures:

- Spatial road connectivity
- Congestion propagation
- Neighboring road influence
- Graph-based traffic relationships

The temporal and spatial representations are fused to predict future traffic conditions.

---

## 4. Vehicle Detection

YOLOv8 processes CCTV footage to estimate traffic density by detecting:

- Cars
- Buses
- Trucks
- Motorcycles

Vehicle counts are incorporated as additional features for congestion prediction.

---

## 5. Adaptive Route Optimization

Predicted traffic speeds are converted into dynamic edge weights within the road network graph.

A **Proximal Policy Optimization (PPO)** agent continuously learns the optimal routing policy by minimizing:

- Travel time
- Traffic congestion
- Route delay

Unlike conventional shortest-path algorithms, PPO adapts to changing traffic conditions in real time.

---

# 💻 Technology Stack

## Programming Languages

- Python
- HTML
- CSS
- JavaScript

## Machine Learning & Deep Learning

- PyTorch
- LSTM
- Graph Attention Networks (GAT)

## Computer Vision

- YOLOv8
- OpenCV

## Reinforcement Learning

- Stable-Baselines3
- Proximal Policy Optimization (PPO)

## Data Processing

- NumPy
- Pandas

## Graph Processing

- OSMnx
- NetworkX

## Backend

- Flask REST API

## Frontend

- HTML
- CSS
- JavaScript

---

# 📊 Experimental Results

## Traffic Prediction Performance

| Model | MAE | RMSE | R² Score |
|------|------:|------:|------:|
| LSTM | 0.5354 | 0.7880 | 0.9766 |
| Hybrid LSTM-GAT | 1.3447 | 2.0056 | 0.9309 |

Although the standalone LSTM achieved lower numerical prediction error, the Hybrid LSTM-GAT model effectively captures **spatial traffic dependencies**, enabling more informed routing decisions and improved congestion management.

---

## Adaptive Routing Performance

Compared with the traditional **Dijkstra shortest-path algorithm**, the PPO-based routing agent achieved:

- 🚀 **10–25% reduction in travel time**
- 🚦 Better congestion avoidance
- 🔄 Dynamic route adaptation
- 📈 Improved traffic distribution

---

# 🌍 Case Study

The framework was evaluated using the **Electronic City road network in Bengaluru**, a densely populated urban region with highly dynamic traffic conditions.

### Road Network Statistics

| Parameter | Value |
|------------|-------|
| Nodes | 250 |
| Road Segments | 598 |
| Duration | 60 Days |
| Sampling Interval | 5 Minutes |
| Traffic Samples | 10+ Million |

---

# 📸 Project Demonstration

The project includes:

- 🚦 Traffic Prediction Dashboard
- 🛣️ Route Recommendation Interface
- 🌐 Road Network Visualization
- 📊 Traffic Prediction Graphs
- 🔥 GAT Attention Heatmap
- 🚗 YOLOv8 Vehicle Detection

*(Screenshots can be added in the `/assets` folder.)*

---

# 🚀 Getting Started

## Clone the Repository

```bash
git clone https://github.com/Mounanjali19/AITrafficPredictionandRouteOptimization.git
```

## Navigate to the Project

```bash
cd AITrafficPredictionandRouteOptimization
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Traffic Prediction

```bash
python congestion_prediction.py
```

## Run Route Optimization

```bash
python route_opt_tester.py
```

## Start the Flask Server

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

# 📂 Project Structure

```
AITrafficPredictionandRouteOptimization
│
├── Traffic-app/
├── models/
├── dataset/
├── static/
├── templates/
├── congestion_prediction.py
├── route_opt_tester.py
├── app.py
├── requirements.txt
└── README.md
```

---

# 🎯 Motivation

Conventional navigation systems react only to existing traffic conditions and often fail to anticipate future congestion.

This project bridges the gap between **traffic prediction** and **routing decisions** by integrating deep learning with reinforcement learning into a unified framework. By predicting congestion before it occurs and incorporating those predictions into route planning, the system enables proactive traffic management and more efficient urban mobility.

---

# 🔮 Future Work

- Integration with live traffic APIs
- Deployment on AWS or Azure Cloud
- Multi-camera traffic monitoring
- Smart traffic signal optimization
- Edge deployment using NVIDIA Jetson
- Fuel-efficient route optimization
- Multi-modal transportation support
- Large-scale metropolitan deployment

---


# 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

Feel free to fork the repository, open issues, or submit pull requests.

---

# ⭐ Support

If you found this project useful or interesting, consider giving the repository a **⭐ Star**.

It helps others discover the project and supports future development.
