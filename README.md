AI Traffic Prediction & Route Optimization System

Overview

This project implements a hybrid deep learning and reinforcement learning framework for real-time traffic congestion prediction and intelligent route optimization.

The system integrates:

LSTM (Long Short-Term Memory) for learning temporal traffic patterns

Graph Attention Networks (GAT) for modeling spatial relationships between connected roads

PPO Reinforcement Learning to select optimal routes based on predicted congestion

The framework was tested on traffic data from Electronic City, Bengaluru, one of the most congested road networks in the city. The system predicts congestion and dynamically recommends faster routes to reduce travel time. 


Key Features

Hybrid LSTM + GAT deep learning model for spatio-temporal traffic prediction

YOLOv8 vehicle detection from CCTV feeds to estimate traffic density

Reinforcement learning (PPO) for adaptive route selection

Flask REST API backend for model integration

Interactive web interface built using HTML, CSS, and JavaScript

Real-time congestion visualization and route recommendation

System Architecture

The system consists of five major components:

1. Data Collection

Traffic data was collected for 60 days at 5-minute intervals from the Electronic City road network.

The road network was extracted using OSMnx, creating a graph where:

Nodes represent intersections

Edges represent road segments. 



2. Vehicle Detection

Traffic density is estimated using YOLOv8 object detection, which detects:

cars

buses

trucks

two-wheelers

Vehicle counts are used as features for congestion prediction.

3. Traffic Prediction

The system predicts future congestion using a hybrid deep learning model:

LSTM captures temporal patterns in traffic flow

Graph Attention Network (GAT) captures spatial dependencies between connected roads

This allows the model to understand both time-based patterns and road network relationships.

4. Route Optimization

A Proximal Policy Optimization (PPO) reinforcement learning agent learns optimal routes.

The reward function balances:

vehicle speed

congestion level

travel delay

This allows the agent to select less congested and faster routes dynamically. 



5. Visualization Interface

A lightweight web frontend was built using:

HTML

CSS

JavaScript

The interface displays:

predicted traffic speed

congestion levels

recommended routes

A color-coded system indicates traffic status:

🟥 Red → Heavy congestion

🟨 Yellow → Moderate traffic

🟩 Green → Smooth flow. 



Tech Stack
Programming

Python

Deep Learning

PyTorch

LSTM

Graph Attention Networks

Computer Vision

YOLOv8

Reinforcement Learning

PPO (Stable-Baselines3)

Backend

Flask REST API

Frontend

HTML

CSS

JavaScript

Data Processing

Pandas

NumPy

OSMnx

Results

The model achieved strong performance in predicting traffic congestion.

Model	MAE	RMSE	R²
LSTM	0.535	0.788	0.976
GAT-LSTM	1.344	2.005	0.930

Integrating YOLO vehicle detection improved responsiveness during sudden congestion spikes. 


How to Run

Clone the repository

git clone https://github.com/Mounanjali19/AITrafficPredictionandRouteOptimization

Install dependencies

pip install -r requirements.txt

Run congestion prediction

python congestion_prediction.py

Run route optimization

python route_opt_tester.py

Future Improvements

Integration with live traffic APIs

Deployment on cloud platforms

Edge deployment using Jetson Nano

Multi-camera traffic monitoring

Smart traffic signal optimization. 
