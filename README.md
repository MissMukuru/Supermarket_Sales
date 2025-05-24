# 🛒 Supermarket Sales Forecasting System

**Real-Time Machine Learning Pipeline for Sales Prediction**

This project builds an end-to-end system that predicts daily sales for a supermarket chain using machine learning and live data streaming. It features a production-ready ML pipeline, a Django API, and a Streamlit dashboard for real-time sales monitoring.

---

## 📌 Problem Statement

Supermarkets face challenges forecasting product-level sales due to changing customer behavior, promotions, and holidays. Most rely on historical reporting rather than predictive analytics.

> **Goal:** Build a system to predict next-day product sales per store using historical data, calendar events, weather (optional), and real-time transactions.

---

## 🚀 Deliverables

| Component              | Description                                                     |
|------------------------|-----------------------------------------------------------------|
| 🤖 ML Model            | Trained ML model for daily sales prediction                    |
| ⚙️ Backend API         | Django REST API serving predictions via `/predict/` endpoint   |
| 📊 Streamlit Dashboard | Real-time frontend for monitoring live forecasts                |
| 🔁 Live Data Simulator | Simulates transaction data and pushes to API                   |
| 🐳 Dockerized Services | Fully containerized backend, dashboard, and streaming system   |
| 🔎 Logging & Monitoring| App logs, prediction tracking, model monitoring                |
| 📘 Documentation       | README, setup guides, system architecture                      |

---

## 🧰 Tech Stack

| Layer            | Tools / Frameworks                                         |
|------------------|------------------------------------------------------------|
| 🧠 Machine Learning | TensorFlow, Scikit-learn, Pandas                         |
| 🖥 Backend API    | Django, Django REST Framework, Channels (WebSockets)       |
| 📊 Dashboard     | Streamlit                                                  |
| 🔁 Streaming     | Custom Python simulator or Kafka (optional)                |
| 🐳 Deployment     | Docker, Docker Compose, GitHub Actions                    |
| 🗃️ Data Format    | `.xlsx`, `.csv`, JSON                                     |
| 📡 Monitoring     | Python logging, optional: MLflow, Prometheus              |

---

## 🔁 Pipeline Architecture

```plaintext
[ Live Transaction Data ] --> [ Django API ]
                                |
                                ↓
                      [ ML Inference Engine ]
                                ↓
                   [ Streamlit Dashboard (Live) ]
                                ↓
                        [ Logging & Monitoring ]
