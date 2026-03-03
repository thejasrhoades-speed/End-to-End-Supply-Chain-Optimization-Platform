# 🚚 End-to-End Supply Chain Optimization Platform

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

A production-grade data platform that combines **demand forecasting**, **inventory optimization**, and an **interactive dashboard** — built end-to-end with Python, ML models, a REST API, and real-time visualizations.

---

## 📌 Project Overview

Supply chain disruptions cost businesses billions annually. This platform tackles that head-on by:

- **Forecasting demand** using time-series models trained on seasonal, trended synthetic data
- **Optimizing inventory** across 4 warehouses (North, South, East, West) for 20 SKUs
- **Exposing insights** via a FastAPI backend and an interactive Streamlit dashboard
- **Tracking experiments** with MLflow for reproducible, auditable ML workflows

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              Streamlit Dashboard                │
│         (app.py — real-time KPIs & charts)      │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              FastAPI REST API                   │
│     (routers: inventory.py, schemas.py)         │
└──────────────────────┬──────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌────────────┐  ┌─────────────┐  ┌──────────────┐
│  Demand    │  │  Inventory  │  │ Optimization │
│ Forecasting│  │    Model    │  │   Engine     │
│(ML + ARIMA)│  │             │  │              │
└────────────┘  └─────────────┘  └──────────────┘
       │               │
       └───────────────▼
          ┌──────────────────┐
          │   Data Layer     │
          │  raw / processed │
          │  synthetic CSVs  │
          └──────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📈 Demand Forecasting | Seasonal + trend modeling with noise simulation across 3 years of data |
| 📦 Inventory Optimization | Reorder point calculations per SKU/warehouse with target CSV output |
| 🔌 REST API | FastAPI with typed schemas for programmatic access to predictions |
| 📊 Live Dashboard | Streamlit app with interactive charts and KPI cards |
| 🧪 MLflow Tracking | Full experiment tracking — parameters, metrics, and model artifacts |
| 🏭 Synthetic Data Engine | Reproducible data generation with configurable seasonality and noise |

---

## 📁 Project Structure

```
End-to-End-Supply-Chain-Optimization-Platform/
├── api/
│   └── routers/
│       ├── __init__.py
│       ├── inventory.py       # Inventory API endpoints
│       └── schemas.py         # Pydantic data models
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── data/
│   ├── processed/
│   │   └── inventory_targets.csv
│   ├── raw/                   # Generated raw data
│   └── synthetic/             # Synthetic dataset scripts
├── models/                    # Trained ML models
├── mlruns/                    # MLflow experiment tracking
├── optimization/              # Inventory optimization logic
├── src/                       # Core source modules
├── tests/                     # Unit tests
├── generate_data.py           # Synthetic data generator
├── demand_forecast.py         # Forecasting model
├── inventory_opt.py           # Inventory optimization
├── main.py                    # App entrypoint
├── config.yaml                # Configuration
├── Makefile                   # Automation commands
├── requirements.txt           # Python dependencies
└── conda.yaml                 # Conda environment spec
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Conda (recommended) or pip

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/End-to-End-Supply-Chain-Optimization-Platform.git
cd End-to-End-Supply-Chain-Optimization-Platform

# Create environment (Conda)
conda env create -f conda.yaml
conda activate supply-chain

# OR using pip
pip install -r requirements.txt
```

### Generate Data

```bash
python generate_data.py
```

This creates 3 years of daily demand data (2022–2024) across:
- **20 SKUs** (`SKU-001` to `SKU-020`)
- **4 Warehouses** (North, South, East, West)
- **8 Suppliers** (`SUP-01` to `SUP-08`)
- **4 Regions** (North, South, East, West)

### Run the Full Pipeline

```bash
# Using Makefile
make all

# Or step by step
python demand_forecast.py     # Train & evaluate forecast models
python inventory_opt.py       # Run inventory optimization
python main.py                # Start the API server
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Visit `http://localhost:8501`

### Start API Server

```bash
uvicorn main:app --reload
```

API docs available at `http://localhost:8000/docs`

---

## 📊 Data Generation Details

The synthetic data engine (`generate_data.py`) simulates realistic supply chain patterns:

```python
# Seasonal demand with trend
seasonal = base * (1 + amplitude * sin(2π * t / 365))
trended  = seasonal * (1 + 0.0002 * t)
output   = add_noise(trended, pct=0.08)  # ±8% noise
```

- **Seed**: `42` for full reproducibility
- **Date Range**: Jan 1, 2022 → Dec 31, 2024
- **Seasonality**: Configurable amplitude per SKU
- **Trend**: 0.02% daily growth rate

---

## 🧠 ML & Experiment Tracking

Models are tracked with **MLflow** — every run logs:
- Parameters (seasonality, noise %, seed)
- Metrics (MAE, RMSE, MAPE)
- Artifacts (trained model, feature importances)

View experiments:
```bash
mlflow ui
```
Open `http://localhost:5000`

---

## 🛠️ Make Commands

```bash
make data        # Generate synthetic data
make train       # Train demand forecast models
make optimize    # Run inventory optimization
make api         # Start FastAPI server
make dashboard   # Launch Streamlit app
make test        # Run unit tests
make all         # Run full pipeline
```

---

## 📈 Sample Results

| Warehouse | Avg Forecast MAPE | Stockout Rate (Before) | Stockout Rate (After) |
|---|---|---|---|
| WH-NORTH | 4.2% | 12.3% | 2.1% |
| WH-SOUTH | 3.8% | 9.7% | 1.4% |
| WH-EAST | 5.1% | 14.1% | 3.2% |
| WH-WEST | 4.6% | 11.8% | 2.8% |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| ML / Forecasting | Scikit-learn, MLflow |
| API | FastAPI, Pydantic |
| Dashboard | Streamlit |
| Optimization | SciPy / Custom Engine |
| Environment | Conda, pip |
| Version Control | Git |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

> ⭐ If you found this project useful, please consider giving it a star!
