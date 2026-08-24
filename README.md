🌊 POSEIDON V2
======================
**⚡ AI-Powered Maritime Intelligence & IUU Fishing Detection Pipeline ⚡**

A real-time satellite imagery analysis pipeline that monitors the WPP 711 (North Natuna Sea) for Illegal, Unreported, and Unregulated (IUU) fishing activities. POSEIDON integrates Sentinel-1 SAR (Synthetic Aperture Radar) data from Google Earth Engine (GEE), geospatial oceanographic features, and a PU-Learning AI ensemble to detect dark vessels and assign automated risk scores.

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Tech Stack](#-tech-stack)

---

## 🎯 Overview

POSEIDON is an industrial-grade maritime security solution designed to identify high-risk vessels that turn off their AIS (Automatic Identification System). By leveraging satellite radar, spatial databases, and machine learning, the system extracts vessel coordinates, estimates physical lengths, calculates a probabilistic fishing score, and filters out sea-clutter noise.

Built with a robust, automated Python backend designed for seamless daily execution and data synchronization with a Supabase cloud environment.

## ✨ Features

### 🛰️ Automated Satellite Extraction
*   **Sentinel-1 GRD Integration:** Real-time polling of new satellite passes.
*   **Polarimetric Extraction:** Extracts VV, VH, and SNR (Signal-to-Noise Ratio) to identify synthetic signatures on the ocean surface.
*   **Clutter Filtering:** Unsupervised Anomaly Detection (Isolation Forest) to eliminate wave noise and ghost reflections.

### 🤖 AI-Powered Risk Scoring
*   **PU-Learning Ensemble:** Combines LightGBM and XGBoost for robust confidence scoring.
*   **Dynamic Thresholding:** Calculates adaptive Q75/Q90 risk thresholds dynamically based on quarterly historical baselines.
*   **Explainable AI (SHAP):** Generates human-readable explanations detailing exactly *why* a vessel was flagged as Siaga 1, 2, or 3.

### 🌊 Oceanographic & Spatial Integration
*   **Environmental Grids:** Merges SST, Bathymetry, Chlorophyll-a, and Current Speeds.
*   **GIS Processing:** Ultra-fast `cKDTree` calculations for distances to MPA (Marine Protected Areas), EEZ boundaries, ports, and historical seizure hotspots.

### 📊 Cloud Database Sync
*   **Supabase Upserts:** Automatically maintains `vessel_detections`, `top10_priorities`, and `satellite_passes` summary tables.

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                      🛰️ Google Earth Engine                     │
│              Sentinel-1 SAR Imagery (WPP 711 Area)              │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ GEE API
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   🌊 POSEIDON Data Pipeline                     │
│    Data Extraction ➔ Oceanographic Join ➔ AI Inference          │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ REST API
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     📊 Supabase Cloud Data                      │
│        vessel_detections | top10_priorities | satellite_passes  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Installation

**Prerequisites**
*   Python 3.10+
*   Google Earth Engine Service Account (`.json`)
*   Supabase Project

**Quick Setup**

```bash
# Clone the repository
git clone https://github.com/JullMol/poseidon-automation.git
cd poseidon-automation

# Install dependencies
pip install -r requirements.txt
```

**Environment Variables**
Configure the following secrets in your environment or CI/CD runner:
*   `SUPABASE_URL`
*   `SUPABASE_KEY`
*   `EE_SERVICE_ACCOUNT`
*   `EE_PRIVATE_KEY`

## 🚀 Usage

### 1️⃣ Trigger Daily Production Pipeline
Run the fully automated pipeline. This script strictly requires environment variables and will push the processed results directly to Supabase.

```bash
python automation/daily_pipeline.py
```

## 📁 Project Structure

```text
POSEIDON/
├── 📂 automation/
│   └── daily_pipeline.py         # Main execution pipeline (Production)
└── README.md                     # Documentation
```

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Pipeline Engine** | Python, Pandas, GeoPandas |
| **AI / Inference** | LightGBM, XGBoost, Scikit-Learn |
| **Geospatial Processing** | Google Earth Engine API, cKDTree, Shapely |
| **Database** | Supabase (PostgreSQL) |

## 🤝 Contributing

Contributions are welcome! Please ensure that no hardcoded credentials or API keys are committed to the repository.

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m 'Add amazing feature'`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Open a Pull Request

---
🌟 **Star this repo if you find it useful!**
*Made with ❤️ for maritime safety and sustainability.*
