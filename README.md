# Algi. — Aquatic Carbon Removal & Ecosystem Verification Platform

An industrial dual-core **MRV (Measurement, Reporting, and Verification)** platform for aquatic microalgae carbon bio-sequestration, built under **Verra VM0042** and **ISO 14064-2** protocols.

Now secured with **Role-Based Access Control (RBAC)**, multi-tenant farm data isolation ("bank-account" model), and cryptographic PDF audit report generation.

---

## Quick Start for Evaluators

### 1. Environment Setup & Dependencies
Ensure Python 3.10+ is installed, then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys (`.env`)
Create a `.env` file in the project root with your credentials (or enter them as Admin in the UI):

```env
GEMINI_API_KEY="your_gemini_api_key"
GROQ_API_KEY="your_groq_api_key"
KAGGLE_API_TOKEN="your_kaggle_api_token"
```

---

## 3. Portal Authentication & Demo Credentials

The platform protects farm data behind a secure SQLite password vault (PBKDF2/SHA-256 hashed). No public sign-up is permitted; accounts are provisioned exclusively by the Administrator.

| Role | User ID / Username | Password | Purpose & Capabilities |
| :--- | :--- | :--- | :--- |
| **Platform Administrator** | `admin` | `Admin@Algi2026` | Provision new farm operators, view registry, configure global API keys (Gemini, Groq), inspect full MRV platform. |
| **Farm Operator (User)** | `operator_alpha` | `Farm@1234` | Access isolated farm workspace (`Alpha Spirulina Bioreactor Station`), manage parameters, run ML inference, export certified PDF report, change password. *(API keys strictly hidden)* |

---

## 4. Dataset Instructions (2.3 GB Kaggle Dataset)

> **Note on GitHub Upload**: Due to GitHub's file size limits, the full **2.3 GB authentic image dataset** (`1,305` high-resolution 1080p pond bloom photographs + ground-truth biological segmentation masks) is not stored directly in this Git repository.

### How to Download / Import the 2.3 GB Dataset:

We provide an automated download script that fetches and unpacks the authentic dataset directly into the required directory structure (`data/images/dataset/`):

1. Make sure your Kaggle credentials are set in either:
   - Your `.env` file as `KAGGLE_API_TOKEN="your_token"`
   - Or in `~/.kaggle/access_token` or `~/.kaggle/kaggle.json`

2. Run this single command in your terminal:
   ```bash
   python src/fetch_images.py
   ```
   *This automatically pulls `beyondstellaris/bluegreen-algae-dataset` from Kaggle, unzips the RGB images and segmentation masks into `data/images/dataset/`, and verifies integrity.*

> **Instant Demo Without Downloading**: If you wish to test the application immediately without downloading the full 2.3 GB dataset, the platform includes pre-packaged curated presentation samples (`sample_*.jpg`) located in `data/images/`. The app detects available images and functions seamlessly in both modes!

---

## 5. Run the Machine Learning Models (Optional)

The pre-trained production model files are already included in `models/`:
* `algae_growth_model.pkl` (XGBRegressor — $R^2 = 0.959$)
* `water_quality_model.pkl` (XGBClassifier — $99.6\%$ accuracy)
* `label_encoder.pkl` (3-class ecological status)

To retrain the models from the raw data CSVs at any time, run:
```bash
python train_models.py
```

---

## 6. Launch the Web Application

To launch the secure interactive MRV dashboard, execute:

```bash
streamlit run app.py
```

Once launched, open your browser at **`http://localhost:8501`** (or the port specified in terminal output).

---

## Key Platform Features

1. **Role-Based Access Control (RBAC)**:
   - **Administrator**: Dedicated user provisioning interface, full operator registry, and exclusive API key vault.
   - **Farm Operator**: Isolated parameters saved per user in SQLite (`user_farm_data`). Cannot view or edit API keys. Double-verification password changing.
2. **Universal Shared Intelligence**:
   - High-performance universal XGBoost models (`algae_growth_model.pkl` and `water_quality_model.pkl`) shared across all tenants.
3. **Continuous Dynamic MRV Index**:
   - Multi-dimensional biological verification score (0.0% – 100.0%) reacting in real-time to DO, pH, Temperature, Ammonia toxicity, and ML confidence.
4. **Carbon Credit Financial Valuation ($ USD)**:
   - Evaluates Gross Removal vs. Certified Tradable Value based on standard VCM benchmarks ($42.50/tCO2e) and MRV uncertainty discounting.
5. **Certified Downloadable PDF Audit Report**:
   - Generate and download an official, cryptographically verifiable PDF report on carbon absorption, MRV scores, and credit calculations directly from the dashboard.

---

## Architecture Overview

```text
algae-carbon-platform/
│
├── data/
│   ├── algi_auth.db              # SQLite database (Users & isolated farm data)
│   ├── raw/                      # Research on Algae Growth & Pondsdata CSVs
│   └── images/                   # Sample images + Kaggle 2.3 GB dataset
│       └── dataset/              # Generated via: python src/fetch_images.py
│           ├── leftImg8bit/      # 1,305 1080p RGB bloom captures
│           └── gtFine/           # 1,305 biological segmentation masks
│
├── models/                       # Trained universal XGBoost models
│   ├── algae_growth_model.pkl
│   ├── water_quality_model.pkl
│   └── label_encoder.pkl
│
├── src/                          # Core intelligence & security modules
│   ├── auth_manager.py           # RBAC, PBKDF2 hashing, SQLite data isolation
│   ├── pdf_generator.py          # ReportLab certified PDF report generator
│   ├── fetch_images.py           # Automated Kaggle 2.3 GB dataset downloader
│   ├── telemetry_engine.py       # Live IoT simulation & Open-Meteo weather sync
│   ├── mrv_engine.py             # Continuous MRV Index & Carbon Credit USD engine
│   ├── api_services.py           # Multimodal Gemini 3.6 / Groq vision inspection
│   └── fusion_layer.py           # ISO 14064 AI auditor report generator
│
├── .gitignore                    # Excludes 2.3GB dataset, SQLite db, & keys
├── requirements.txt              # Complete Python dependencies
├── train_models.py               # Pipeline to train XGBoost models
└── app.py                        # Streamlit web application with RBAC & PDF
```
