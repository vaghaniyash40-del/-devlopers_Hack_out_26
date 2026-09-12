import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def resolve_csv(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find any of: {candidates}")

def train_all_models():
    print("=" * 60)
    print("TRAINING ALGAE CARBON SEQUESTRATION ML MODELS")
    print("=" * 60)

    # 1. Algae Growth Model
    growth_candidates = [
        os.path.join(DATA_RAW, "Research on Algae Growth_2.csv"),
        os.path.join(DATA_RAW, "Research on Algae Growth.csv"),
        os.path.join(BASE_DIR, "Research on Algae Growth.csv")
    ]
    growth_path = resolve_csv(growth_candidates)
    print(f"\n[1/2] Loading Algae Growth data from: {os.path.basename(growth_path)}")
    growth_df = pd.read_csv(growth_path)
    growth_df.columns = [c.strip() for c in growth_df.columns]

    growth_inputs = ['Light', 'Nitrate', 'Iron', 'Phosphate', 'Temperature', 'pH', 'CO2']
    for col in growth_inputs + ['Population']:
        if col not in growth_df.columns:
            raise KeyError(f"Missing required column '{col}' in {growth_path}. Available: {growth_df.columns.tolist()}")
        growth_df[col] = pd.to_numeric(growth_df[col], errors='coerce')

    # Drop NaNs
    growth_df = growth_df.dropna(subset=growth_inputs + ['Population'])

    X_growth = growth_df[growth_inputs]
    y_growth = growth_df['Population']

    X_g_train, X_g_test, y_g_train, y_g_test = train_test_split(X_growth, y_growth, test_size=0.15, random_state=42)

    print(f"Training XGBRegressor on {len(X_g_train)} records...")
    growth_predictor = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    growth_predictor.fit(X_g_train, y_g_train)

    # Evaluation
    g_preds = growth_predictor.predict(X_g_test)
    r2 = r2_score(y_g_test, g_preds)
    rmse = np.sqrt(mean_squared_error(y_g_test, g_preds))
    print(f"-- Algae Growth Model R2 Score: {r2:.4f} | RMSE: {rmse:.2f}")

    growth_model_file = os.path.join(MODELS_DIR, "algae_growth_model.pkl")
    joblib.dump(growth_predictor, growth_model_file)
    print(f"-- Saved model to: {growth_model_file}")

    # 2. Water Quality Model
    pond_candidates = [
        os.path.join(DATA_RAW, "Pondsdata_2.csv"),
        os.path.join(DATA_RAW, "Pondsdata.csv"),
        os.path.join(BASE_DIR, "Pondsdata.csv")
    ]
    pond_path = resolve_csv(pond_candidates)
    print(f"\n[2/2] Loading Pond Water Quality data from: {os.path.basename(pond_path)}")
    pond_df = pd.read_csv(pond_path)
    pond_df.columns = [c.strip() for c in pond_df.columns]

    pond_inputs = ['NITRATE(PPM)', 'PH', 'AMMONIA(mg/l)', 'TEMP', 'DO', 'TURBIDITY', 'MANGANESE(mg/l)']
    for col in pond_inputs:
        if col not in pond_df.columns:
            raise KeyError(f"Missing required column '{col}' in {pond_path}. Available: {pond_df.columns.tolist()}")
        pond_df[col] = pd.to_numeric(pond_df[col], errors='coerce')

    pond_df = pond_df.dropna(subset=pond_inputs + ['label_3class'])

    # Standardize label_3class mapping
    label_raw = pond_df['label_3class'].astype(str).str.replace('.0', '', regex=False)
    status_map = {
        '0': 'Healthy / Optimal Water',
        '1': 'Moderate Stress (Caution)',
        '2': 'Critical Risk (Eutrophic / Hypoxic)'
    }
    pond_df['status_label'] = label_raw.map(lambda x: status_map.get(x, f"Class {x}"))

    encoder = LabelEncoder()
    pond_df['encoded_status'] = encoder.fit_transform(pond_df['status_label'])

    X_pond = pond_df[pond_inputs]
    y_pond = pond_df['encoded_status']

    X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X_pond, y_pond, test_size=0.15, random_state=42, stratify=y_pond)

    print(f"Training XGBClassifier on {len(X_p_train)} records...")
    water_classifier = XGBClassifier(
        n_estimators=80,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.85,
        random_state=42
    )
    water_classifier.fit(X_p_train, y_p_train)

    # Evaluation
    p_preds = water_classifier.predict(X_p_test)
    acc = accuracy_score(y_p_test, p_preds)
    print(f"-- Water Quality Model Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_p_test, p_preds, target_names=encoder.classes_))

    water_model_file = os.path.join(MODELS_DIR, "water_quality_model.pkl")
    encoder_file = os.path.join(MODELS_DIR, "label_encoder.pkl")
    joblib.dump(water_classifier, water_model_file)
    joblib.dump(encoder, encoder_file)
    print(f"-- Saved classifier to: {water_model_file}")
    print(f"-- Saved label encoder to: {encoder_file}")

    print("\n" + "=" * 60)
    print("SUCCESS: All models trained and saved securely!")
    print("=" * 60)

if __name__ == "__main__":
    train_all_models()
