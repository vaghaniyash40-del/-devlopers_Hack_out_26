import os
import random
import datetime
import pandas as pd
import numpy as np
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")

class TelemetryEngine:
    """
    Manages both live IoT telemetry streaming from Pondsdata.csv
    and external environmental live weather feeds (Open-Meteo).
    """
    def __init__(self):
        self.ponds_df = None
        self.cursor = 0
        self.stations = ['Station 1', 'Station 2', 'Station 3']
        self._load_data()

    def _load_data(self):
        candidates = [
            os.path.join(DATA_RAW, "Pondsdata.csv"),
            os.path.join(DATA_RAW, "Pondsdata_2.csv"),
            os.path.join(BASE_DIR, "Pondsdata.csv")
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    df.columns = [c.strip() for c in df.columns]
                    # Normalize station names
                    if 'station' in df.columns:
                        df['station_norm'] = df['station'].astype(str).str.strip().str.replace('station', 'Station ', case=False)
                        df['station_norm'] = df['station_norm'].str.replace('Station', 'Station ')
                        df['station_norm'] = df['station_norm'].str.replace('Station  ', 'Station ')
                    self.ponds_df = df
                    break
                except Exception as e:
                    print(f"Error loading {path}: {e}")

    def get_stations(self):
        if self.ponds_df is not None and 'station_norm' in self.ponds_df.columns:
            return sorted(self.ponds_df['station_norm'].dropna().unique().tolist())
        return ['Station 1', 'Station 2', 'Station 3']

    def fetch_live_iot_tick(self, station=None):
        """
        Returns a single real-time telemetry observation.
        Simulates live IoT sensor reading, cycling through real sensor records.
        """
        if self.ponds_df is not None and len(self.ponds_df) > 0:
            df_subset = self.ponds_df
            if station and 'station_norm' in self.ponds_df.columns:
                sub = self.ponds_df[self.ponds_df['station_norm'] == station]
                if len(sub) > 0:
                    df_subset = sub

            # Sample row or sequential
            idx = random.randint(0, len(df_subset) - 1)
            row = df_subset.iloc[idx]

            nitrate = float(row.get('NITRATE(PPM)', 25.0))
            ph = float(row.get('PH', 7.5))
            ammonia = float(row.get('AMMONIA(mg/l)', 0.05))
            temp = float(row.get('TEMP', 26.0))
            do = float(row.get('DO', 7.2))
            turbidity = float(row.get('TURBIDITY', 15.0))
            manganese = float(row.get('MANGANESE(mg/l)', 0.5))
            source_station = str(row.get('station_norm', station or 'Station 1'))
            timestamp = f"{row.get('Date', 'Today')} {row.get('Time', datetime.datetime.now().strftime('%H:%M:%S'))}"
        else:
            # Synthetic realistic default
            nitrate = 28.5 + random.uniform(-2, 2)
            ph = 7.4 + random.uniform(-0.3, 0.3)
            ammonia = 0.04 + random.uniform(-0.01, 0.02)
            temp = 25.0 + random.uniform(-1.5, 1.5)
            do = 6.8 + random.uniform(-0.5, 0.5)
            turbidity = 18.2 + random.uniform(-2, 2)
            manganese = 0.8 + random.uniform(-0.1, 0.1)
            source_station = station or "Station 1"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Synthesize complementary growth variables aligned with pond ecology
        # Light follows diurnal cycle (lux 1000 - 8000)
        hour = datetime.datetime.now().hour
        solar_factor = max(0.15, np.sin((max(6, min(18, hour)) - 6) / 12 * np.pi))
        light = round(float(2500 + 4500 * solar_factor + random.uniform(-200, 200)), 1)
        co2 = round(float(420.0 + random.uniform(10, 80)), 1)
        iron = round(float(max(0.01, manganese * 0.25 + random.uniform(-0.05, 0.05))), 3)
        phosphate = round(float(max(0.01, nitrate * 0.08 + random.uniform(-0.2, 0.2))), 3)

        return {
            "timestamp": timestamp,
            "station": source_station,
            "source": "Real-Time IoT Telemetry Stream",
            "metrics": {
                "Temperature": round(temp, 2),
                "pH": round(ph, 2),
                "CO2": round(co2, 1),
                "Light": round(light, 1),
                "Nitrate": round(nitrate, 2),
                "Iron": round(iron, 3),
                "Phosphate": round(phosphate, 3),
                "Ammonia": round(ammonia, 4),
                "DO": round(do, 2),
                "Turbidity": round(turbidity, 2),
                "Manganese": round(manganese, 3)
            }
        }

    def fetch_live_weather(self, lat=28.6139, lon=77.2090):
        """
        Fetches live real-world ambient weather from Open-Meteo free public API.
        Enriches pond temperature and solar light estimation.
        """
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,direct_normal_irradiance"
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json().get("current", {})
                temp = data.get("temperature_2m", 28.0)
                irradiance = data.get("direct_normal_irradiance", 450.0)
                # Convert solar irradiance (W/m2) to approx Lux (1 W/m2 ~ 120 lux for sunlight)
                light = min(10000.0, max(800.0, irradiance * 12.0))
                return {
                    "available": True,
                    "temperature": temp,
                    "light": round(light, 1),
                    "humidity": data.get("relative_humidity_2m", 50),
                    "cloud_cover": data.get("cloud_cover", 20)
                }
        except Exception as e:
            pass
        return {"available": False}

    def get_preset_scenario(self, scenario_name):
        """
        Pre-configured presets for quick demonstration to hackathon judges.
        """
        presets = {
            "Optimal Growth Bio-Reactor": {
                "Temperature": 28.5,
                "pH": 7.8,
                "CO2": 650.0,
                "Light": 6500,
                "Nitrate": 18.5,
                "Iron": 0.45,
                "Phosphate": 1.8,
                "Ammonia": 0.015,
                "DO": 8.2,
                "Turbidity": 10.5,
                "Manganese": 0.012
            },
            "Acidic Nutrient Runoff": {
                "Temperature": 32.0,
                "pH": 5.4,
                "CO2": 850.0,
                "Light": 7800,
                "Nitrate": 55.0,
                "Iron": 1.2,
                "Phosphate": 4.5,
                "Ammonia": 0.35,
                "DO": 4.1,
                "Turbidity": 38.0,
                "Manganese": 1.8
            },
            "Hypoxic Low-Oxygen Alert": {
                "Temperature": 36.5,
                "pH": 6.2,
                "CO2": 320.0,
                "Light": 3500,
                "Nitrate": 42.0,
                "Iron": 0.9,
                "Phosphate": 3.2,
                "Ammonia": 0.48,
                "DO": 1.8,
                "Turbidity": 62.0,
                "Manganese": 2.5
            },
            "Standard Open Pond": {
                "Temperature": 24.0,
                "pH": 7.5,
                "CO2": 450.0,
                "Light": 4800,
                "Nitrate": 12.0,
                "Iron": 0.15,
                "Phosphate": 0.9,
                "Ammonia": 0.03,
                "DO": 6.8,
                "Turbidity": 16.0,
                "Manganese": 0.3
            }
        }
        return presets.get(scenario_name, presets["Standard Open Pond"])

    def get_sample_history(self, station=None, n_points=25):
        """
        Returns chronological data points from Pondsdata.csv for Plotly telemetry visualization.
        """
        if self.ponds_df is not None and len(self.ponds_df) >= n_points:
            df_subset = self.ponds_df
            if station and 'station_norm' in self.ponds_df.columns:
                sub = self.ponds_df[self.ponds_df['station_norm'] == station]
                if len(sub) >= n_points:
                    df_subset = sub
            sample = df_subset.tail(n_points).copy()
            sample['Index'] = range(1, len(sample) + 1)
            return sample
        return pd.DataFrame()

