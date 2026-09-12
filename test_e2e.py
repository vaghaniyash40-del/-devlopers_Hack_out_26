import os
import joblib
import pandas as pd
from src.telemetry_engine import TelemetryEngine
from src.api_services import inspect_image_with_groq
from src.fusion_layer import generate_audit_report

def run_tests():
    print("=== STARTING END-TO-END VERIFICATION ===")
    
    # 1. Models check
    print("[1] Loading models...")
    g_model = joblib.load("models/algae_growth_model.pkl")
    w_model = joblib.load("models/water_quality_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    print("    Models loaded successfully.")
    
    # 2. Telemetry Engine check
    print("[2] Testing Telemetry Engine...")
    engine = TelemetryEngine()
    stations = engine.get_stations()
    print(f"    Available stations: {stations}")
    tick = engine.fetch_live_iot_tick(station="Station 1")
    print(f"    Station 1 tick: {tick['timestamp']} -> Temp: {tick['metrics']['Temperature']} C, pH: {tick['metrics']['pH']}, DO: {tick['metrics']['DO']}")
    
    # Weather check
    w = engine.fetch_live_weather()
    print(f"    Live Open-Meteo Weather available: {w.get('available')} (Temp: {w.get('temperature')} C, Solar Light: {w.get('light')} Lux)")
    
    # 3. Manual Entry Inference Simulation
    print("[3] Simulating Manual Entry Mode Inference...")
    manual_params = engine.get_preset_scenario("Optimal Growth Bio-Reactor")
    m_growth_df = pd.DataFrame([[
        manual_params['Light'], manual_params['Nitrate'], manual_params['Iron'], manual_params['Phosphate'],
        manual_params['Temperature'], manual_params['pH'], manual_params['CO2']
    ]], columns=['Light', 'Nitrate', 'Iron', 'Phosphate', 'Temperature', 'pH', 'CO2'])
    manual_biomass = float(g_model.predict(m_growth_df)[0])
    
    m_water_df = pd.DataFrame([[
        manual_params['Nitrate'], manual_params['pH'], manual_params['Ammonia'],
        manual_params['Temperature'], manual_params['DO'], manual_params['Turbidity'], manual_params['Manganese']
    ]], columns=['NITRATE(PPM)', 'PH', 'AMMONIA(mg/l)', 'TEMP', 'DO', 'TURBIDITY', 'MANGANESE(mg/l)'])
    manual_water_status = encoder.inverse_transform([w_model.predict(m_water_df)[0]])[0]
    print(f"    Optimal Preset -> Predicted Biomass: {manual_biomass:.2f} cells/mL | Water Status: {manual_water_status}")
    print(f"    Gross CO2 Uptake: {manual_biomass * 1.83:.2f} kg CO2")
    
    # 4. Real-Time Telemetry Inference Simulation
    print("[4] Simulating Real-Time IoT Telemetry Stream Inference...")
    t_m = tick['metrics']
    rt_growth_df = pd.DataFrame([[
        t_m['Light'], t_m['Nitrate'], t_m['Iron'], t_m['Phosphate'],
        t_m['Temperature'], t_m['pH'], t_m['CO2']
    ]], columns=['Light', 'Nitrate', 'Iron', 'Phosphate', 'Temperature', 'pH', 'CO2'])
    rt_biomass = float(g_model.predict(rt_growth_df)[0])
    
    rt_water_df = pd.DataFrame([[
        t_m['Nitrate'], t_m['pH'], t_m['Ammonia'],
        t_m['Temperature'], t_m['DO'], t_m['Turbidity'], t_m['Manganese']
    ]], columns=['NITRATE(PPM)', 'PH', 'AMMONIA(mg/l)', 'TEMP', 'DO', 'TURBIDITY', 'MANGANESE(mg/l)'])
    rt_water_status = encoder.inverse_transform([w_model.predict(rt_water_df)[0]])[0]
    print(f"    Live IoT Tick -> Predicted Biomass: {rt_biomass:.2f} cells/mL | Water Status: {rt_water_status}")
    
    # 5. Image & Vision Inspection
    print("[5] Testing Vision Inspection...")
    sample_img = "data/images/sample_healthy_spirulina.jpg"
    vision_assessment = inspect_image_with_groq(sample_img)
    print("    Vision Assessment output:")
    for line in vision_assessment.splitlines():
        print(f"      {line}")
        
    # 6. Carbon Auditor Fusion
    print("[6] Testing Carbon Auditor Synthesis...")
    audit = generate_audit_report(rt_biomass, rt_water_status, vision_assessment, t_m)
    print("    Audit report generated successfully (length: " + str(len(audit)) + " characters).")
    
    print("=== ALL SYSTEM TESTS PASSED PERFECTLY! ===")

if __name__ == "__main__":
    run_tests()
