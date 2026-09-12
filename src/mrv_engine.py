import numpy as np

def compute_continuous_mrv_index(metrics, water_model, encoder, water_input_df, predicted_biomass, vision_result=None):
    """
    Computes a scientific, continuous Measurement, Reporting, and Verification (MRV)
    Verification Index (0.0% - 100.0%) based on:
    1. Dissolved Oxygen & Hypoxia Safety
    2. pH & Inorganic Carbon Chemical Equilibrium
    3. Photosynthetic Temperature Kinetics
    4. Ammonia & Heavy Metal Toxicity Penalties
    5. Machine Learning XGBClassifier Probabilistic Confidence
    6. Chlorophyll & Nutrient Balance
    """
    temp = float(metrics.get("Temperature", 26.0))
    ph = float(metrics.get("pH", 7.5))
    do = float(metrics.get("DO", 7.5))
    ammonia = float(metrics.get("Ammonia", 0.02))
    nitrate = float(metrics.get("Nitrate", 18.0))
    co2 = float(metrics.get("CO2", 450.0))
    light = float(metrics.get("Light", 5000))
    turbidity = float(metrics.get("Turbidity", 15.0))

    # 1. Dissolved Oxygen Sub-score (Sigmoidal aerobic threshold)
    # Algae dark respiration requires DO > 4.0. Optimal: 6.5 - 12.0 mg/L
    if do >= 6.5:
        do_score = min(100.0, 95.0 + (min(do, 10.0) - 6.5) * 1.4)
    elif do >= 4.0:
        do_score = 65.0 + (do - 4.0) * 12.0 # 65% to 95%
    else:
        # Hypoxia critical crash zone
        do_score = max(5.0, (do / 4.0) * 65.0)

    # 2. pH Equilibrium Sub-score (Gaussian centered around 7.8)
    # Algal carbonic anhydrase functions best at pH 7.4 - 8.2
    ph_score = 100.0 * np.exp(-0.5 * ((ph - 7.8) / 0.85) ** 2)

    # 3. Temperature Kinetics (Optimal 26.5°C)
    # Enzyme denaturation above 35°C; metabolic arrest below 15°C
    temp_score = 100.0 * np.exp(-0.5 * ((temp - 26.5) / 5.2) ** 2)

    # 4. Toxicity Penalty: Ammonia & Turbidity
    # Un-ionized ammonia > 0.05 mg/L is lethal to microalgae monocultures
    if ammonia <= 0.03:
        ammonia_score = 100.0
    elif ammonia <= 0.10:
        ammonia_score = max(40.0, 100.0 - (ammonia - 0.03) * 800.0)
    else:
        ammonia_score = max(5.0, 40.0 * np.exp(-12.0 * (ammonia - 0.10)))

    # Turbidity light penetration
    turb_penalty = max(0.0, (turbidity - 30.0) * 0.8)
    ammonia_score = max(5.0, ammonia_score - turb_penalty)

    # 5. Photosynthetic Saturation (Light & CO2)
    light_factor = min(100.0, (light / (light + 1400.0)) * 128.0)
    co2_factor = min(100.0, max(20.0, (co2 / 600.0) * 95.0))
    photo_score = 0.6 * light_factor + 0.4 * co2_factor

    # 6. Machine Learning Class Probabilities
    ml_confidence_score = 75.0
    class_probs = {}
    if water_model is not None and encoder is not None:
        try:
            probs = water_model.predict_proba(water_input_df)[0]
            for idx, c_name in enumerate(encoder.classes_):
                class_probs[c_name] = float(probs[idx])
            
            # Weighted expected confidence
            # Healthy: 100%, Moderate: 60%, Critical: 15%
            ml_confidence_score = 0.0
            for c_name, p in class_probs.items():
                if "Healthy" in c_name:
                    ml_confidence_score += p * 100.0
                elif "Moderate" in c_name:
                    ml_confidence_score += p * 62.0
                else:
                    ml_confidence_score += p * 18.0
        except Exception:
            ml_confidence_score = 70.0

    # 7. Optical / Vision Saturation boost
    vision_score = 80.0
    if vision_result:
        v_lower = str(vision_result).lower()
        if "high density" in v_lower or "optimal" in v_lower or "emerald" in v_lower:
            vision_score = 95.0
        elif "moderate" in v_lower or "healthy" in v_lower:
            vision_score = 80.0
        elif "low" in v_lower or "turbid" in v_lower or "dilute" in v_lower:
            vision_score = 50.0

    # 8. Multi-Criteria Synthesized MRV Index
    # Weights:
    # 25% DO Health, 20% pH Stability, 15% Temp Kinetics, 15% Toxicity/Ammonia, 15% ML Classifier, 10% Photo/Optical
    raw_mrv = (
        0.25 * do_score +
        0.20 * ph_score +
        0.15 * temp_score +
        0.15 * ammonia_score +
        0.15 * ml_confidence_score +
        0.10 * ((photo_score + vision_score) / 2.0)
    )

    # Clean clamp
    final_mrv = max(10.0, min(98.5, float(raw_mrv)))

    # Detailed breakdown for dashboard transparency
    breakdown = {
        "final_mrv": round(final_mrv, 1),
        "do_score": round(float(do_score), 1),
        "ph_score": round(float(ph_score), 1),
        "temp_score": round(float(temp_score), 1),
        "ammonia_score": round(float(ammonia_score), 1),
        "photo_score": round(float(photo_score), 1),
        "ml_confidence_score": round(float(ml_confidence_score), 1),
        "class_probs": class_probs
    }
    return breakdown


def compute_carbon_financials(predicted_biomass, mrv_index, carbon_price_per_ton=42.50):
    """
    Computes Carbon Sequestration Chemistry and Financial Credit Valuations.
    
    Chemistry benchmark:
    - 1.0 kg of dry microalgae biomass fixes ~1.83 kg of CO2 via photosynthesis.
    - 1 Carbon Credit = 1 Metric Tonne (MT) of verified CO2 equivalent removed.
    
    Finance benchmark:
    - Carbon credit benchmark: $42.50 USD / MT CO2e (Voluntary Carbon Market standard).
    - MRV Discount: Registries (Verra VM0042) require an uncertainty buffer discount.
      Net Certified Tradable Value = Gross Value * (MRV Index / 100).
    """
    gross_co2_kg = float(predicted_biomass) * 1.83
    gross_co2_tonnes = gross_co2_kg / 1000.0
    daily_capture_rate_kg = gross_co2_kg * 0.35 # ~35% specific daily growth rate

    gross_credit_usd = gross_co2_tonnes * carbon_price_per_ton
    mrv_discount_factor = float(mrv_index) / 100.0
    certified_tradable_usd = gross_credit_usd * mrv_discount_factor

    return {
        "gross_co2_kg": gross_co2_kg,
        "gross_co2_tonnes": gross_co2_tonnes,
        "daily_capture_rate_kg": daily_capture_rate_kg,
        "carbon_price_per_ton": carbon_price_per_ton,
        "gross_credit_usd": gross_credit_usd,
        "certified_tradable_usd": certified_tradable_usd,
        "mrv_discount_factor": mrv_discount_factor
    }
