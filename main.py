from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import os
from datetime import datetime
import analyze_health

app = FastAPI()

# In-memory store for incoming telemetry
stored_records = []

# Normalized mapping from iOS Health Auto Export keys to model parameter names
METRIC_MAP = {
    "heart_rate": "HeartRate",
    "heartrate": "HeartRate",
    "heart_rate_variability": "HeartRateVariabilitySDNN",
    "heartratevariability": "HeartRateVariabilitySDNN",
    "resting_heart_rate": "RestingHeartRate",
    "restingheartrate": "RestingHeartRate",
    "step_count": "StepCount",
    "stepcount": "StepCount",
    "active_energy": "ActiveEnergyBurned",
    "activeenergyburned": "ActiveEnergyBurned",
    "basal_energy_burned": "BasalEnergyBurned",
    "apple_exercise_time": "AppleExerciseTime",
    "flights_climbed": "FlightsClimbed",
    "respiratory_rate": "RespiratoryRate",
    "walking_speed": "WalkingSpeed",
    "walking_step_length": "WalkingStepLength",
    "walking_double_support_percentage": "WalkingDoubleSupportPercentage",
    "walking_asymmetry_percentage": "WalkingAsymmetryPercentage",
    "apple_walking_steadiness": "AppleWalkingSteadiness",
    "environmental_audio_exposure": "EnvironmentalAudioExposure",
    "headphone_audio_exposure": "HeadphoneAudioExposure",
    "time_in_daylight": "TimeInDaylight",
    "oxygen_saturation": "OxygenSaturation",
    "apple_sleeping_wrist_temperature": "AppleSleepingWristTemperature",
    "wrist_temperature": "AppleSleepingWristTemperature",
    "vo2_max": "VO2Max",
    "heart_rate_recovery": "HeartRateRecoveryOneMinute"
}

@app.post("/webhook")
async def receive_health_data(request: Request):
    """Ingests and standardizes JSON payload from Health Auto Export."""
    payload = await request.json()
    metrics = payload.get("data", {}).get("metrics", [])
    
    count = 0
    for group in metrics:
        raw_name = group.get("name", "")
        clean_key = raw_name.lower().replace(" ", "_")
        target_metric = METRIC_MAP.get(clean_key, raw_name)
        
        for item in group.get("data", []):
            val = item.get("qty") if item.get("qty") is not None else item.get("avg")
            date_str = item.get("date")
            
            if val is not None and date_str:
                try:
                    dt = pd.to_datetime(date_str)
                    stored_records.append({
                        "metric": target_metric,
                        "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "date": dt.strftime("%Y-%m-%d"),
                        "hour": round(dt.hour + dt.minute / 60.0, 2),
                        "epoch_5m": int(dt.timestamp() // 300),
                        "value": float(val),
                        "unit": group.get("units", "")
                    })
                    count += 1
                except Exception:
                    continue
                    
    return {"status": "success", "records_ingested": count, "total": len(stored_records)}

@app.get("/", response_class=HTMLResponse)
async def serve_cockpit():
    """Executes the 3-Activity engines and serves the dashboard."""
    if len(stored_records) < 10:
        return f"""
        <html>
            <body style="background:#0b0f19; color:#38bdf8; font-family:-apple-system, sans-serif; padding: 32px;">
                <h2>⚡ Biometric Cloud Engine Live</h2>
                <p style="color:#94a3b8;">Records accumulated: <b>{len(stored_records)}</b> (Needs at least 10 readings to build matrix).</p>
                <p style="color:#10b981;">Trigger a "Manual Export" from your iPhone app to populate telemetry.</p>
            </body>
        </html>
        """

    df_raw = pd.DataFrame(stored_records).drop_duplicates(subset=["metric", "timestamp"])
    
    # Initialize sleep and workout frames (populated as more categories sync)
    df_sleep = pd.DataFrame(columns=["date", "stage", "hours", "onset_hr", "wake_hr", "start_epoch", "end_epoch"])
    workout_windows = []
    
    # Initialize empty diary DataFrame if none exists
    log_cols = [
        "date", "energy_morning", "energy_afternoon", "energy_evening",
        "study_hours", "productivity_rating", "best_focus_window",
        "stress_rating", "mood_rating", "workload_rating", "caffeine_cups", "calendar_event",
        "caffeine_cutoff_hr", "screen_off_mins", "last_meal_hrs_before_bed",
        "nap_mins", "bedtime_routine", "experiment_name", "experiment_phase"
    ]
    log_df = pd.DataFrame(columns=log_cols)

    # Run the three scientific computation modules
    stress_bundle = analyze_health.compute_stress_detective_engine(df_raw, df_sleep, workout_windows, log_df)
    pred_bundle = analyze_health.compute_energy_predictor_engine(df_raw, df_sleep, log_df)
    sleep_bundle = analyze_health.compute_sleep_experiment_lab_engine(df_raw, df_sleep, log_df, stress_bundle, pred_bundle)

    # Build and capture the HTML cockpit
    analyze_health.build_dashboard(df_raw, df_sleep, log_df, stress_bundle, pred_bundle, sleep_bundle)
    
    with open("biometric_science_lab.html", "r", encoding="utf-8") as f:
        html_content = f.read()
        
    return html_content
