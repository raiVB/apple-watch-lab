from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import math
import analyze_health

app = FastAPI()

stored_records = []
stored_sleep = []
stored_sleep_raw = []  # Added: Stores pure Apple Watch JSON for the hypnogram

METRIC_MAP = {
    "heart_rate": "HeartRate",
    "heartrate": "HeartRate",
    "heart_rate_variability": "HeartRateVariabilitySDNN",
    "heartratevariability": "HeartRateVariabilitySDNN",
    "resting_heart_rate": "RestingHeartRate",
    "restingheartrate": "RestingHeartRate",
    "step_count": "StepCount",
    "stepcount": "StepCount",
    "walking_running_distance": "DistanceWalkingRunning",
    "distance_walking_running": "DistanceWalkingRunning",
    "active_energy": "ActiveEnergyBurned",
    "active_energy_burned": "ActiveEnergyBurned",
    "basal_energy_burned": "BasalEnergyBurned",
    "resting_energy": "BasalEnergyBurned",
    "apple_exercise_time": "AppleExerciseTime",
    "exercise_time": "AppleExerciseTime",
    "apple_stand_time": "AppleStandTime",
    "stand_time": "AppleStandTime",
    "flights_climbed": "FlightsClimbed",
    "respiratory_rate": "RespiratoryRate",
    "walking_speed": "WalkingSpeed",
    "walking_step_length": "WalkingStepLength",
    "walking_double_support_percentage": "WalkingDoubleSupportPercentage",
    "walking_asymmetry_percentage": "WalkingAsymmetryPercentage",
    "apple_walking_steadiness": "AppleWalkingSteadiness",
    "walking_heart_rate_average": "WalkingHeartRateAverage",
    "environmental_audio_exposure": "EnvironmentalAudioExposure",
    "headphone_audio_exposure": "HeadphoneAudioExposure",
    "time_in_daylight": "TimeInDaylight",
    "blood_oxygen_saturation": "OxygenSaturation",
    "oxygen_saturation": "OxygenSaturation",
    "apple_sleeping_wrist_temperature": "AppleSleepingWristTemperature",
    "wrist_temperature": "AppleSleepingWristTemperature",
    "vo2_max": "VO2Max",
    "heart_rate_recovery": "HeartRateRecoveryOneMinute",
    "heart_rate_recovery_one_minute": "HeartRateRecoveryOneMinute",
}

STAGE_MAP = {
    "deep": "Deep Sleep (Body Repair)",
    "rem": "REM Sleep (Brain & Memory)",
    "core": "Core Sleep (Light)",
    "asleep": "Core Sleep (Light)",
    "unspecified": "Core Sleep (Light)",
    "awake": "Awake in Bed",
}

def clean_nans(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nans(x) for x in obj]
    return obj

@app.post("/webhook")
async def receive_health_data(request: Request):
    payload = await request.json()
    data_block = payload.get("data", payload)
    metrics = data_block.get("metrics", [])

    added_records = 0
    added_sleep = 0

    for group in metrics:
        raw_name = str(group.get("name", "")).strip()
        clean_key = raw_name.lower().replace(" ", "_")
        unit = str(group.get("units", ""))

        if "sleep" in clean_key:
            for item in group.get("data", []):
                # Save pure untouched JSON for the new dashboard builder
                stored_sleep_raw.append(item)
                
                # Maintain the old format for the Stress and Energy backend engines
                if any(k in item for k in ("deep", "rem", "core", "asleep")):
                    d_str = str(item.get("date") or item.get("sleepEnd") or "")[:10]
                    s_start = str(item.get("sleepStart") or f"{d_str} 00:30:00")[:19]
                    s_end = str(item.get("sleepEnd") or f"{d_str} 07:30:00")[:19]
                    try:
                        s_dt = pd.to_datetime(s_start)
                        e_dt = pd.to_datetime(s_end)
                        s_hr = s_dt.hour + s_dt.minute / 60.0
                        if s_hr < 14.0:
                            s_hr += 24.0
                        e_hr = e_dt.hour + e_dt.minute / 60.0
                        s_ep = int(s_dt.timestamp() // 300)
                        e_ep = int(e_dt.timestamp() // 300)

                        stage_pairs = [
                            ("deep", "Deep Sleep (Body Repair)"),
                            ("rem", "REM Sleep (Brain & Memory)"),
                            ("core", "Core Sleep (Light)"),
                            ("awake", "Awake in Bed"),
                        ]
                        for stg_key, stg_label in stage_pairs:
                            hrs = float(item.get(stg_key) or 0.0)
                            if hrs > 16.0:
                                hrs /= 60.0
                            if hrs > 0:
                                stored_sleep.append({
                                    "date": d_str,
                                    "stage": stg_label,
                                    "hours": hrs,
                                    "onset_hr": s_hr,
                                    "wake_hr": e_hr,
                                    "start_epoch": s_ep,
                                    "end_epoch": e_ep,
                                })
                                added_sleep += 1
                    except Exception:
                        continue
                else:
                    s_str = str(item.get("startDate") or item.get("date") or "")[:19]
                    e_str = str(item.get("endDate") or s_str)[:19]
                    raw_stage = str(item.get("value") or item.get("stage") or "core").lower()
                    stg_label = next((v for k, v in STAGE_MAP.items() if k in raw_stage), "Core Sleep (Light)")
                    try:
                        s_dt = pd.to_datetime(s_str)
                        e_dt = pd.to_datetime(e_str)
                        dur = (e_dt - s_dt).total_seconds() / 3600.0
                        if dur <= 0:
                            dur = float(item.get("qty") or 0.0)
                        if 0 < dur < 16:
                            s_hr = s_dt.hour + s_dt.minute / 60.0
                            if s_hr < 14.0:
                                s_hr += 24.0
                            e_hr = e_dt.hour + e_dt.minute / 60.0
                            stored_sleep.append({
                                "date": e_str[:10],
                                "stage": stg_label,
                                "hours": dur,
                                "onset_hr": s_hr,
                                "wake_hr": e_hr,
                                "start_epoch": int(s_dt.timestamp() // 300),
                                "end_epoch": int(e_dt.timestamp() // 300),
                            })
                            added_sleep += 1
                    except Exception:
                        continue
            continue

        target_metric = METRIC_MAP.get(clean_key, raw_name)
        for item in group.get("data", []):
            val = None
            for k in ("qty", "Avg", "avg", "value", "Value", "Min", "min"):
                if item.get(k) is not None:
                    val = item.get(k)
                    break

            date_str = item.get("date") or item.get("startDate")
            if val is not None and date_str:
                try:
                    val = float(val)
                    if target_metric == "WalkingSpeed" and "m/s" in unit:
                        val *= 3.6
                    elif target_metric == "WalkingStepLength" and unit == "m":
                        val *= 100.0
                    elif target_metric == "DistanceWalkingRunning" and unit == "m":
                        val /= 1000.0
                    elif target_metric in (
                        "OxygenSaturation",
                        "WalkingAsymmetryPercentage",
                        "WalkingDoubleSupportPercentage",
                        "AppleWalkingSteadiness",
                    ) and val <= 1.0:
                        val *= 100.0

                    dt = pd.to_datetime(str(date_str)[:19])
                    stored_records.append({
                        "metric": target_metric,
                        "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "date": dt.strftime("%Y-%m-%d"),
                        "hour": round(dt.hour + dt.minute / 60.0, 2),
                        "epoch_5m": int(dt.timestamp() // 300),
                        "value": val,
                        "unit": unit,
                    })
                    added_records += 1
                except Exception:
                    continue

    return {
        "status": "success",
        "records_added": added_records,
        "sleep_added": added_sleep,
        "total": len(stored_records),
    }

@app.get("/", response_class=HTMLResponse)
async def serve_cockpit():
    if len(stored_records) < 5:
        return (
            "<html><body style='background:#0b0f19;color:#38bdf8;font-family:sans-serif;padding:32px;'>"
            "<h2>Biometric Cloud Engine Ready</h2>"
            f"<p style='color:#94a3b8;'>Records in memory: <b>{len(stored_records)}</b></p>"
            "<p style='color:#10b981;'>Trigger an export from Health Auto Export on your iPhone, then refresh this page.</p>"
            "</body></html>"
        )

    df_raw = pd.DataFrame(stored_records).drop_duplicates(subset=["metric", "timestamp", "value"])
    if stored_sleep:
        df_sleep = pd.DataFrame(stored_sleep).drop_duplicates()
    else:
        df_sleep = pd.DataFrame(columns=["date", "stage", "hours", "onset_hr", "wake_hr", "start_epoch", "end_epoch"])
        
    # Convert pure untouched JSON sleep blocks into a dataframe
    df_raw_sleep = pd.DataFrame(stored_sleep_raw) if stored_sleep_raw else None

    log_df = pd.DataFrame(columns=[
        "date", "energy_morning", "energy_afternoon", "energy_evening",
        "study_hours", "productivity_rating", "best_focus_window",
        "stress_rating", "mood_rating", "workload_rating", "caffeine_cups", "calendar_event",
        "caffeine_cutoff_hr", "screen_off_mins", "last_meal_hrs_before_bed",
        "nap_mins", "bedtime_routine", "experiment_name", "experiment_phase",
    ])

    stress_bundle = analyze_health.compute_stress_detective_engine(df_raw, df_sleep, [], log_df)
    pred_bundle = analyze_health.compute_energy_predictor_engine(df_raw, df_sleep, log_df)
    sleep_bundle = analyze_health.compute_sleep_experiment_lab_engine(
        df_raw, df_sleep, log_df, stress_bundle, pred_bundle
    )

    sleep_bundle["diary_rows"] = clean_nans(sleep_bundle["diary_rows"])

    # Pass the untouched raw_sleep data directly into the new dashboard builder
    analyze_health.build_dashboard(
        df_raw, df_sleep, log_df, stress_bundle, pred_bundle, sleep_bundle, raw_sleep=df_raw_sleep
    )

    with open("biometric_science_lab.html", "r", encoding="utf-8") as f:
        return f.read()
