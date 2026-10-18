from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import math
import os
import certifi
from pymongo import MongoClient, UpdateOne
import analyze_health

app = FastAPI()

# Connect to MongoDB if MONGO_URI is set in Render Environment
MONGO_URI = os.getenv("MONGO_URI")
if MONGO_URI:
    # Uses certifi to fix the SSL handshake crash, and a 5s timeout to prevent freezing
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
    db = client.biolab
    use_db = True
else:
    use_db = False
    stored_records = []
    stored_sleep = []
    stored_sleep_raw = []

METRIC_MAP = {
    "heart_rate": "HeartRate", "heartrate": "HeartRate",
    "heart_rate_variability": "HeartRateVariabilitySDNN", "heartratevariability": "HeartRateVariabilitySDNN",
    "resting_heart_rate": "RestingHeartRate", "restingheartrate": "RestingHeartRate",
    "step_count": "StepCount", "stepcount": "StepCount",
    "walking_running_distance": "DistanceWalkingRunning", "distance_walking_running": "DistanceWalkingRunning",
    "apple_exercise_time": "AppleExerciseTime", "exercise_time": "AppleExerciseTime",
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
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    if isinstance(obj, dict): return {k: clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean_nans(x) for x in obj]
    return obj

@app.post("/webhook")
async def receive_health_data(request: Request):
    payload = await request.json()
    data_block = payload.get("data", payload)
    metrics = data_block.get("metrics", [])

    new_records, new_sleep, new_sleep_raw = [], [], []

    for group in metrics:
        raw_name = str(group.get("name", "")).strip()
        clean_key = raw_name.lower().replace(" ", "_")
        unit = str(group.get("units", ""))

        if "sleep" in clean_key:
            for item in group.get("data", []):
                new_sleep_raw.append(item)
                
                if any(k in item for k in ("deep", "rem", "core", "asleep")):
                    d_str = str(item.get("date") or item.get("sleepEnd") or "")[:10]
                    s_start = str(item.get("sleepStart") or f"{d_str} 00:30:00")[:19]
                    s_end = str(item.get("sleepEnd") or f"{d_str} 07:30:00")[:19]
                    try:
                        s_dt, e_dt = pd.to_datetime(s_start), pd.to_datetime(s_end)
                        s_hr = s_dt.hour + s_dt.minute / 60.0
                        if s_hr < 14.0: s_hr += 24.0
                        e_hr = e_dt.hour + e_dt.minute / 60.0
                        s_ep, e_ep = int(s_dt.timestamp() // 300), int(e_dt.timestamp() // 300)

                        for stg_key, stg_label in [("deep", "Deep Sleep (Body Repair)"), ("rem", "REM Sleep (Brain & Memory)"), ("core", "Core Sleep (Light)"), ("awake", "Awake in Bed")]:
                            hrs = float(item.get(stg_key) or 0.0)
                            if hrs > 16.0: hrs /= 60.0
                            if hrs > 0:
                                new_sleep.append({
                                    "date": d_str, "stage": stg_label, "hours": hrs,
                                    "onset_hr": s_hr, "wake_hr": e_hr,
                                    "start_epoch": s_ep, "end_epoch": e_ep,
                                })
                    except Exception:
                        continue
                else:
                    s_str = str(item.get("startDate") or item.get("date") or "")[:19]
                    e_str = str(item.get("endDate") or s_str)[:19]
                    raw_stage = str(item.get("value") or item.get("stage") or "core").lower()
                    stg_label = next((v for k, v in STAGE_MAP.items() if k in raw_stage), "Core Sleep (Light)")
                    try:
                        s_dt, e_dt = pd.to_datetime(s_str), pd.to_datetime(e_str)
                        dur = (e_dt - s_dt).total_seconds() / 3600.0
                        if dur <= 0: dur = float(item.get("qty") or 0.0)
                        if 0 < dur < 16:
                            s_hr = s_dt.hour + s_dt.minute / 60.0
                            if s_hr < 14.0: s_hr += 24.0
                            e_hr = e_dt.hour + e_dt.minute / 60.0
                            new_sleep.append({
                                "date": e_str[:10], "stage": stg_label, "hours": dur,
                                "onset_hr": s_hr, "wake_hr": e_hr,
                                "start_epoch": int(s_dt.timestamp() // 300),
                                "end_epoch": int(e_dt.timestamp() // 300),
                            })
                    except Exception:
                        continue
            continue

        target_metric = METRIC_MAP.get(clean_key, raw_name)
        for item in group.get("data", []):
            val = next((item.get(k) for k in ("qty", "Avg", "avg", "value", "Value", "Min", "min") if item.get(k) is not None), None)
            date_str = item.get("date") or item.get("startDate")
            if val is not None and date_str:
                try:
                    val = float(val)
                    dt = pd.to_datetime(str(date_str)[:19])
                    new_records.append({
                        "metric": target_metric, "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "date": dt.strftime("%Y-%m-%d"), "hour": round(dt.hour + dt.minute / 60.0, 2),
                        "epoch_5m": int(dt.timestamp() // 300), "value": val, "unit": unit,
                    })
                except Exception:
                    continue

    if use_db:
        if new_records:
            db.records.bulk_write([UpdateOne({"metric": r["metric"], "timestamp": r["timestamp"]}, {"$set": r}, upsert=True) for r in new_records], ordered=False)
        if new_sleep:
            db.sleep.bulk_write([UpdateOne({"date": r["date"], "stage": r["stage"], "start_epoch": r["start_epoch"]}, {"$set": r}, upsert=True) for r in new_sleep], ordered=False)
        if new_sleep_raw:
            db.sleep_raw.bulk_write([UpdateOne({"startDate": r.get("startDate", r.get("start")), "value": r.get("value")}, {"$set": r}, upsert=True) for r in new_sleep_raw if r.get("startDate") or r.get("start")], ordered=False)
    else:
        stored_records.extend(new_records)
        stored_sleep.extend(new_sleep)
        stored_sleep_raw.extend(new_sleep_raw)

    return {"status": "success", "records_processed": len(new_records), "sleep_processed": len(new_sleep)}

@app.get("/", response_class=HTMLResponse)
def serve_cockpit():  # Removed async to prevent the server from hanging
    if use_db:
        all_records = list(db.records.find({}, {"_id": 0}))
        all_sleep = list(db.sleep.find({}, {"_id": 0}))
        all_sleep_raw = list(db.sleep_raw.find({}, {"_id": 0}))
    else:
        all_records, all_sleep, all_sleep_raw = stored_records, stored_sleep, stored_sleep_raw

    if len(all_records) < 5:
        return (
            "<html><body style='background:#0b0f19;color:#38bdf8;font-family:sans-serif;padding:32px;'>"
            "<h2>Biometric Cloud Engine Ready</h2>"
            f"<p style='color:#10b981;'>Database Active: {use_db}</p>"
            "<p style='color:#94a3b8;'>Please push an initial sync from your iPhone.</p>"
            "</body></html>"
        )

    df_raw = pd.DataFrame(all_records).drop_duplicates(subset=["metric", "timestamp", "value"])
    df_sleep = pd.DataFrame(all_sleep).drop_duplicates() if all_sleep else pd.DataFrame(columns=["date", "stage", "hours", "onset_hr", "wake_hr", "start_epoch", "end_epoch"])
    df_raw_sleep = pd.DataFrame(all_sleep_raw) if all_sleep_raw else None

    log_df = pd.DataFrame(columns=[
        "date", "energy_morning", "energy_afternoon", "energy_evening",
        "study_hours", "productivity_rating", "best_focus_window",
        "stress_rating", "mood_rating", "workload_rating", "caffeine_cups", "calendar_event",
    ])

    stress_bundle = analyze_health.compute_stress_detective_engine(df_raw, df_sleep, [], log_df)
    pred_bundle = analyze_health.compute_energy_predictor_engine(df_raw, df_sleep, log_df)
    sleep_bundle = analyze_health.compute_sleep_experiment_lab_engine(df_raw, df_sleep, log_df, stress_bundle, pred_bundle)

    sleep_bundle["diary_rows"] = clean_nans(sleep_bundle["diary_rows"])

    analyze_health.build_dashboard(df_raw, df_sleep, log_df, stress_bundle, pred_bundle, sleep_bundle, raw_sleep=df_raw_sleep)

    with open("biometric_science_lab.html", "r", encoding="utf-8") as f:
        return f.read()
