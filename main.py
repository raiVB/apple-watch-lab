import os
import math
import time
import secrets
import tempfile
import threading
import traceback
from datetime import datetime, timedelta

import certifi
import numpy as np
import pandas as pd
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from pymongo import MongoClient

import analyze_health
import ingest

app = FastAPI()

class CheckInRecord(BaseModel):
    stress: Optional[int] = None
    arousal: Optional[int] = None
    valence: Optional[str] = None
    context: Optional[str] = None
    skipped: bool = False
    client_timezone_offset: Optional[int] = None

# ------------------------------------------------------------------------------
# Render environment variables
#   MONGO_URI       required
#   WEBHOOK_TOKEN   optional. If set, every URL needs ?token=<value>
#   DASH_DAYS       optional, default 90. How many days the dashboard loads (keeps RAM low)
# ------------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("Set MONGO_URI in the Render Environment tab.")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN")
DASH_DAYS = int(os.getenv("DASH_DAYS", "90"))

client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=8000)
db = client.biolab

LOCK = threading.Lock()  # only one upload is processed at a time, so RAM never stacks up
STATUS = {"last_upload": None, "ok": None, "detail": None, "seconds": None}
INDEXES = {"ready": False, "error": None}


def ensure_indexes_once():
    if INDEXES["ready"]:
        return
    try:
        ingest.ensure_indexes(db)
        INDEXES.update(ready=True, error=None)
    except Exception as e:
        INDEXES["error"] = str(e)
        print("INDEX PROBLEM:", e)


ensure_indexes_once()


def check_token(request: Request):
    if WEBHOOK_TOKEN and not secrets.compare_digest(request.query_params.get("token", ""), WEBHOOK_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")


def clean_nans(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nans(x) for x in obj]
    return obj


def process_file(path: str, size: int):
    """Runs after the HTTP response is sent. Streams the saved file, never loads it whole."""
    with LOCK:
        t0 = time.time()
        try:
            ensure_indexes_once()
            ing = ingest.Ingestor()
            ingest.ingest_file(path, ing)
            detail = ingest.write(db, ing)
            detail["upload_mb"] = round(size / 1024 / 1024, 1)
            STATUS.update(ok=True, detail=detail)
        except Exception as e:
            traceback.print_exc()
            STATUS.update(ok=False, detail=repr(e))
        finally:
            STATUS.update(last_upload=datetime.utcnow().isoformat(timespec="seconds") + "Z", seconds=round(time.time() - t0, 1))
            try:
                os.remove(path)
            except OSError:
                pass


@app.post("/webhook")
async def receive_health_data(request: Request, background: BackgroundTasks):
    check_token(request)
    # 1) Save the upload to disk piece by piece (flat memory, however big it is).
    fd, path = tempfile.mkstemp(suffix=".json")
    size = 0
    with os.fdopen(fd, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            size += len(chunk)
    # 2) Answer the iPhone right away; the parsing happens in the background.
    background.add_task(process_file, path, size)
    return {"status": "accepted", "bytes": size}


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/checkin")
def save_checkin(record: CheckInRecord, request: Request):
    """
    Persists momentary ecological momentary assessment (EMA) check-ins.
    A skipped check-in explicitly prevents manufacturing a 'zero' stress label.
    """
    check_token(request)
    doc = record.model_dump()
    doc["server_timestamp"] = datetime.utcnow().isoformat() + "Z"
    doc["schema_version"] = "1.0"
    
    # Do not treat 'skipped' as a physiological baseline of 0
    if doc["skipped"]:
        doc["stress"] = None
        doc["arousal"] = None
        doc["valence"] = None
        doc["context"] = None
        
    db.psych_checkins.insert_one(doc)
    return {"status": "recorded", "skipped": doc["skipped"]}


@app.get("/stats")

@app.get("/api/records")
def get_records(
    request: Request,
    page: int = 1,
    limit: int = 100,
    metric: str = "",
    start: str = "",
    end: str = "",
):
    check_token(request)

    page = max(1, page)
    limit = min(max(1, limit), 500)

    query = {}
    if metric:
        query["metric"] = metric
    if start:
        query["date"] = {"$gte": start}
    if end:
        query.setdefault("date", {})["$lte"] = end

    total = db.records.count_documents(query)
    skip = (page - 1) * limit

    rows = list(
        db.records.find(query, {"_id": 0})
        .sort([("timestamp", -1), ("metric", 1)])
        .skip(skip)
        .limit(limit)
    )

    # Summarize the time gaps for each metric in the returned page.
    previous = {}
    for row in reversed(rows):
        metric_name = row.get("metric", "Unknown")
        ts = row.get("timestamp", "")
        if metric_name in previous:
            row["gap_minutes"] = round(
                (datetime.strptime(previous[metric_name], "%Y-%m-%d %H:%M:%S")
                 - datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                ).total_seconds() / 60, 1
            )
        else:
            row["gap_minutes"] = None
        previous[metric_name] = ts

    metrics = db.records.distinct("metric")
    return {
        "page": page,
        "limit": limit,
        "total": total,
        "pages": (total + limit - 1) // limit,
        "metrics": sorted(metrics),
        "rows": rows,
        "data_type": "stored_5_minute_bins",
    }

def stats(request: Request):
    check_token(request)
    db_stats = db.command("dbstats")
    return {
        "last_ingest": STATUS,
        "indexes": INDEXES,
        "records": db.records.estimated_document_count(),
        "sleep_rows": db.sleep.estimated_document_count(),
        "sleep_raw_rows": db.sleep_raw.estimated_document_count(),
        "storage_mb": round(db_stats.get("storageSize", 0) / 1024 / 1024, 1),
    }


def _waiting_page(msg):
    return ("<html><body style='background:#0b0f19;color:#38bdf8;font-family:sans-serif;padding:32px;'>"
            "<h2>Biometric Cloud Engine Ready</h2>"
            f"<p style='color:#94a3b8;'>{msg}</p>"
            f"<p style='color:#10b981;'>Last upload: {STATUS['last_upload']} (ok: {STATUS['ok']})</p>"
            "</body></html>")


@app.get("/", response_class=HTMLResponse)
def serve_cockpit(request: Request):
    check_token(request)
    latest = db.records.find_one({}, {"date": 1, "_id": 0}, sort=[("date", -1)])
    if not latest:
        return _waiting_page("No data yet. Send an export from your iPhone.")

    # Only the most recent DASH_DAYS days are loaded, so RAM stays small however big the database grows.
    cutoff = (datetime.strptime(latest["date"], "%Y-%m-%d") - timedelta(days=DASH_DAYS)).strftime("%Y-%m-%d")
    recs = list(db.records.find({"date": {"$gte": cutoff}}, {"_id": 0}))
    if len(recs) < 5:
        return _waiting_page(f"Only {len(recs)} records so far.")
    df_raw = pd.DataFrame(recs).sort_values("timestamp", kind="stable").reset_index(drop=True)
    del recs

    sleep_rows = list(db.sleep.find({"date": {"$gte": cutoff}}, {"_id": 0}))
    df_sleep = pd.DataFrame(sleep_rows).drop_duplicates() if sleep_rows else pd.DataFrame(
        columns=["date", "stage", "hours", "onset_hr", "wake_hr", "start_epoch", "end_epoch"])
    raw_rows = list(db.sleep_raw.find({"d": {"$gte": cutoff}}, {"_id": 0}))
    df_raw_sleep = pd.DataFrame(raw_rows) if raw_rows else None

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

@app.get("/api/backtest/readiness")
def backtest_readiness(request: Request):
    check_token(request)

    # Count completed check-ins with valid stress labels.
    checkins = list(
        db.psych_checkins.find(
            {"skipped": False, "stress": {"$gte": 0, "$lte": 4}},
            {"_id": 0, "stress": 1, "server_timestamp": 1}
        ).sort("server_timestamp", 1)
    )

    # Inspect the stored heart-rate date range.
    first_hr = db.records.find_one(
        {"metric": "HeartRate", "date": {"$exists": True}},
        {"_id": 0, "date": 1},
        sort=[("date", 1)]
    )
    last_hr = db.records.find_one(
        {"metric": "HeartRate", "date": {"$exists": True}},
        {"_id": 0, "date": 1},
        sort=[("date", -1)]
    )

    return {
        "status": "insufficient_data" if len(checkins) < 10 else "data_available_for_review",
        "completed_stress_checkins": len(checkins),
        "minimum_checkins_for_initial_review": 10,
        "heart_rate_first_date": first_hr.get("date") if first_hr else None,
        "heart_rate_last_date": last_hr.get("date") if last_hr else None,
        "checkins": [
            {
                "timestamp": c.get("server_timestamp"),
                "stress": c.get("stress")
            }
            for c in checkins
        ],
        "note": (
            "This endpoint checks data readiness only. "
            "It does not calculate predictive accuracy or prove model validity."
        )
    }
