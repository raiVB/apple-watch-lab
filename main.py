import os
import math
import time
import secrets
import tempfile
import threading
import traceback
from datetime import datetime, timedelta, timezone

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
import cardio_lab

app = FastAPI()

class CheckInRecord(BaseModel):
    stress: Optional[int] = None
    arousal: Optional[int] = None
    valence: Optional[str] = None
    context: Optional[str] = None
    skipped: bool = False
    client_timezone_offset: Optional[int] = None

class ClassAttendanceRecord(BaseModel):
    class_id: str
    status: str
    actual_start: Optional[str] = None
    actual_end: Optional[str] = None
    notes: Optional[str] = None
    client_timezone_offset: Optional[int] = None

class ECGProtocolLogRecord(BaseModel):
    ecg_id: Optional[str] = None
    recording_type: str = "scheduled_baseline"  # scheduled_baseline | diurnal_second | hypothesis_context | symptom_triggered
    posture: str = "Seated quietly"
    recent_exercise: str = "None in past 2 hours"
    caffeine_recent: str = "None in past 4 hours"
    perceived_stress: Optional[int] = None
    symptoms: Optional[str] = None
    notes: Optional[str] = None
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
RENDER_LOCK = threading.Lock()  # build_dashboard writes one shared HTML file; serialise render+read

# Everything shown to you is India Standard Time. Render's server clock is UTC, so never use
# datetime.now()/utcnow() for anything you read; use these helpers instead.
IST = timezone(timedelta(hours=5, minutes=30))


def now_ist_str():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def to_ist_str(iso_utc):
    """'2026-10-18T18:28:00.123Z' -> '2026-10-18 23:58:00' (IST). Returns None if unparseable."""
    try:
        dt = datetime.fromisoformat(str(iso_utc).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None
STATUS = {"last_upload": None, "ok": None, "detail": None, "seconds": None}
INDEXES = {"ready": False, "error": None}


def ensure_indexes_once():
    if INDEXES["ready"]:
        return
    try:
        ingest.ensure_indexes(db)
        db.ecg_recordings.create_index("ecg_id", unique=True)
        db.ecg_recordings.create_index([("start_epoch", -1)])
        db.ecg_recordings.create_index([("date", -1)])
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


def records_to_df(recs):
    """DataFrame of 5-minute bins in strict chronological order. Sorts on the numeric UTC epoch,
    never on the timestamp string (string order breaks as soon as two formats are mixed)."""
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    df["epoch_5m"] = pd.to_numeric(df["epoch_5m"], errors="coerce")
    df = df.dropna(subset=["epoch_5m"])
    df = df.drop_duplicates(subset=["metric", "epoch_5m"], keep="last")
    return df.sort_values(["epoch_5m", "metric"], kind="stable").reset_index(drop=True)


def process_file(path: str, size: int):
    """Runs after the HTTP response is sent. Streams the saved file, never loads it whole."""
    with LOCK:
        t0 = time.time()
        try:
            ensure_indexes_once()
            ing = ingest.Ingestor()
            ingest.ingest_file(path, ing)
            detail = ingest.write(db, ing)
            ecg_detail = cardio_lab.ingest_ecg_from_json_file(path, db)
            detail["ecg_found"] = ecg_detail.get("ecg_found", 0)
            detail["ecg_saved"] = ecg_detail.get("ecg_saved", 0)
            detail["upload_mb"] = round(size / 1024 / 1024, 1)
            STATUS.update(ok=True, detail=detail)
        except Exception as e:
            traceback.print_exc()
            STATUS.update(ok=False, detail=repr(e))
        finally:
            STATUS.update(last_upload=now_ist_str(), seconds=round(time.time() - t0, 1))
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

@app.post("/api/system/sanitize")
def sanitize_database(request: Request):
    """DESTRUCTIVE. POST only; a token must be configured; explicit confirmation; never during an upload."""
    if not WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Set WEBHOOK_TOKEN before using this endpoint.")
    check_token(request)
    if request.query_params.get("confirm") != "DELETE-SENSOR-DATA":
        raise HTTPException(status_code=400, detail="Add confirm=DELETE-SENSOR-DATA to proceed.")

    with LOCK:
        # 1. Drop the automated sensor collections (the corrupted data)
        db.records.drop()
        db.sleep.drop()
        db.sleep_raw.drop()

        # 2. Reset the system so it builds fresh, clean indexes on your next upload
        INDEXES["ready"] = False
        ensure_indexes_once()

    return {"status": "Sanitization complete. Your old ghost data is cleared. Please re-upload your Apple Health export."}

@app.post("/api/checkin")
def save_checkin(record: CheckInRecord, request: Request):
    """
    Persists momentary ecological momentary assessment (EMA) check-ins.
    A skipped check-in explicitly prevents manufacturing a 'zero' stress label.
    """
    check_token(request)
    doc = record.model_dump()
    doc["server_timestamp"] = datetime.utcnow().isoformat() + "Z"   # canonical UTC instant
    doc["server_timestamp_ist"] = to_ist_str(doc["server_timestamp"])  # what you read
    doc["schema_version"] = "1.0"
    
    # Do not treat 'skipped' as a physiological baseline of 0
    if doc["skipped"]:
        doc["stress"] = None
        doc["arousal"] = None
        doc["valence"] = None
        doc["context"] = None
        
    db.psych_checkins.insert_one(doc)
    return {"status": "recorded", "skipped": doc["skipped"]}

@app.post("/api/class-attendance")
def save_class_attendance(record: ClassAttendanceRecord, request: Request):
    check_token(request)

    allowed_statuses = {"attended", "late", "missed", "unsure"}
    if record.status not in allowed_statuses:
        raise HTTPException(status_code=422, detail="Invalid attendance status")

    doc = record.model_dump()
    doc["server_timestamp"] = datetime.utcnow().isoformat() + "Z"   # canonical UTC instant
    doc["server_timestamp_ist"] = to_ist_str(doc["server_timestamp"])  # what you read
    doc["schema_version"] = "1.0"

    db.class_attendance.insert_one(doc)

    return {"status": "recorded", "class_id": doc["class_id"]}


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
        .sort([("epoch_5m", -1), ("metric", 1)])
        .skip(skip)
        .limit(limit)
    )

    # Summarize the time gaps for each metric in the returned page.
    previous = {}
    for row in reversed(rows):
        metric_name = row.get("metric", "Unknown")
        ts = row.get("timestamp", "")
        if metric_name in previous:
            # rows are walked oldest -> newest: this row minus the previous (older) one is positive
            row["gap_minutes"] = round(
                (datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                 - datetime.strptime(previous[metric_name], "%Y-%m-%d %H:%M:%S")
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


@app.get("/stats")
def stats(request: Request):
    check_token(request)
    db_stats = db.command("dbstats")
    return {
        "last_ingest": STATUS,
        "indexes": INDEXES,
        "records": db.records.estimated_document_count(),
        "sleep_rows": db.sleep.estimated_document_count(),
        "sleep_raw_rows": db.sleep_raw.estimated_document_count(),
        "ecg_recordings": db.ecg_recordings.estimated_document_count(),
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
    df_raw = records_to_df(recs)
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

    with RENDER_LOCK:
        analyze_health.build_dashboard(df_raw, df_sleep, log_df, stress_bundle, pred_bundle, sleep_bundle, raw_sleep=df_raw_sleep)
        with open("biometric_science_lab.html", "r", encoding="utf-8") as f:
            html = f.read()

    checkin_widget = r"""
    <section id="bio-checkin" style="
        max-width: 720px; margin: 32px auto; padding: 24px;
        background: #111827; color: #e5e7eb; border: 1px solid #374151;
        border-radius: 16px; font-family: system-ui, sans-serif;
    ">
      <h2 style="margin-top:0">🧠 Bio Lab — Stress Check-in</h2>
      <p style="color:#9ca3af">
        Record how you feel right now. Your watch data will be matched later.
      </p>

      <form id="bio-checkin-form">
        <label for="bio-stress">Stress level (0–4)</label>
        <select id="bio-stress" required style="display:block;width:100%;padding:10px;margin:6px 0 16px">
          <option value="0">0 — None</option>
          <option value="1" selected>1 — Mild</option>
          <option value="2">2 — Moderate</option>
          <option value="3">3 — High</option>
          <option value="4">4 — Extreme</option>
        </select>

        <label for="bio-arousal">Arousal / activation (0–4)</label>
        <select id="bio-arousal" required style="display:block;width:100%;padding:10px;margin:6px 0 16px">
          <option value="0">0 — Very low / calm</option>
          <option value="1">1 — Low</option>
          <option value="2" selected>2 — Moderate</option>
          <option value="3">3 — High / activated</option>
          <option value="4">4 — Very high</option>
        </select>

        <label for="bio-valence">Mood / feeling</label>
        <select id="bio-valence" required style="display:block;width:100%;padding:10px;margin:6px 0 16px">
          <option value="negative">Negative</option>
          <option value="neutral" selected>Neutral</option>
          <option value="positive">Positive</option>
        </select>

        <label for="bio-context">What are you doing?</label>
        <select id="bio-context" required style="display:block;width:100%;padding:10px;margin:6px 0 20px">
          <option value="Resting / everyday activity">Resting / everyday activity</option>
          <option value="Studying">Studying</option>
          <option value="Walking">Walking</option>
          <option value="Exercise">Exercise</option>
          <option value="Social interaction">Social interaction</option>
          <option value="After a meal">After a meal</option>
          <option value="Other">Other</option>
        </select>

        <button type="submit" id="bio-submit" style="
          padding:12px 18px; border:0; border-radius:10px;
          background:#34d399; color:#052e16; font-weight:700; cursor:pointer;
        ">Save check-in</button>

        <button type="button" id="bio-skip" style="
          padding:12px 18px; margin-left:8px; border:1px solid #6b7280;
          border-radius:10px; background:transparent; color:#e5e7eb; cursor:pointer;
        ">Skip</button>

        <p id="bio-message" role="status" aria-live="polite" style="margin-bottom:0"></p>
      </form>
    </section>

    <script>
    (() => {
      const form = document.getElementById("bio-checkin-form");
      if (!form) return;

      const message = document.getElementById("bio-message");
      const submit = document.getElementById("bio-submit");
      const skip = document.getElementById("bio-skip");

      async function saveCheckin(skipped) {
        submit.disabled = true;
        skip.disabled = true;
        message.textContent = "Saving…";

        const payload = {
          stress: Number(document.getElementById("bio-stress").value),
          arousal: Number(document.getElementById("bio-arousal").value),
          valence: document.getElementById("bio-valence").value,
          context: document.getElementById("bio-context").value,
          skipped: skipped,
          client_timezone_offset: new Date().getTimezoneOffset()
        };

        try {
          const response = await fetch("/api/checkin" + window.location.search, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
          });

          const result = await response.json();

          if (!response.ok) {
            throw new Error(
              response.status === 401
                ? "Unauthorized. Reopen the dashboard using its authorized URL."
                : (result.detail || "Request failed (" + response.status + ")")
            );
          }

          message.textContent = skipped
            ? "Skipped check-in recorded."
            : "Check-in saved successfully.";

          if (!skipped) {
            form.reset();
            document.getElementById("bio-stress").value = "1";
            document.getElementById("bio-arousal").value = "2";
            document.getElementById("bio-valence").value = "neutral";
          }
        } catch (error) {
          message.textContent = "Not saved: " + error.message;
        } finally {
          submit.disabled = false;
          skip.disabled = false;
        }
      }

      form.addEventListener("submit", event => {
        event.preventDefault();
        saveCheckin(false);
      });

      skip.addEventListener("click", () => saveCheckin(true));
    })();
    </script>
    """

    if "</body>" in html:
        html = html.replace("</body>", checkin_widget + "\n</body>", 1)
    else:
        raise RuntimeError("Dashboard HTML has no closing body tag.")

    return HTMLResponse(content=html)

@app.get("/api/backtest/readiness")
def backtest_readiness(request: Request):
    check_token(request)

    # Count completed check-ins with valid stress labels.
    checkins = list(
        db.psych_checkins.find(
            {
                "skipped": False,
                "stress": {"$gte": 0, "$lte": 4},
                "server_timestamp": {"$exists": True},
            },
            {
                "_id": 0,
                "stress": 1,
                "server_timestamp": 1,
            },
        ).sort("server_timestamp", 1)
    )

    # Count observations at each stress level.
    stress_label_counts = {
        str(label): sum(
            1 for c in checkins if c.get("stress") == label
        )
        for label in range(5)
    }

    represented_labels = sum(
        1 for count in stress_label_counts.values() if count > 0
    )

    # Inspect the stored heart-rate date range.
    first_hr = db.records.find_one(
        {"metric": "HeartRate", "date": {"$exists": True}},
        {"_id": 0, "date": 1},
        sort=[("date", 1)],
    )

    last_hr = db.records.find_one(
        {"metric": "HeartRate", "date": {"$exists": True}},
        {"_id": 0, "date": 1},
        sort=[("date", -1)],
    )

    # Preliminary screening only, not a model-validation standard.
    if len(checkins) < 10:
        status = "insufficient_data"
    elif represented_labels < 2:
        status = "insufficient_label_diversity"
    else:
        status = "data_available_for_initial_review"

    return {
        "status": status,
        "completed_stress_checkins": len(checkins),
        "minimum_checkins_for_initial_review": 10,
        "stress_label_counts": stress_label_counts,
        "represented_stress_labels": represented_labels,
        "heart_rate_first_date": first_hr.get("date") if first_hr else None,
        "heart_rate_last_date": last_hr.get("date") if last_hr else None,
        "checkins": [
            {
                "timestamp": c.get("server_timestamp"),
                "timestamp_ist": to_ist_str(c.get("server_timestamp")),
                "stress": c.get("stress"),
            }
            for c in checkins
        ],
        "note": (
            "This endpoint checks preliminary data readiness only. "
            "Ten check-ins and two represented labels are screening rules, "
            "not validated minimums for model training. Sensor-window "
            "coverage, label balance, and independent evaluation are "
            "still required before assessing predictive performance."
        ),
    }

@app.get("/api/backtest/dataset")
def backtest_dataset(request: Request):
    check_token(request)

    checkins = list(
        db.psych_checkins.find(
            {
                "skipped": False,
                "stress": {"$gte": 0, "$lte": 4},
                "server_timestamp": {"$exists": True},
            },
            {
                "_id": 0,
                "stress": 1,
                "server_timestamp": 1,
                "client_timezone_offset": 1,
            },
        ).sort("server_timestamp", 1)
    )

    results = []

    for checkin in checkins:
        try:
            timestamp = datetime.fromisoformat(
                checkin["server_timestamp"].replace("Z", "+00:00")
            )
            if timestamp.utcoffset() is None:
                continue

            # db.records timestamps are always IST, so the match window must be IST too
            # (not the phone's offset, which changes with travel/VPN/incognito).
         
            local_end = timestamp.astimezone(IST).replace(tzinfo=None)
            local_start = local_end - timedelta(minutes=60)

            rows = list(
                db.records.find(
                    {
                        "metric": {"$in": ["HeartRate", "HeartRateVariabilitySDNN"]},
                        "date": {
                            "$gte": local_start.strftime("%Y-%m-%d"),
                            "$lte": local_end.strftime("%Y-%m-%d"),
                        },
                        "timestamp": {
                            "$gte": local_start.strftime("%Y-%m-%d %H:%M:%S"),
                            "$lte": local_end.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                    },
                    {"_id": 0, "metric": 1, "timestamp": 1, "value": 1},
                )
            )

            features = {}
            for metric in ["HeartRate", "HeartRateVariabilitySDNN"]:
                values = [
                    float(row["value"])
                    for row in rows
                    if row.get("metric") == metric
                    and row.get("value") is not None
                ]
                features[metric] = {
                    "sample_count": len(values),
                    "mean": round(sum(values) / len(values), 2) if values else None,
                }

            results.append(
                {
                    "checkin_timestamp_utc": checkin["server_timestamp"],
                    "stress_label": int(checkin["stress"]),
                    "window_start_local": local_start.isoformat(sep=" "),
                    "window_end_local": local_end.isoformat(sep=" "),
                    "features": features,
                    "has_heart_rate": features["HeartRate"]["sample_count"] > 0,
                }
            )
        except (ValueError, TypeError, KeyError) as exc:
            results.append(
                {
                    "checkin_timestamp_utc": checkin.get("server_timestamp"),
                    "error": str(exc),
                }
            )

    
    matched = sum(1 for row in results if row.get("has_heart_rate"))

    # Preliminary data-quality checks for each observation.
    MIN_HR_SAMPLES = 6

    for row in results:
        if row.get("error"):
            row["data_quality"] = "error"
            row["usable_for_preliminary_analysis"] = False
            continue

        hr_count = row.get("features", {}).get("HeartRate", {}).get(
            "sample_count", 0
        )
        hrv_count = row.get("features", {}).get(
            "HeartRateVariabilitySDNN", {}
        ).get("sample_count", 0)

        row["data_quality"] = (
            "insufficient_heart_rate"
            if hr_count < MIN_HR_SAMPLES
            else "heart_rate_available"
        )
        row["usable_for_preliminary_analysis"] = hr_count >= MIN_HR_SAMPLES
        row["hrv_available"] = hrv_count > 0

    usable = sum(
        1 for row in results
        if row.get("usable_for_preliminary_analysis") is True
    )
    
    # Count usable observations for each stress label.
    stress_label_summary = {}

    for label in range(5):
        label_rows = [
            row for row in results
            if row.get("stress_label") == label
        ]
        usable_rows = [
            row for row in label_rows
            if row.get("usable_for_preliminary_analysis") is True
        ]

        stress_label_summary[str(label)] = {
            "total_observations": len(label_rows),
            "usable_observations": len(usable_rows),
            "observations_with_hrv": sum(
                1 for row in usable_rows
                if row.get("hrv_available") is True
            ),
        }


    return {
        "status": "dataset_prepared",
        "checkins_processed": len(results),
        "checkins_with_heart_rate": matched,
        "checkins_without_heart_rate": len(results) - matched,
        "checkins_usable_for_preliminary_analysis": usable,
        "stress_label_summary": stress_label_summary,
        "checkins_below_minimum_hr_samples": len(results) - usable,
        "minimum_hr_samples": MIN_HR_SAMPLES,
        "lookback_minutes": 60,
        "future_sensor_data_used": False,
        "results": results,
        "note": (
            "Feature preparation only; no predictive accuracy is calculated. "
            "Six heart-rate samples is a preliminary data-quality rule, "
            "not a clinically validated threshold. HRV availability is "
            "reported separately because HRV samples may be sparse."
        ),
    }


# ==============================================================================
# CARDIOVASCULAR RESEARCH LAB ENDPOINTS (MODULES A - G)
# ==============================================================================

@app.post("/api/ecg/upload")
async def upload_ecg_json(request: Request, background: BackgroundTasks):
    """
    Accepts an Apple Health JSON file, writes it securely to disk, and defers 
    both standard metrics and ECG extraction to the locked background thread.
    """
    check_token(request)
    fd, path = tempfile.mkstemp(suffix=".json")
    size = 0
    with os.fdopen(fd, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            size += len(chunk)

    # Immediately release the event loop and let the locked background thread handle extraction
    background.add_task(process_file, path, size)
    return {
        "status": "accepted",
        "bytes": size,
        "message": "ECG processing safely deferred to background task."
    }


@app.post("/api/ecg/protocol-log")
def save_ecg_protocol_log(record: ECGProtocolLogRecord, request: Request):
    """
    Persists contextual metadata for an ECG recording (scheduled baseline vs symptom-triggered,
    posture, recent exercise, caffeine, perceived stress, and notes).
    """
    check_token(request)
    allowed_types = {
        "scheduled_baseline",
        "diurnal_second",
        "hypothesis_context",
        "symptom_triggered",
    }
    if record.recording_type not in allowed_types:
        raise HTTPException(status_code=422, detail="Invalid recording_type")

    doc = record.model_dump()
    doc["server_timestamp"] = datetime.utcnow().isoformat() + "Z"   # canonical UTC instant
    doc["server_timestamp_ist"] = to_ist_str(doc["server_timestamp"])  # what you read
    doc["schema_version"] = "1.0"
    db.ecg_protocol_logs.insert_one(doc)
    return {"status": "recorded", "recording_type": doc["recording_type"]}


@app.get("/api/ecg/recording/{ecg_id}")
def get_single_ecg_recording(ecg_id: str, request: Request):
    """
    Returns the complete signal-processed ECG waveform (raw & filtered mV vs time),
    Pan-Tompkins beat detections, P-QRS-T median ensemble template, and NN/HRV arrays
    for a single selected recording.
    """
    check_token(request)
    doc = db.ecg_recordings.find_one({"ecg_id": ecg_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="ECG recording not found")
    return clean_nans(doc)


@app.get("/api/ecg/lab-overview")
def get_cardio_lab_overview(request: Request):
    """
    Returns the complete Cardiovascular Research Lab overview:
      - Lightweight summary list of all stored ECG recordings (excluding bulk voltage arrays)
      - Module D: Longitudinal Personal Cardiovascular Baseline & Sensor Availability Audit
      - Module E: Personalized Anomaly Detection (Univariate Robust-Z, Mahalanobis, Isolation Forest)
      - Module F: Cardiovascular Response & Post-Exercise Exponential Recovery Curves
      - Module G: Multimodal Physiological Hypotheses (BH-FDR adjusted) & ECG Sampling Protocol
      - Scientific Methodology, Limitations & Self-Test Verification
    """
    check_token(request)

    # Fetch stored ECG summaries without the heavy `waveform` array so payload stays compact
    ecg_docs = list(
        db.ecg_recordings.find({}, {"_id": 0, "waveform": 0})
        .sort([("start_epoch", -1)])
        .limit(200)
    )

    latest_rec = db.records.find_one({}, {"date": 1, "_id": 0}, sort=[("date", -1)])
    if latest_rec and latest_rec.get("date"):
        cutoff = (datetime.strptime(latest_rec["date"], "%Y-%m-%d") - timedelta(days=DASH_DAYS)).strftime("%Y-%m-%d")
        recs = list(db.records.find({"date": {"$gte": cutoff}}, {"_id": 0}))
        df_raw = records_to_df(recs)
        sleep_rows = list(db.sleep.find({"date": {"$gte": cutoff}}, {"_id": 0}))
        df_sleep = pd.DataFrame(sleep_rows).drop_duplicates() if sleep_rows else pd.DataFrame()
    else:
        df_raw = pd.DataFrame()
        df_sleep = pd.DataFrame()

    baseline_bundle = cardio_lab.compute_longitudinal_baseline(df_raw, df_sleep, ecg_docs)
    anomaly_bundle = cardio_lab.detect_cardio_anomalies(df_raw, df_sleep, ecg_docs)
    recovery_bundle = cardio_lab.analyze_exercise_recovery(df_raw)
    multimodal_bundle = cardio_lab.run_multimodal_research_and_protocol(df_raw, df_sleep, ecg_docs)
    protocol_logs = list(db.ecg_protocol_logs.find({}, {"_id": 0}).sort([("server_timestamp", -1)]).limit(30))
    self_tests = cardio_self_tests_cached()

    return clean_nans({
        "ecg_count": len(ecg_docs),
        "ecg_recordings": ecg_docs,
        "baseline": baseline_bundle,
        "anomalies": anomaly_bundle,
        "recovery": recovery_bundle,
        "multimodal": multimodal_bundle,
        "protocol_logs": protocol_logs,
        "methodology": cardio_lab.METHODOLOGY_REGISTRY,
        "self_tests": self_tests,
    })

_SELF_TEST_CACHE = {}


def cardio_self_tests_cached():
    """The DSP self-test checks code, not your data: run it once per process and never let it 500 the endpoint."""
    if "r" not in _SELF_TEST_CACHE:
        try:
            _SELF_TEST_CACHE["r"] = cardio_lab.run_cardio_lab_self_tests()
        except Exception as e:
            _SELF_TEST_CACHE["r"] = {"all_passed": False, "error": repr(e)}
    return _SELF_TEST_CACHE["r"]


@app.get("/api/backtest/evaluate")
def backtest_evaluate(request: Request):
    """Audit whether a real, frozen predictor can be evaluated against check-ins.

    This endpoint intentionally does NOT manufacture predictions. The previous
    version mapped ordinal self-reports to arbitrary percentages and generated a
    heart-rate-only score with hand-picked constants, then reported MAE/RMSE as
    though it were a trained model. That is not a valid model evaluation.
    """
    check_token(request)

    # Dataset preparation is useful; it is not itself a prediction model.
    dataset_response = backtest_dataset(request)
    results = dataset_response.get("results", [])
    usable_data = [
        row for row in results
        if row.get("usable_for_preliminary_analysis") is True
        and row.get("stress_label") in (0, 1, 2, 3, 4)
    ]

    return {
        "status": "not_evaluable_model_not_connected",
        "dataset_status": dataset_response.get("status", "unknown"),
        "checkins_processed": len(results),
        "usable_labeled_observations": len(usable_data),
        "metrics": None,
        "reason": (
            "No trained, frozen stress predictor is connected to this endpoint. "
            "The previous hand-written HR-to-score formula and arbitrary mapping "
            "from the 0-4 self-report scale to percentages have been removed; "
            "reporting MAE/RMSE from those values would be misleading."
        ),
        "data_notes": [
            "Stress check-ins are ordinal self-reports (0-4), not validated percentages.",
            "Heart-rate sample-count eligibility is a preliminary data-quality rule, not a clinical threshold.",
            "A valid evaluation needs a defined target, a predictor trained only on earlier observations, and chronological held-out predictions.",
            "Any model using HRV must explicitly handle missing HRV rather than silently treating it as zero.",
        ],
        "future_sensor_data_used": dataset_response.get("future_sensor_data_used"),
    }
