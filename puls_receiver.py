"""Puls Sync Protocol v1 receiver for the Apple Watch Lab MongoDB backend.

The receiver validates a complete bounded NDJSON batch before touching MongoDB.
Writes are idempotent so PulsHealth can safely retry after a timeout or restart.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import uuid as uuidlib
from datetime import datetime, timedelta, timezone

from pymongo import UpdateOne
from pymongo.errors import BulkWriteError, DuplicateKeyError

PROTOCOL = 1
DEFAULT_USER_ID = "5ea4d000-0000-4000-8000-000000000001"
MAX_COMPRESSED_BYTES = 250 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_LINE_BYTES = 4 * 1024 * 1024
MAX_COUNT = 100_000
MAX_TOTAL_LINES = 200_000
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
TIMESTAMP_MIN = -62135596800000
TIMESTAMP_MAX = 253402300799999
EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)

COUNTS = ("sampleCount", "deletionCount", "routeCount", "seriesCount", "aggregateCount", "activitySummaryCount", "profileCount")
WRAPPERS = ("deleted", "route", "series", "aggregate", "activitySummary", "profile")
KINDS = {"quantity", "category", "workout", "heartbeatSeries", "ecg", "stateOfMind", "medicationDose"}


class ProtocolError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _fail(message):
    raise ProtocolError(message)


def _is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _uuid(value, label):
    if not isinstance(value, str) or not UUID_RE.fullmatch(value):
        _fail(f"{label} must be a UUID")
    return value.lower()


def _time(value, label, required=False):
    if value is None and not required:
        return None
    if not _is_finite_number(value):
        _fail(f"{label} must be an epoch-millisecond number")
    if value < TIMESTAMP_MIN or value > TIMESTAMP_MAX:
        _fail(f"{label} is outside the supported timestamp range")
    return float(value)


def _epoch_datetime(value):
    """Convert protocol epoch milliseconds without platform-specific fromtimestamp limits."""
    return EPOCH_UTC + timedelta(milliseconds=float(value))


def _nonempty(value, label):
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _temporal_context(value, label):
    if value is None:
        return
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    for key in ("timeZoneID", "utcOffsetSeconds", "source", "confidence"):
        if key not in value:
            _fail(f"{label}.{key} is required")


def validate_header(header, protocol_header=None):
    if not isinstance(header, dict):
        _fail("line 1 must be a header object")
    schema = header.get("schemaVersion", 1)
    if schema is None:
        schema = 1
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != PROTOCOL:
        raise ProtocolError('{"error":"unsupported protocol version","supportedVersions":[1]}', 400)
    if protocol_header is not None and protocol_header != "1":
        raise ProtocolError('{"error":"unsupported protocol version","supportedVersions":[1]}', 400)
    _uuid(header.get("batchID"), "batchID")
    _nonempty(header.get("deviceID"), "deviceID")
    _nonempty(header.get("type"), "type")
    reason = header.get("reason")
    if not isinstance(reason, str) or reason not in {"backfill", "incremental", "manual", "reconciliation"}:
        _fail("header.reason is invalid")
    _time(header.get("exportedAt"), "header.exportedAt", required=True)
    counts = {}
    for key in COUNTS:
        if key in ("sampleCount", "deletionCount") and key not in header:
            _fail(f"header.{key} is required")
        value = header.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MAX_COUNT:
            _fail(f"header.{key} must be an integer from 0 to {MAX_COUNT}")
        counts[key] = value
    if sum(counts.values()) > MAX_TOTAL_LINES:
        raise ProtocolError("batch declares too many lines", 413)
    if counts["profileCount"] not in (0, 1):
        _fail("header.profileCount must be 0 or 1")
    if header.get("timeZoneID") not in (None, "") and (not isinstance(header["timeZoneID"], str) or len(header["timeZoneID"]) > 100):
        _fail("header.timeZoneID is invalid")
    return counts


def _validate_sample(obj, index):
    where = f"sample {index}"
    _uuid(obj.get("uuid"), f"{where}.uuid")
    _nonempty(obj.get("type"), f"{where}.type")
    kind = obj.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        _fail(f"{where}: invalid kind {kind!r}")
    start = _time(obj.get("start"), f"{where}.start", required=True)
    end = _time(obj.get("end"), f"{where}.end")
    if end is not None and end < start:
        _fail(f"{where}.end precedes start")
    for key in ("startContext", "endContext"):
        _temporal_context(obj.get(key), f"{where}.{key}")
    for key in ("sourceName", "sourceBundleID", "sourceVersion", "device"):
        if obj.get(key) is not None and not isinstance(obj[key], str):
            _fail(f"{where}.{key} must be a string")
    if obj.get("metadata") is not None and not isinstance(obj["metadata"], dict):
        _fail(f"{where}.metadata must be an object")
    for key in ("category", "value", "unit"):
        if key in obj and obj[key] is not None and key in ("category", "value"):
            if not _is_finite_number(obj[key]):
                _fail(f"{where}.{key} must be numeric")
            if key == "category" and not isinstance(obj[key], int):
                _fail(f"{where}.category must be an integer")
        if key == "unit" and obj.get(key) is not None and not isinstance(obj[key], str):
            _fail(f"{where}.unit must be a string")
    detail = {"quantity": ("value", "unit"), "category": ("category",), "workout": ("workout",),
              "heartbeatSeries": ("heartbeats",), "ecg": ("ecg",), "stateOfMind": ("stateOfMind",),
              "medicationDose": ("medicationDose",)}[kind]
    for key in detail:
        if key not in obj or obj[key] is None:
            _fail(f"{where}.{key} is required for {kind}")
    if kind == "ecg":
        ecg = obj["ecg"]
        if not isinstance(ecg, dict) or not isinstance(ecg.get("voltagesUV"), list):
            _fail(f"{where}.ecg.voltagesUV must be an array")
        if any(not _is_finite_number(v) for v in ecg["voltagesUV"]):
            _fail(f"{where}.ecg.voltagesUV contains a non-numeric value")
        for key in ("averageHeartRateBpm", "samplingFrequencyHz"):
            if key in ecg and ecg[key] is not None and not _is_finite_number(ecg[key]):
                _fail(f"{where}.ecg.{key} must be numeric")
        if ecg.get("samplingFrequencyHz") is not None and not 1 <= float(ecg["samplingFrequencyHz"]) <= 10000:
            _fail(f"{where}.ecg.samplingFrequencyHz is outside the supported range")
        if ecg.get("symptomsStatus") not in (None, "notSet", "none", "present"):
            _fail(f"{where}.ecg.symptomsStatus is invalid")
    if kind == "workout":
        workout = obj["workout"]
        if (not isinstance(workout, dict) or not isinstance(workout.get("activityType"), str)
                or not _is_finite_number(workout.get("duration"))):
            _fail(f"{where}.workout is incomplete")
        if workout["duration"] < 0:
            _fail(f"{where}.workout.duration cannot be negative")
    if kind == "heartbeatSeries":
        if not isinstance(obj["heartbeats"], list) or any(
            not isinstance(pair, list) or len(pair) != 2
            or not _is_finite_number(pair[0])
            or not isinstance(pair[1], bool)
            for pair in obj["heartbeats"]
        ):
            _fail(f"{where}.heartbeats contains an invalid point")
    if kind == "stateOfMind":
        mood = obj["stateOfMind"]
        if not isinstance(mood, dict) or mood.get("kind") not in ("momentaryEmotion", "dailyMood"):
            _fail(f"{where}.stateOfMind is invalid")
        if not _is_finite_number(mood.get("valence")):
            _fail(f"{where}.stateOfMind.valence is invalid")
        for key in ("labels", "associations"):
            if key in mood and (not isinstance(mood[key], list) or any(not isinstance(v, str) for v in mood[key])):
                _fail(f"{where}.stateOfMind.{key} is invalid")
    if kind == "medicationDose" and not isinstance(obj["medicationDose"], dict):
        _fail(f"{where}.medicationDose is invalid")


def _validate_line(line, slot, index):
    if not isinstance(line, dict):
        _fail(f"line {index} must be an object")
    if slot == "sampleCount":
        if any(k in line for k in WRAPPERS):
            _fail(f"line {index} has an unexpected wrapper")
        _validate_sample(line, index)
        return
    wrapper = {"deletionCount": "deleted", "routeCount": "route", "seriesCount": "series",
               "aggregateCount": "aggregate", "activitySummaryCount": "activitySummary", "profileCount": "profile"}[slot]
    if set(line.keys()) != {wrapper} or not isinstance(line[wrapper], dict):
        _fail(f"line {index} must contain only the {wrapper!r} wrapper")
    p = line[wrapper]
    if wrapper == "deleted":
        _uuid(p.get("uuid"), f"line {index}.deleted.uuid")
        _nonempty(p.get("type"), f"line {index}.deleted.type")
    elif wrapper == "route":
        _uuid(p.get("workoutUUID"), f"line {index}.route.workoutUUID")
        if not isinstance(p.get("points"), list) or len(p["points"]) > 4000:
            _fail(f"line {index}.route.points is invalid")
        for point in p["points"]:
            if (not isinstance(point, dict) or _time(point.get("t"), "route point t", True) is None
                    or not _is_finite_number(point.get("lat")) or not _is_finite_number(point.get("lon"))
                    or not -90 <= point["lat"] <= 90 or not -180 <= point["lon"] <= 180):
                _fail(f"line {index} contains an invalid route point")
    elif wrapper == "series":
        _uuid(p.get("workoutUUID"), f"line {index}.series.workoutUUID")
        _nonempty(p.get("type"), f"line {index}.series.type")
        _nonempty(p.get("unit"), f"line {index}.series.unit")
        if not isinstance(p.get("points"), list) or len(p["points"]) > 4000:
            _fail(f"line {index}.series.points is invalid")
        for point in p["points"]:
            if not isinstance(point, dict) or _time(point.get("t"), "series point t", True) is None or not _is_finite_number(point.get("value")):
                _fail(f"line {index} contains an invalid series point")
    elif wrapper == "aggregate":
        for key in ("type", "func", "intervalUnit", "deviceFilter"):
            _nonempty(p.get(key), f"line {index}.aggregate.{key}")
        if p["func"] not in {"sum", "average", "min", "max", "mostRecent", "duration"} or p["intervalUnit"] not in {"minute", "hour", "day", "week", "month"} or p["deviceFilter"] not in {"all", "watch", "iphone"}:
            _fail(f"line {index}.aggregate contains an invalid enum")
        if isinstance(p.get("intervalValue"), bool) or not isinstance(p.get("intervalValue"), int) or p["intervalValue"] < 1:
            _fail(f"line {index}.aggregate.intervalValue must be a positive integer")
        _time(p.get("bucketStart"), f"line {index}.aggregate.bucketStart", True)
        _time(p.get("bucketEnd"), f"line {index}.aggregate.bucketEnd", True)
        if p["bucketEnd"] <= p["bucketStart"]:
            _fail(f"line {index}.aggregate.bucketEnd must follow bucketStart")
        if "value" not in p or p["value"] is not None and not _is_finite_number(p["value"]):
            _fail(f"line {index}.aggregate.value is invalid")
    elif wrapper == "activitySummary":
        _time(p.get("date"), f"line {index}.activitySummary.date", True)
        if p.get("localDate") is not None and (not isinstance(p["localDate"], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", p["localDate"])):
            _fail(f"line {index}.activitySummary.localDate is invalid")
        _temporal_context(p.get("temporalContext"), f"line {index}.activitySummary.temporalContext")
        if p.get("moveMode") not in (None, 0, 1):
            _fail(f"line {index}.activitySummary.moveMode is invalid")
    elif wrapper == "profile":
        for key in ("name", "email"):
            if p.get(key) is not None and not isinstance(p[key], str):
                _fail(f"line {index}.profile.{key} is invalid")
        _time(p.get("dateOfBirth"), f"line {index}.profile.dateOfBirth")
        if p.get("biologicalSex") not in (None, "female", "male", "other"):
            _fail(f"line {index}.profile.biologicalSex is invalid")


def _iter_lines(path):
    def reject_constant(value):
        raise ValueError(f"non-standard JSON number {value}")

    def check_finite_tree(value, number):
        if isinstance(value, float) and not math.isfinite(value):
            _fail(f"line {number} contains a non-finite number")
        if isinstance(value, dict):
            for child in value.values():
                check_finite_tree(child, number)
        elif isinstance(value, list):
            for child in value:
                check_finite_tree(child, number)

    with open(path, "rb") as f:
        number = 0
        for raw in f:
            if not raw.strip():
                continue
            number += 1
            if len(raw) > MAX_LINE_BYTES:
                raise ProtocolError(f"line {number} exceeds the {MAX_LINE_BYTES}-byte limit", 413)
            try:
                value = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
                _fail(f"line {number} is not valid UTF-8 JSON: {e}")
            check_finite_tree(value, number)
            yield number, value


def _validate_file(path, protocol_header):
    it = iter(_iter_lines(path))
    try:
        _, header = next(it)
    except StopIteration:
        _fail("batch is empty")
    counts = validate_header(header, protocol_header)
    order = ("sampleCount", "deletionCount", "routeCount", "seriesCount", "aggregateCount", "activitySummaryCount", "profileCount")
    index = 1
    for slot in order:
        for _ in range(counts[slot]):
            try:
                _, line = next(it)
            except StopIteration:
                _fail(f"batch ended before all {slot} lines arrived")
            index += 1
            _validate_line(line, slot, index)
    try:
        next(it)
    except StopIteration:
        return header, counts
    _fail("batch contains trailing lines after its declared counts")


def decode_and_validate(upload_path, content_encoding, protocol_header):
    """Decompress to a temporary NDJSON file, enforce limits, and validate framing before writes."""
    encoding = (content_encoding or "identity").strip().lower()
    if encoding not in ("gzip", "identity", ""):
        _fail("unsupported Content-Encoding")
    plain_fd, plain_path = tempfile.mkstemp(prefix="puls-", suffix=".ndjson")
    total = 0
    try:
        opener = gzip.open if encoding == "gzip" else open
        with opener(upload_path, "rb") as source, os.fdopen(plain_fd, "wb") as target:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_DECOMPRESSED_BYTES:
                    raise ProtocolError("decompressed batch exceeds the size limit", 413)
                target.write(block)
        header, counts = _validate_file(plain_path, protocol_header)
        return plain_path, header, counts
    except (OSError, EOFError) as e:
        try:
            os.close(plain_fd)
        except OSError:
            pass
        try:
            os.remove(plain_path)
        except OSError:
            pass
        raise ProtocolError(f"invalid gzip or unreadable batch: {e}") from e
    except Exception:
        try:
            os.close(plain_fd)
        except OSError:
            pass
        try:
            os.remove(plain_path)
        except OSError:
            pass
        raise


def ensure_indexes(db):
    db.puls_samples.create_index([("user_id", 1), ("uuid", 1)], unique=True)
    db.puls_samples.create_index([("user_id", 1), ("metric", 1), ("epoch_5m", 1)])
    db.puls_deletions.create_index([("user_id", 1), ("uuid", 1)], unique=True)
    db.puls_routes.create_index([("user_id", 1), ("workoutUUID", 1), ("t", 1)], unique=True)
    db.puls_series.create_index([("user_id", 1), ("workoutUUID", 1), ("type", 1), ("t", 1)], unique=True)
    db.puls_workout_sources.create_index([("user_id", 1), ("workoutUUID", 1)], unique=True)
    db.puls_pending_routes.create_index([("user_id", 1), ("workoutUUID", 1), ("t", 1)], unique=True)
    db.puls_pending_series.create_index([("user_id", 1), ("workoutUUID", 1), ("type", 1), ("t", 1)], unique=True)
    db.puls_aggregates.create_index([("user_id", 1), ("type", 1), ("func", 1), ("intervalValue", 1), ("intervalUnit", 1), ("deviceFilter", 1), ("bucketStart", 1)], unique=True)
    db.puls_activity_summaries.create_index([("user_id", 1), ("localDate", 1)], unique=True)
    db.puls_profiles.create_index("user_id", unique=True)
    db.puls_batches.create_index([("user_id", 1), ("batchID", 1)], unique=True)


def _watch_sample(sample):
    # ECG is an Apple Watch-only HealthKit type. For other samples require explicit device/source evidence.
    if sample.get("kind") == "ecg" or sample.get("type") == "HKDataTypeIdentifierElectrocardiogram":
        return True
    device = (sample.get("device") or "").casefold()
    source = (sample.get("sourceName") or "").casefold()
    if "iphone" in device or "iphone" in source:
        return False
    return "apple watch" in device or "apple watch" in source


def _normalise_quantity(sample):
    """Return the existing dashboard metric/value/unit tuple for supported HealthKit quantities."""
    import ingest
    name = ingest.XML_METRIC_MAP.get(sample.get("type"))
    if not name:
        return None
    metric = ingest.METRIC_MAP.get(name, name)
    try:
        value = float(sample["value"])
    except (KeyError, TypeError, ValueError):
        return None
    unit = str(sample.get("unit") or "")
    ul = unit.casefold()
    if metric == "WalkingSpeed":
        if "m/s" in ul and "km" not in ul:
            value, unit = value * 3.6, "km/hr"
        elif ul in ("mi/hr", "mph", "mi/h"):
            value, unit = value * 1.609344, "km/hr"
    elif metric == "WalkingStepLength" and ul == "m":
        value, unit = value * 100, "cm"
    elif metric == "DistanceWalkingRunning":
        if ul == "m":
            value, unit = value / 1000, "km"
        elif ul in ("mi", "mile", "miles"):
            value, unit = value * 1.609344, "km"
    elif metric in ingest.ENERGY_METRICS and ul in ("kj", "kilojoule", "kilojoules"):
        value, unit = value / 4.184, "kcal"
    elif metric in ingest.PCT_METRICS and "%" not in unit and value <= 1:
        value, unit = value * 100, "%"
    if not math.isfinite(value):
        return None
    return metric, value, unit


def _rebuild_metric_bin(db, user_id, metric, epoch_5m):
    """Recompute a dashboard bin from UUID-deduplicated raw HealthKit samples."""
    import ingest
    query = {"user_id": user_id, "source_policy": "apple_watch_only", "metric": metric, "epoch_5m": epoch_5m}
    rows = list(db.puls_samples.find(query, {"_id": 0, "metric_value": 1, "metric_unit": 1, "sourceName": 1, "device": 1}))
    if not rows:
        db.records.delete_one({"metric": metric, "epoch_5m": epoch_5m, "source_policy": "apple_watch_only", "ingest_method": "pulshealth"})
        return
    values = [float(row["metric_value"]) for row in rows if _is_finite_number(row.get("metric_value"))]
    if not values:
        return
    total = sum(values)
    value = total if metric in ingest.SUM_METRICS else total / len(values)
    start_utc = EPOCH_UTC + timedelta(seconds=epoch_5m * 300)
    start_ist = start_utc.astimezone(timezone(timedelta(hours=5, minutes=30)))
    sources = sorted({str(row.get("sourceName") or row.get("device") or "Apple Watch") for row in rows})
    doc = {
        "metric": metric,
        "timestamp": start_ist.strftime("%Y-%m-%d %H:%M:%S"),
        "date": start_ist.strftime("%Y-%m-%d"),
        "hour": round(start_ist.hour + start_ist.minute / 60.0, 2),
        "epoch_5m": epoch_5m,
        "value": float(value),
        "unit": next((str(row.get("metric_unit")) for row in rows if row.get("metric_unit")), ""),
        "sources": sources,
        "source_policy": "apple_watch_only",
        "ingest_method": "pulshealth",
    }
    db.records.update_one({"metric": metric, "epoch_5m": epoch_5m}, {"$set": doc}, upsert=True)


def _store_sleep_sample(db, sample, user_id):
    """Mirror raw Watch sleep stages into the dashboard's existing sleep collections."""
    import ingest
    if sample.get("type") != "HKCategoryTypeIdentifierSleepAnalysis":
        return
    stages = {0: "inBed", 1: "asleep", 2: "awake", 3: "core", 4: "deep", 5: "rem"}
    stage = stages.get(sample.get("category"))
    if stage is None:
        return
    start_ms = float(sample["start"])
    end_ms = float(sample.get("end") or start_ms)
    start_iso = _epoch_datetime(start_ms).isoformat()
    end_iso = _epoch_datetime(end_ms).isoformat()
    start_local, end_local = ingest.ist_str(start_iso), ingest.ist_str(end_iso)
    start_dt, end_dt = ingest.to_ist(ingest.parse_ts(start_iso)), ingest.to_ist(ingest.parse_ts(end_iso))
    d = end_local[:10]
    raw = {"k": sample["uuid"].lower(), "d": d, "s19": start_local, "e19": end_local,
           "value": stage, "stage": stage, "source": sample.get("sourceName") or sample.get("device"),
           "sourceName": sample.get("sourceName"), "device": sample.get("device"),
           "source_policy": "apple_watch_only", "user_id": user_id}
    db.sleep_raw.update_one({"k": raw["k"]}, {"$setOnInsert": raw}, upsert=True)
    labels = {"deep": "Deep Sleep (Body Repair)", "rem": "REM Sleep (Brain & Memory)",
              "core": "Core Sleep (Light)", "asleep": "Core Sleep (Light)", "awake": "Awake in Bed"}
    if stage not in labels or end_ms <= start_ms:
        return
    start_epoch = int(start_ms / 1000 // 300)
    end_epoch = int(end_ms / 1000 // 300)
    duration = (end_ms - start_ms) / 3_600_000
    onset = start_dt.hour + start_dt.minute / 60
    if onset < 14:
        onset += 24
    doc = {"date": d, "stage": labels[stage], "hours": duration, "onset_hr": onset,
           "source": raw["source"], "source_policy": "apple_watch_only", "wake_hr": end_dt.hour + end_dt.minute / 60,
           "start_epoch": start_epoch, "end_epoch": end_epoch, "healthkit_uuid": sample["uuid"].lower(), "ingest_method": "pulshealth"}
    db.sleep.update_one({"stage": doc["stage"], "start_epoch": start_epoch, "end_epoch": end_epoch}, {"$setOnInsert": doc}, upsert=True)


def _bulk_insert_only(collection, operations):
    if not operations:
        return 0
    try:
        result = collection.bulk_write(operations, ordered=False)
        return result.upserted_count
    except BulkWriteError as exc:
        errors = (exc.details or {}).get("writeErrors", [])
        if errors and all(item.get("code") == 11000 for item in errors):
            return (exc.details or {}).get("nUpserted", 0)
        raise


def _save_workout_points(db, user_id, workout_uuid, routes=None, series=None):
    """Keep route/stream points only when the associated workout is explicitly Watch-sourced."""
    decision = db.puls_workout_sources.find_one({"user_id": user_id, "workoutUUID": workout_uuid}, {"_id": 0, "watch": 1})
    known_watch = decision.get("watch") if decision else None
    route_ops, series_ops = [], []
    for point in routes or []:
        doc = {**point, "user_id": user_id, "workoutUUID": workout_uuid}
        if known_watch is True:
            route_ops.append(UpdateOne({"user_id": user_id, "workoutUUID": workout_uuid, "t": doc["t"]}, {"$setOnInsert": {**doc, "source_policy": "apple_watch_only"}}, upsert=True))
        elif known_watch is None:
            db.puls_pending_routes.update_one({"user_id": user_id, "workoutUUID": workout_uuid, "t": doc["t"]}, {"$setOnInsert": doc}, upsert=True)
    for point in series or []:
        doc = {**point, "user_id": user_id, "workoutUUID": workout_uuid}
        if known_watch is True:
            series_ops.append(UpdateOne({"user_id": user_id, "workoutUUID": workout_uuid, "type": doc["type"], "t": doc["t"]}, {"$setOnInsert": {**doc, "source_policy": "apple_watch_only"}}, upsert=True))
        elif known_watch is None:
            db.puls_pending_series.update_one({"user_id": user_id, "workoutUUID": workout_uuid, "type": doc["type"], "t": doc["t"]}, {"$setOnInsert": doc}, upsert=True)
    added_routes = _bulk_insert_only(db.puls_routes, route_ops)
    added_series = _bulk_insert_only(db.puls_series, series_ops)
    if known_watch is False:
        db.puls_pending_routes.delete_many({"user_id": user_id, "workoutUUID": workout_uuid})
        db.puls_pending_series.delete_many({"user_id": user_id, "workoutUUID": workout_uuid})
    return added_routes, added_series


def _resolve_pending_workout(db, user_id, workout_uuid, is_watch):
    route_points = list(db.puls_pending_routes.find({"user_id": user_id, "workoutUUID": workout_uuid}, {"_id": 0}))
    series_points = list(db.puls_pending_series.find({"user_id": user_id, "workoutUUID": workout_uuid}, {"_id": 0}))
    added_routes = added_series = 0
    if is_watch:
        route_ops = [UpdateOne({"user_id": user_id, "workoutUUID": workout_uuid, "t": p["t"]}, {"$setOnInsert": {**p, "source_policy": "apple_watch_only"}}, upsert=True) for p in route_points]
        series_ops = [UpdateOne({"user_id": user_id, "workoutUUID": workout_uuid, "type": p["type"], "t": p["t"]}, {"$setOnInsert": {**p, "source_policy": "apple_watch_only"}}, upsert=True) for p in series_points]
        added_routes = _bulk_insert_only(db.puls_routes, route_ops)
        added_series = _bulk_insert_only(db.puls_series, series_ops)
    db.puls_pending_routes.delete_many({"user_id": user_id, "workoutUUID": workout_uuid})
    db.puls_pending_series.delete_many({"user_id": user_id, "workoutUUID": workout_uuid})
    return added_routes, added_series


def _apply(db, path, header, counts, user_id):
    ensure_indexes(db)
    totals = {"accepted": 0, "deleted": 0, "duplicates": 0, "excludedSources": 0,
              "routePoints": 0, "seriesPoints": 0, "aggregateSamples": 0, "activitySummaries": 0}
    header_line = True
    sample_index = 0
    aggregate_keys = set()
    activity_keys = set()
    for _, line in _iter_lines(path):
        if header_line:
            header_line = False
            continue
        if any(k in line for k in WRAPPERS):
            wrapper = next(k for k in WRAPPERS if k in line)
            p = line[wrapper]
            if wrapper == "deleted":
                uid = p["uuid"].lower()
                old = db.puls_samples.find_one({"user_id": user_id, "uuid": uid}, {"_id": 0})
                result = db.puls_samples.delete_one({"user_id": user_id, "uuid": uid})
                if result.deleted_count:
                    totals["deleted"] += 1
                db.puls_deletions.update_one({"user_id": user_id, "uuid": uid}, {"$set": {"user_id": user_id, "uuid": uid, "type": p["type"], "deletedAt": datetime.now(timezone.utc)}}, upsert=True)
                if old and old.get("kind") == "ecg":
                    db.ecg_recordings.delete_one({"ecg_id": uid})
                if old and old.get("metric") and old.get("epoch_5m") is not None:
                    _rebuild_metric_bin(db, user_id, old["metric"], old["epoch_5m"])
                if old and old.get("type") == "HKCategoryTypeIdentifierSleepAnalysis":
                    db.sleep_raw.delete_one({"k": uid})
                    db.sleep.delete_many({"healthkit_uuid": uid})
                if old and old.get("kind") == "workout":
                    db.puls_routes.delete_many({"user_id": user_id, "workoutUUID": uid})
                    db.puls_series.delete_many({"user_id": user_id, "workoutUUID": uid})
                    db.puls_pending_routes.delete_many({"user_id": user_id, "workoutUUID": uid})
                    db.puls_pending_series.delete_many({"user_id": user_id, "workoutUUID": uid})
                    db.puls_workout_sources.delete_one({"user_id": user_id, "workoutUUID": uid})
                continue
            if wrapper == "route":
                added_routes, _ = _save_workout_points(db, user_id, p["workoutUUID"].lower(), routes=p["points"])
                totals["routePoints"] += added_routes
                continue
            if wrapper == "series":
                points = [{**point, "type": p["type"], "unit": p.get("unit")} for point in p["points"]]
                _, added_series = _save_workout_points(db, user_id, p["workoutUUID"].lower(), series=points)
                totals["seriesPoints"] += added_series
                continue
            if wrapper == "aggregate":
                if p.get("deviceFilter") != "watch":
                    totals["excludedSources"] += 1
                    continue
                doc = dict(p, user_id=user_id, source_policy="apple_watch_only")
                key = {k: doc[k] for k in ("user_id", "type", "func", "intervalValue", "intervalUnit", "deviceFilter", "bucketStart")}
                identity = tuple(key[k] for k in ("user_id", "type", "func", "intervalValue", "intervalUnit", "deviceFilter", "bucketStart"))
                aggregate_keys.add(identity)
                db.puls_aggregates.update_one(key, {"$set": doc}, upsert=True)
                continue
            if wrapper == "activitySummary":
                local_date = p.get("localDate") or _epoch_datetime(p["date"]).date().isoformat()
                doc = {"user_id": user_id, **p, "localDate": local_date, "source_policy": "apple_watch_only"}
                db.puls_activity_summaries.replace_one({"user_id": user_id, "localDate": local_date}, doc, upsert=True)
                activity_keys.add(local_date)
                continue
            if wrapper == "profile":
                db.puls_profiles.replace_one({"user_id": user_id}, {"user_id": user_id, **p}, upsert=True)
                continue
        sample_index += 1
        is_watch = _watch_sample(line)
        if line.get("kind") == "workout":
            db.puls_workout_sources.update_one(
                {"user_id": user_id, "workoutUUID": line["uuid"].lower()},
                {"$set": {"user_id": user_id, "workoutUUID": line["uuid"].lower(), "watch": is_watch}},
                upsert=True,
            )
            added_routes, added_series = _resolve_pending_workout(db, user_id, line["uuid"].lower(), is_watch)
            totals["routePoints"] += added_routes
            totals["seriesPoints"] += added_series
        if not is_watch:
            totals["excludedSources"] += 1
            continue
        uid = line["uuid"].lower()
        known_sample_fields = {
            "uuid", "type", "kind", "start", "end", "startContext", "endContext", "value", "unit",
            "category", "sourceName", "sourceBundleID", "sourceVersion", "device", "metadata", "workout",
            "heartbeats", "ecg", "stateOfMind", "medicationDose",
        }
        doc = {key: value for key, value in line.items() if key in known_sample_fields}
        doc["uuid"] = uid
        doc["user_id"] = user_id
        doc["start"] = float(line["start"])
        if line.get("end") is not None:
            doc["end"] = float(line["end"])
        doc["source_policy"] = "apple_watch_only"
        metric_info = _normalise_quantity(line) if line.get("kind") == "quantity" else None
        if metric_info:
            metric, metric_value, metric_unit = metric_info
            doc.update(metric=metric, metric_value=metric_value, metric_unit=metric_unit,
                       epoch_5m=int((float(line["start"]) / 1000) // 300))
        try:
            result = db.puls_samples.update_one({"user_id": user_id, "uuid": uid}, {"$setOnInsert": doc}, upsert=True)
            is_new = result.upserted_id is not None
            totals["accepted" if is_new else "duplicates"] += 1
        except DuplicateKeyError:
            is_new = False
            totals["duplicates"] += 1
        if metric_info:
            _rebuild_metric_bin(db, user_id, doc["metric"], doc["epoch_5m"])
        if line.get("kind") == "category":
            _store_sleep_sample(db, line, user_id)
        if line.get("kind") == "ecg":
            # The HealthKit UUID is the stable identity; retries replace the same derived ECG row.
            if not db.ecg_recordings.find_one({"ecg_id": uid}, {"_id": 1}):
                from cardio_lab import process_raw_ecg_recording
                start = float(line["start"])
                freq = float((line.get("ecg") or {}).get("samplingFrequencyHz") or 512.0)
                volts = (line.get("ecg") or {}).get("voltagesUV") or []
                raw = {
                    "start": _epoch_datetime(start).isoformat(),
                    "end": _epoch_datetime(float(line.get("end") or start)).isoformat(),
                    "classification": (line.get("ecg") or {}).get("classification", "Unclassified"),
                    "symptoms": (line.get("ecg") or {}).get("symptomsStatus"),
                    "voltageMeasurements": [
                        {"voltage": v, "date": _epoch_datetime(start + i * 1000.0 / freq).isoformat(), "units": "mcV"}
                        for i, v in enumerate(volts)
                    ],
                }
                processed = process_raw_ecg_recording(raw)
                processed.update(ecg_id=uid, healthkit_uuid=uid, sourceName=line.get("sourceName"), device=line.get("device"), source_policy="apple_watch_only")
                db.ecg_recordings.update_one({"ecg_id": uid}, {"$setOnInsert": processed}, upsert=True)
    totals["aggregateSamples"] = len(aggregate_keys)
    totals["activitySummaries"] = len(activity_keys)
    receipt = {**totals, "user_id": user_id, "batchID": header["batchID"].lower(), "deviceID": header["deviceID"],
               "type": header["type"], "reason": header["reason"], "exportedAt": header["exportedAt"],
               "receivedAt": datetime.now(timezone.utc)}
    db.puls_batches.update_one({"user_id": user_id, "batchID": receipt["batchID"]}, {"$set": receipt}, upsert=True)
    return totals


def receive_batch(db, compressed_path, content_encoding, protocol_header, user_id):
    user_id = _uuid(user_id, "X-User-ID")
    plain_path = None
    try:
        plain_path, header, counts = decode_and_validate(compressed_path, content_encoding, protocol_header)
        return _apply(db, plain_path, header, counts, user_id)
    finally:
        if plain_path:
            try:
                os.remove(plain_path)
            except OSError:
                pass
