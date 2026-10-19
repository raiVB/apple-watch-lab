"""
Shared by main.py (Render) and backfill.py (your computer).

- iter_items(): streams a Health Auto Export JSON file one record at a time, so
  memory stays flat no matter how big the file is.
- Ingestor: collapses raw samples into 5-minute bins (the resolution your analysis
  engines already work at), so MongoDB stores ~50x fewer rows.
- write(): upserts everything into MongoDB in small chunks.
"""
import calendar
import json
from datetime import datetime, timedelta, timezone

import ijson
from ijson.common import ObjectBuilder
from pymongo import UpdateOne
from pymongo.errors import DuplicateKeyError, OperationFailure

METRIC_MAP = {
    "heart_rate": "HeartRate", "heartrate": "HeartRate",
    "heart_rate_variability": "HeartRateVariabilitySDNN", "heartratevariability": "HeartRateVariabilitySDNN",
    "resting_heart_rate": "RestingHeartRate", "restingheartrate": "RestingHeartRate",
    "step_count": "StepCount", "stepcount": "StepCount",
    "walking_running_distance": "DistanceWalkingRunning", "distance_walking_running": "DistanceWalkingRunning",
    "active_energy": "ActiveEnergyBurned", "active_energy_burned": "ActiveEnergyBurned",
    "basal_energy_burned": "BasalEnergyBurned", "resting_energy": "BasalEnergyBurned",
    "apple_exercise_time": "AppleExerciseTime", "exercise_time": "AppleExerciseTime",
    "apple_stand_time": "AppleStandTime", "stand_time": "AppleStandTime",
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
    "blood_oxygen_saturation": "OxygenSaturation", "oxygen_saturation": "OxygenSaturation",
    "apple_sleeping_wrist_temperature": "AppleSleepingWristTemperature", "wrist_temperature": "AppleSleepingWristTemperature",
    "vo2_max": "VO2Max",
    "heart_rate_recovery": "HeartRateRecoveryOneMinute", "heart_rate_recovery_one_minute": "HeartRateRecoveryOneMinute",
}

# Metrics that add up inside a 5-minute bin. Everything else is averaged.
SUM_METRICS = {"StepCount", "DistanceWalkingRunning", "ActiveEnergyBurned", "BasalEnergyBurned",
               "AppleExerciseTime", "AppleStandTime", "FlightsClimbed", "TimeInDaylight"}
PCT_METRICS = {"OxygenSaturation", "WalkingAsymmetryPercentage", "WalkingDoubleSupportPercentage", "AppleWalkingSteadiness"}
VALUE_KEYS = ("qty", "Avg", "avg", "value", "Value", "Min", "min")

STAGE_MAP = {
    "deep": "Deep Sleep (Body Repair)", "rem": "REM Sleep (Brain & Memory)",
    "core": "Core Sleep (Light)", "asleep": "Core Sleep (Light)",
    "unspecified": "Core Sleep (Light)", "awake": "Awake in Bed",
}
NIGHT_STAGES = [("deep", "Deep Sleep (Body Repair)"), ("rem", "REM Sleep (Brain & Memory)"),
                ("core", "Core Sleep (Light)"), ("awake", "Awake in Bed")]
EPOCH0 = datetime(1970, 1, 1)


def parse_ts(s):
    """Parse an ISO timestamp, preserving timezone information."""
    if not s:
        return None

    try:
        return datetime.fromisoformat(str(s).strip())
    except (ValueError, TypeError):
        return None


def to_epoch(dt):
    """Convert a datetime to Unix seconds without losing its timezone."""
    if dt is None:
        return None

    if dt.tzinfo is None:
        # Explicit policy for timezone-naive timestamps:
        # interpret them as UTC until their source timezone is known.
        dt = dt.replace(tzinfo=datetime.timezone.utc)

    return dt.timestamp()

def _compact(item):
    """Tiny stand-in for an item whose group name isn't known yet (keeps RAM low)."""
    if isinstance(item.get("value"), str) or any(k in item for k in ("deep", "rem", "core", "asleep")):
        return item  # sleep-like record: keep whole (there are only a few)
    ds = item.get("date") or item.get("startDate")
    for k in VALUE_KEYS:
        if item.get(k) is not None:
            return (ds, item[k])
    return None


def iter_items(fileobj):
    """Yield (group_name, units, item) one record at a time from a Health Auto Export file.

    The app writes each group's keys (name / units / data) in no fixed order. Normally name and
    units arrive before the data and records are yielded immediately. If they arrive after,
    records wait in a compact buffer and are released when the group ends."""
    name = units = ""
    got_name = got_units = False
    buf, builder = [], None
    for prefix, event, value in ijson.parse(fileobj, use_float=True):
        i = prefix.find("metrics.item")
        if i < 0:
            continue
        rest = prefix[i + 12:]
        if rest == "":
            if event in ("start_map", "end_map"):
                for b in buf:  # group finished: release anything that was waiting for its name
                    yield name, units, ({"date": b[0], "qty": b[1]} if isinstance(b, tuple) else b)
                name = units = ""
                got_name = got_units = False
                buf = []
        elif rest == ".name":
            if event == "string": name, got_name = value, True
        elif rest == ".units":
            if event == "string": units, got_units = value, True
        elif rest.startswith(".data.item"):
            if rest == ".data.item" and event == "start_map":
                builder = ObjectBuilder()
            if builder is not None:
                builder.event(event, value)
                if rest == ".data.item" and event == "end_map":
                    item, builder = builder.value, None
                    if got_name and got_units:
                        yield name, units, item
                    else:
                        c = _compact(item)
                        if c is not None:
                            buf.append(c)


class Ingestor:
    def __init__(self, keep_from=None):
        self.keep_from = keep_from   # "YYYY-MM-DD": ignore anything older
        self.bins = {}               # (metric, 5-min epoch) -> [total, count, unit]
        self.legacy = {}             # (stage, start_ep, end_ep) -> doc  (collection "sleep")
        self.legacy_win = []
        self.raw = {}                # key -> doc                        (collection "sleep_raw")
        self.items = 0
        self.unnamed = 0

    def add(self, name, units, item):
        self.items += 1
        if not name:
            self.unnamed += 1
            return
        key = name.strip().lower().replace(" ", "_")
        if "sleep" in key:
            self._sleep(item)
            return
        ds = item.get("date") or item.get("startDate")
        if not ds or (self.keep_from and str(ds)[:10] < self.keep_from):
            return
        val = None
        for k in VALUE_KEYS:
            if item.get(k) is not None:
                val = item[k]
                break
        if val is None:
            return
        metric = METRIC_MAP.get(key, name.strip())
        try:
            val = float(val)
            dt = parse_ts(ds)
        except (ValueError, TypeError):
            return
        if metric == "WalkingSpeed" and "m/s" in units: val *= 3.6
        elif metric == "WalkingStepLength" and units == "m": val *= 100.0
        elif metric == "DistanceWalkingRunning" and units == "m": val /= 1000.0
        elif metric in PCT_METRICS and val <= 1.0: val *= 100.0

        bk = (metric, to_epoch(dt) // 300)
        b = self.bins.get(bk)
        if b is None:
            self.bins[bk] = [val, 1, units]
        else:
            b[0] += val
            b[1] += 1

    def record_docs(self):
        for (metric, ep), (total, n, unit) in self.bins.items():
            bs = EPOCH0 + timedelta(seconds=ep * 300)
            yield {
                "metric": metric, "timestamp": bs.strftime("%Y-%m-%d %H:%M:%S"), "date": bs.strftime("%Y-%m-%d"),
                "hour": round(bs.hour + bs.minute / 60.0, 2), "epoch_5m": ep,
                "value": round(total if metric in SUM_METRICS else total / n, 4), "unit": unit,
            }

    def _sleep(self, item):
        s19 = str(item.get("start") or item.get("startDate") or "")[:19]
        e19 = str(item.get("end") or item.get("endDate") or "")[:19]
        d = (s19 or str(item.get("date") or item.get("sleepEnd") or ""))[:10]
        if self.keep_from and d and d < self.keep_from:
            return
        if s19 and e19:
            k = f"{s19}|{e19}|{item.get('value') or item.get('stage') or ''}"
        else:
            k = "agg|" + (d or json.dumps(item, sort_keys=True))
        self.raw[k] = {**item, "k": k, "d": d, "s19": s19, "e19": e19}

        # Old per-stage format used by the Stress and Energy engines
        if any(x in item for x in ("deep", "rem", "core", "asleep")):
            d_str = str(item.get("date") or item.get("sleepEnd") or "")[:10]
            try:
                s_dt = parse_ts(item.get("sleepStart") or f"{d_str} 00:30:00")
                e_dt = parse_ts(item.get("sleepEnd") or f"{d_str} 07:30:00")
                s_hr = s_dt.hour + s_dt.minute / 60.0
                if s_hr < 14.0: s_hr += 24.0
                e_hr = e_dt.hour + e_dt.minute / 60.0
                s_ep, e_ep = to_epoch(s_dt) // 300, to_epoch(e_dt) // 300
                for stg_key, stg_label in NIGHT_STAGES:
                    hrs = float(item.get(stg_key) or 0.0)
                    if hrs > 16.0: hrs /= 60.0
                    if hrs > 0:
                        self.legacy[(stg_label, s_ep, e_ep)] = {
                            "date": d_str, "stage": stg_label, "hours": hrs, "onset_hr": s_hr,
                            "wake_hr": e_hr, "start_epoch": s_ep, "end_epoch": e_ep}
            except (ValueError, TypeError):
                pass
        else:
            raw_stage = str(item.get("value") or item.get("stage") or "core").lower()
            label = next((v for k2, v in STAGE_MAP.items() if k2 in raw_stage), "Core Sleep (Light)")
            try:
                s_dt = parse_ts(item.get("startDate") or item.get("date") or "")
                e_dt = parse_ts(item.get("endDate") or item.get("startDate") or item.get("date") or "")
                dur = (e_dt - s_dt).total_seconds() / 3600.0
                if dur <= 0: dur = float(item.get("qty") or 0.0)
                if 0 < dur < 16:
                    s_hr = s_dt.hour + s_dt.minute / 60.0
                    if s_hr < 14.0: s_hr += 24.0
                    e_hr = e_dt.hour + e_dt.minute / 60.0
                    s_ep, e_ep = to_epoch(s_dt) // 300, to_epoch(e_dt) // 300
                    self.legacy[(label, s_ep, e_ep)] = {
                        "date": e_dt.strftime("%Y-%m-%d"), "stage": label, "hours": dur, "onset_hr": s_hr,
                        "wake_hr": e_hr, "start_epoch": s_ep, "end_epoch": e_ep}
                    self.legacy_win.append((s_ep, e_ep))
            except (ValueError, TypeError):
                pass


def ingest_file(path, ing):
    with open(path, "rb") as f:
        for name, units, item in iter_items(f):
            ing.add(name, units, item)
    return ing


def ensure_indexes(db):
    """Without these, every upsert scans the whole collection (very slow)."""
    try:
        db.records.create_index([("metric", 1), ("epoch_5m", 1)], unique=True)
        db.records.create_index("date")
        db.sleep.create_index([("stage", 1), ("start_epoch", 1), ("end_epoch", 1)], unique=True)
        db.sleep.create_index("date")
        db.sleep_raw.create_index("k", unique=True)
        db.sleep_raw.create_index("d")
    except (DuplicateKeyError, OperationFailure) as e:
        raise RuntimeError("Old raw-format rows are in the database. Run backfill.py with --wipe once. (" + str(e)[:120] + ")")


def _bulk(coll, ops):
    if ops:
        coll.bulk_write(ops, ordered=False)


def write(db, ing, chunk=2000):
    """Upsert everything the Ingestor collected. Safe to repeat: same data = same rows."""
    ops, n_rec = [], 0
    for doc in ing.record_docs():
        ops.append(UpdateOne({"metric": doc["metric"], "epoch_5m": doc["epoch_5m"]}, {"$set": doc}, upsert=True))
        if len(ops) >= chunk:
            _bulk(db.records, ops); n_rec += len(ops); ops = []
    _bulk(db.records, ops); n_rec += len(ops)

    # If a payload re-sends a stretch of sleep, replace that stretch (Apple can re-segment a night).
    segs = [d for d in ing.raw.values() if d["s19"] and d["e19"]]
    if segs:
        db.sleep_raw.delete_many({"s19": {"$gte": min(d["s19"] for d in segs)}, "e19": {"$lte": max(d["e19"] for d in segs)}})
    if ing.legacy_win:
        db.sleep.delete_many({"start_epoch": {"$gte": min(w[0] for w in ing.legacy_win)},
                              "end_epoch": {"$lte": max(w[1] for w in ing.legacy_win)}})
    raw_ops = [UpdateOne({"k": d["k"]}, {"$set": d}, upsert=True) for d in ing.raw.values()]
    leg_ops = [UpdateOne({"stage": d["stage"], "start_epoch": d["start_epoch"], "end_epoch": d["end_epoch"]}, {"$set": d}, upsert=True)
               for d in ing.legacy.values()]
    for i in range(0, len(raw_ops), chunk): _bulk(db.sleep_raw, raw_ops[i:i + chunk])
    for i in range(0, len(leg_ops), chunk): _bulk(db.sleep, leg_ops[i:i + chunk])
    return {"items_read": ing.items, "bins_written": n_rec, "sleep_segments": len(ing.raw), "unnamed_skipped": ing.unnamed}
