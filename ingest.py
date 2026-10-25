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
import os
import re
import xml.etree.ElementTree as ET
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
    "apple_stand_hour": "AppleStandHour",
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
    "breathing_disturbances": "BreathingDisturbances",
    "physical_effort": "PhysicalEffort",
    "stair_speed_up": "StairSpeedUp", "stair_speed_down": "StairSpeedDown",
    "time_in_daylight": "TimeInDaylight",
    "blood_oxygen_saturation": "OxygenSaturation", "oxygen_saturation": "OxygenSaturation",
    "apple_sleeping_wrist_temperature": "AppleSleepingWristTemperature", "wrist_temperature": "AppleSleepingWristTemperature",
    "vo2_max": "VO2Max",
    "heart_rate_recovery": "HeartRateRecoveryOneMinute", "heart_rate_recovery_one_minute": "HeartRateRecoveryOneMinute",
}

# Native Apple Health export.xml identifiers mapped to Health Auto Export names.
# XML records are individual samples and carry sourceName, so Watch-only provenance is explicit.
XML_METRIC_MAP = {
    "HKQuantityTypeIdentifierHeartRate": "heart_rate",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "heart_rate_variability",
    "HKQuantityTypeIdentifierRestingHeartRate": "resting_heart_rate",
    "HKQuantityTypeIdentifierStepCount": "step_count",
    "HKQuantityTypeIdentifierDistanceWalkingRunning": "walking_running_distance",
    "HKQuantityTypeIdentifierActiveEnergyBurned": "active_energy",
    "HKQuantityTypeIdentifierBasalEnergyBurned": "basal_energy_burned",
    "HKQuantityTypeIdentifierAppleExerciseTime": "apple_exercise_time",
    "HKQuantityTypeIdentifierAppleStandTime": "apple_stand_time",
    "HKQuantityTypeIdentifierAppleStandHour": "apple_stand_hour",
    "HKQuantityTypeIdentifierFlightsClimbed": "flights_climbed",
    "HKQuantityTypeIdentifierRespiratoryRate": "respiratory_rate",
    "HKQuantityTypeIdentifierWalkingSpeed": "walking_speed",
    "HKQuantityTypeIdentifierWalkingStepLength": "walking_step_length",
    "HKQuantityTypeIdentifierWalkingDoubleSupportPercentage": "walking_double_support_percentage",
    "HKQuantityTypeIdentifierWalkingAsymmetryPercentage": "walking_asymmetry_percentage",
    "HKQuantityTypeIdentifierAppleWalkingSteadiness": "apple_walking_steadiness",
    "HKQuantityTypeIdentifierWalkingHeartRateAverage": "walking_heart_rate_average",
    "HKQuantityTypeIdentifierEnvironmentalAudioExposure": "environmental_audio_exposure",
    "HKQuantityTypeIdentifierHeadphoneAudioExposure": "headphone_audio_exposure",
    "HKQuantityTypeIdentifierTimeInDaylight": "time_in_daylight",
    "HKQuantityTypeIdentifierOxygenSaturation": "blood_oxygen_saturation",
    "HKQuantityTypeIdentifierAppleSleepingWristTemperature": "apple_sleeping_wrist_temperature",
    "HKQuantityTypeIdentifierVO2Max": "vo2_max",
    "HKQuantityTypeIdentifierHeartRateRecoveryOneMinute": "heart_rate_recovery_one_minute",
}

SLEEP_XML_STAGES = {
    "HKCategoryValueSleepAnalysisAsleepDeep": "deep",
    "HKCategoryValueSleepAnalysisAsleepREM": "rem",
    "HKCategoryValueSleepAnalysisAsleepCore": "core",
    "HKCategoryValueSleepAnalysisAsleepUnspecified": "asleep",
    "HKCategoryValueSleepAnalysisAwake": "awake",
}

# Metrics that add up inside a 5-minute bin. Everything else is averaged.
SUM_METRICS = {"StepCount", "DistanceWalkingRunning", "ActiveEnergyBurned", "BasalEnergyBurned",
               "AppleExerciseTime", "AppleStandTime", "AppleStandHour", "FlightsClimbed", "TimeInDaylight"}
PCT_METRICS = {"OxygenSaturation", "WalkingAsymmetryPercentage", "WalkingDoubleSupportPercentage", "AppleWalkingSteadiness"}
VALUE_KEYS = ("qty", "Avg", "avg", "value", "Value")  # never substitute a minimum for a reading
ENERGY_METRICS = {"ActiveEnergyBurned", "BasalEnergyBurned"}
# Only these metric groups are real sleep data. (The old test was `"sleep" in name`, which also
# captured apple_sleeping_wrist_temperature and silently kept it out of `records`.)
SLEEP_GROUPS = {"sleep_analysis", "sleepanalysis", "sleep"}
# Nightly values whose HAE `date` is the START of the night; they belong to the wake-up date.
END_DATED = {"AppleSleepingWristTemperature"}

STAGE_MAP = {
    "deep": "Deep Sleep (Body Repair)", "rem": "REM Sleep (Brain & Memory)",
    "core": "Core Sleep (Light)", "asleep": "Core Sleep (Light)",
    "unspecified": "Core Sleep (Light)", "awake": "Awake in Bed",
}
NIGHT_STAGES = [("deep", "Deep Sleep (Body Repair)"), ("rem", "REM Sleep (Brain & Memory)"),
                ("core", "Core Sleep (Light)"), ("awake", "Awake in Bed")]
EPOCH0 = datetime(1970, 1, 1)
IST = timezone(timedelta(hours=5, minutes=30))
# Timezone assumed for timestamps that carry NO offset. Default IST. If your export sends naive UTC,
# set NAIVE_TZ_OFFSET_MIN=0 in Render and re-upload.
NAIVE_TZ = timezone(timedelta(minutes=int(os.getenv("NAIVE_TZ_OFFSET_MIN", "330"))))
_OFFSET_FIX = re.compile(r"\s*([+-])(\d{2}):?(\d{2})$")


def parse_ts(s):
    """Parse an ISO timestamp, preserving timezone information and handling 'Z'."""
    if not s:
        return None

    try:
        # Safely handle 'Z' for UTC to ensure broad Python version compatibility
        s_str = str(s).strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s_str)
        except ValueError:
            pass
        # Health Auto Export default: "2026-10-18 23:58:00 +0530". Python < 3.11 cannot parse the
        # space before the offset (or a bare +0530), so normalise it to "+05:30" and retry.
        m = _OFFSET_FIX.search(s_str)
        if m and len(s_str) > 19:
            s_str = s_str[:m.start()] + f"{m.group(1)}{m.group(2)}:{m.group(3)}"
            return datetime.fromisoformat(s_str.replace(" ", "T", 1))
        return None
    except (ValueError, TypeError):
        return None


def to_ist(dt):
    """Wall-clock datetime in IST (naive timestamps are assumed to already be IST)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=NAIVE_TZ)
    return dt.astimezone(IST).replace(tzinfo=None)


def ist_str(s):
    """Normalise any timestamp string to 'YYYY-MM-DD HH:MM:SS' IST; fall back to the first 19 chars."""
    dt = to_ist(parse_ts(s)) if s else None
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else str(s or "")[:19]

def to_epoch(dt):
    """Convert a datetime to Unix seconds without losing its timezone."""
    if dt is None:
        return None

    if dt.tzinfo is None:
        # Explicit policy for timezone-naive timestamps:
        # Default to India Standard Time (+05:30) to preserve wall-clock alignment
        # rather than defaulting to UTC and shifting the user's day by 5.5 hours.
        dt = dt.replace(tzinfo=NAIVE_TZ)

    return dt.timestamp()

def is_apple_watch_source(source):
    """Fail closed: mixed/unknown source attribution cannot establish Watch-only provenance."""
    if not isinstance(source, str) or not source.strip():
        return False
    # HAE may emit a non-breaking space in display names (Apple\u00a0Watch).
    s = re.sub(r"\s+", " ", source.replace("|", "| ").strip().casefold())
    return "apple watch" in s and "iphone" not in s and "|" not in s


def _compact(item):
    """Tiny stand-in for an item whose group name isn't known yet (keeps RAM low)."""
    if isinstance(item.get("value"), str) or any(k in item for k in ("deep", "rem", "core", "asleep")):
        return item  # sleep-like record: keep whole (there are only a few)
    ds = item.get("date") or item.get("startDate")
    for k in VALUE_KEYS:
        if item.get(k) is not None:
            return (ds, item[k], item.get("source"))
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
                    yield name, units, ({"date": b[0], "qty": b[1], "source": b[2]} if isinstance(b, tuple) else b)
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
        self.source_included = 0
        self.source_excluded = {"missing": 0, "mixed_or_iphone": 0, "other": 0}
        self.unnamed = 0
        self.skipped_stages = 0

    def add(self, name, units, item):
        self.items += 1
        if not name:
            self.unnamed += 1
            return
        source = str(item.get("source") or "").strip()
        if not is_apple_watch_source(source):
            reason = "missing" if not source else ("mixed_or_iphone" if "iphone" in source.casefold() or "|" in source else "other")
            self.source_excluded[reason] += 1
            return
        self.source_included += 1
        key = name.strip().lower().replace(" ", "_")
        if key in SLEEP_GROUPS:
            self._sleep(item, units)
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
        if metric in END_DATED and item.get("end"):
            ds = item["end"]   # attribute the nightly value to the wake-up date, like sleep.date
        try:
            val = float(val)
            dt = parse_ts(ds)
        except (ValueError, TypeError):
            return
        if dt is None or val != val:   # unparseable timestamp or NaN value: skip it, never crash the upload
            return
        u = (units or "").strip()
        ul = u.lower()
        if metric == "WalkingSpeed":
            if "m/s" in ul and "km" not in ul: val *= 3.6; u = "km/hr"
            elif ul in ("mph", "mi/hr", "mi/h"): val *= 1.609344; u = "km/hr"
        elif metric == "WalkingStepLength" and ul == "m": val *= 100.0; u = "cm"
        elif metric == "DistanceWalkingRunning":
            if ul == "m": val /= 1000.0; u = "km"
            elif ul in ("mi", "mile", "miles"): val *= 1.609344; u = "km"
        elif metric in ENERGY_METRICS and ul in ("kj", "kilojoule", "kilojoules"):
            val /= 4.184; u = "kcal"       # HAE can export energy in kJ; every engine/label expects kcal
        elif metric in PCT_METRICS:
            # HAE already sends unit "%" as 0-100. Only a unit-less fraction (0-1) needs x100.
            if "%" not in u and val <= 1.0: val *= 100.0; u = "%"

        bk = (metric, int(to_epoch(dt) // 300))
        b = self.bins.get(bk)
        if b is None:
            self.bins[bk] = [val, 1, u, {source}]
        else:
            b[0] += val
            b[1] += 1
            b[3].add(source)

    def record_docs(self):
        for (metric, ep), (total, n, unit, sources) in self.bins.items():
            # Convert absolute UTC epoch to India Standard Time (+05:30) for dashboard strings
            bs_utc = EPOCH0 + timedelta(seconds=ep * 300)
            bs_ist = bs_utc + timedelta(hours=5, minutes=30)
            yield {
                "metric": metric, "timestamp": bs_ist.strftime("%Y-%m-%d %H:%M:%S"), "date": bs_ist.strftime("%Y-%m-%d"),
                "hour": round(bs_ist.hour + bs_ist.minute / 60.0, 2), "epoch_5m": ep,
                "value": float(total if metric in SUM_METRICS else total / n), "unit": unit,
                "sources": sorted(sources), "source_policy": "apple_watch_only",
            }
          
    def _sleep(self, item, units=""):
        # Normalise to IST wall-clock so sleep timing lines up with the IST-based 5-minute records.
        s19 = ist_str(item.get("start") or item.get("startDate")) if (item.get("start") or item.get("startDate")) else ""
        e19 = ist_str(item.get("end") or item.get("endDate")) if (item.get("end") or item.get("endDate")) else ""
        if s19 and e19:
            item = dict(item)
            for ks, ke in (("start", "end"), ("startDate", "endDate")):
                if item.get(ks): item[ks] = s19
                if item.get(ke): item[ke] = e19
        d = (s19 or str(item.get("date") or item.get("sleepEnd") or ""))[:10]
        if self.keep_from and d and d < self.keep_from:
            return
        if s19 and e19:
            k = f"{s19}|{e19}|{item.get('value') or item.get('stage') or ''}"
        else:
            k = "agg|" + (d or json.dumps(item, sort_keys=True))
        self.raw[k] = {**item, "k": k, "d": d, "s19": s19, "e19": e19, "source_policy": "apple_watch_only"}

        # Old per-stage format used by the Stress and Energy engines
        if any(x in item for x in ("deep", "rem", "core", "asleep")):
            d_str = str(item.get("date") or item.get("sleepEnd") or "")[:10]
            try:
                s_raw = parse_ts(item.get("sleepStart") or f"{d_str} 00:30:00")
                e_raw = parse_ts(item.get("sleepEnd") or f"{d_str} 07:30:00")
                s_dt, e_dt = to_ist(s_raw), to_ist(e_raw)
                s_hr = s_dt.hour + s_dt.minute / 60.0
                if s_hr < 14.0: s_hr += 24.0
                e_hr = e_dt.hour + e_dt.minute / 60.0
                s_ep, e_ep = int(to_epoch(s_raw) // 300), int(to_epoch(e_raw) // 300)
                for stg_key, stg_label in NIGHT_STAGES:
                    hrs = float(item.get(stg_key) or 0.0)
                    ul = (units or "").strip().lower()
                    # trust the declared unit; only guess by magnitude when no unit is given
                    if ul.startswith("min") or (not ul and hrs > 16.0): hrs /= 60.0
                    if hrs > 0:
                        self.legacy[(stg_label, s_ep, e_ep)] = {
                            "date": d_str, "stage": stg_label, "hours": hrs, "onset_hr": s_hr,
                            "source": item.get("source"), "source_policy": "apple_watch_only",
                            "wake_hr": e_hr, "start_epoch": s_ep, "end_epoch": e_ep}
            except (ValueError, TypeError, AttributeError):
                pass
        else:
            raw_stage = str(item.get("value") or item.get("stage") or "core").lower()
            norm_stage = raw_stage.replace(" ", "").replace("_", "").replace("-", "")
            if norm_stage == "inbed":
                return   # time-in-bed marker, not a sleep stage (it is still kept in sleep_raw above)
            label = next((v for k2, v in STAGE_MAP.items() if k2 in norm_stage), None)
            if label is None:
                self.skipped_stages += 1   # unknown stage name: skip and count, never default to Core
                return
            try:
                s_raw = parse_ts(item.get("startDate") or item.get("date") or "")
                e_raw = parse_ts(item.get("endDate") or item.get("startDate") or item.get("date") or "")
                s_dt, e_dt = to_ist(s_raw), to_ist(e_raw)
                dur = (e_dt - s_dt).total_seconds() / 3600.0
                if dur <= 0: dur = float(item.get("qty") or 0.0)
                if 0 < dur < 16:
                    s_hr = s_dt.hour + s_dt.minute / 60.0
                    if s_hr < 14.0: s_hr += 24.0
                    e_hr = e_dt.hour + e_dt.minute / 60.0
                    s_ep, e_ep = int(to_epoch(s_raw) // 300), int(to_epoch(e_raw) // 300)
                    self.legacy[(label, s_ep, e_ep)] = {
                        "date": e_dt.strftime("%Y-%m-%d"), "stage": label, "hours": dur, "onset_hr": s_hr,
                        "source": item.get("source"), "source_policy": "apple_watch_only",
                        "wake_hr": e_hr, "start_epoch": s_ep, "end_epoch": e_ep}
                    self.legacy_win.append((s_ep, e_ep))
            except (ValueError, TypeError, AttributeError):
                pass


def detect_file_format(path):
    """Return xml for native Apple Health exports, otherwise json (HAE format)."""
    with open(path, "rb") as f:
        prefix = f.read(512).lstrip(b"\xef\xbb\xbf \t\r\n")
    return "xml" if prefix.startswith(b"<") else "json"


def ingest_xml_file(path, ing):
    """Stream native Apple Health export.xml and retain only individually Watch-sourced records."""
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "Record":
            attrs = elem.attrib
            rec_type = attrs.get("type", "")
            source = attrs.get("sourceName", "")
            if rec_type in XML_METRIC_MAP:
                name = XML_METRIC_MAP[rec_type]
                item = {
                    "date": attrs.get("startDate", ""),
                    "end": attrs.get("endDate", ""),
                    "qty": attrs.get("value"),
                    "source": source,
                }
                ing.add(name, attrs.get("unit", ""), item)
            elif rec_type == "HKCategoryTypeIdentifierSleepAnalysis":
                stage = SLEEP_XML_STAGES.get(attrs.get("value", ""))
                if stage:
                    ing.add("sleep_analysis", "hr", {
                        "value": stage,
                        "startDate": attrs.get("startDate", ""),
                        "endDate": attrs.get("endDate", ""),
                        "source": source,
                    })
            elem.clear()
        elif elem.tag == "Workout":
            elem.clear()
    return ing


def ingest_file(path, ing):
    if detect_file_format(path) == "xml":
        return ingest_xml_file(path, ing)
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


def _naive_ts(s):
    """'YYYY-MM-DD HH:MM:SS' -> seconds. No timezone maths: only used to order spans and measure gaps."""
    return calendar.timegm(datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").timetuple())


def _fmt_naive(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _merge_spans(spans, gap):
    """Merge (start, end) spans that are closer together than `gap`; returns [[lo, hi], ...]."""
    out = []
    for s, e in sorted(spans):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def write(db, ing, chunk=2000):
    """Write Watch-only rows, refusing to silently replace conflicting overlapping bins."""
    docs = list(ing.record_docs())

    # A later partial export can otherwise replace a complete 5-minute aggregate with only
    # the subset present in that upload. Identical re-uploads remain idempotent; conflicts
    # require a complete re-import (normally backfill.py --wipe) so we never guess at a merge.
    dates = sorted({doc["date"] for doc in docs})
    existing = {}
    if dates:
        for row in db.records.find(
            {"source_policy": "apple_watch_only", "date": {"$in": dates}},
            {"_id": 0, "metric": 1, "epoch_5m": 1, "value": 1, "sources": 1},
        ):
            existing[(row.get("metric"), row.get("epoch_5m"))] = row
    conflicts = []
    for doc in docs:
        old = existing.get((doc["metric"], doc["epoch_5m"]))
        if old is not None and (
            old.get("value") != doc["value"]
            or sorted(old.get("sources") or []) != doc["sources"]
        ):
            conflicts.append((doc["metric"], doc["timestamp"]))
    if conflicts:
        preview = ", ".join(f"{metric}@{stamp}" for metric, stamp in conflicts[:5])
        raise RuntimeError(
            f"Upload conflicts with {len(conflicts)} existing Watch-only 5-minute bins "
            f"({preview}). No sensor rows were written. Rebuild using complete exports "
            "and backfill.py --wipe, or resolve the conflicting date range first."
        )

    ops, n_rec = [], 0
    for doc in docs:
        ops.append(UpdateOne({"metric": doc["metric"], "epoch_5m": doc["epoch_5m"]}, {"$set": doc}, upsert=True))
        if len(ops) >= chunk:
            _bulk(db.records, ops); n_rec += len(ops); ops = []
    _bulk(db.records, ops); n_rec += len(ops)

    # Replace-in-place, never date-wide: remove only stored sleep rows that OVERLAP a time span this
    # payload actually contains, so a partial/windowed upload cannot erase the rest of a night.
    leg_spans = _merge_spans([(d["start_epoch"], d["end_epoch"]) for d in ing.legacy.values()], 36)  # 36 bins = 3 h
    for lo, hi in leg_spans:
        db.sleep.delete_many({"start_epoch": {"$lt": hi}, "end_epoch": {"$gt": lo}})
    raw_iv = []
    for d in ing.raw.values():
        if d.get("s19") and d.get("e19"):
            try: raw_iv.append((_naive_ts(d["s19"]), _naive_ts(d["e19"])))
            except ValueError: pass
    for lo, hi in _merge_spans(raw_iv, 3 * 3600):
        db.sleep_raw.delete_many({"s19": {"$lt": _fmt_naive(hi)}, "e19": {"$gt": _fmt_naive(lo)}})

    raw_ops = [UpdateOne({"k": d["k"]}, {"$set": d}, upsert=True) for d in ing.raw.values()]
    leg_ops = [UpdateOne({"stage": d["stage"], "start_epoch": d["start_epoch"], "end_epoch": d["end_epoch"]}, {"$set": d}, upsert=True)
               for d in ing.legacy.values()]
    for i in range(0, len(raw_ops), chunk): _bulk(db.sleep_raw, raw_ops[i:i + chunk])
    for i in range(0, len(leg_ops), chunk): _bulk(db.sleep, leg_ops[i:i + chunk])
    
    return {"items_read": ing.items, "bins_written": n_rec, "sleep_segments": len(ing.raw), "unnamed_skipped": ing.unnamed,
            "source_included": ing.source_included, "source_excluded": dict(ing.source_excluded),
            "unknown_sleep_stages_skipped": ing.skipped_stages, "source_policy": "apple_watch_only"}
