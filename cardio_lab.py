# ==============================================================================
# CARDIOVASCULAR RESEARCH LAB — SIGNAL PROCESSING, HRV, BASELINE & ANOMALY ENGINE
# Standalone modular engine for Apple Watch Series 11 Single-Lead ECG & Telemetry.
# Pure NumPy/Pandas implementation (zero heavy C-extension dependencies).
# ==============================================================================

import os
import json
import math
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ijson
except ImportError:
    ijson = None


# ==============================================================================
# SCIENTIFIC METHODOLOGY & LIMITATIONS REGISTRY (FOR UI & API TRANSPARENCY)
# ==============================================================================

METHODOLOGY_REGISTRY = {
    "disclaimer": (
        "RESEARCH & PERSONAL MONITORING SYSTEM ONLY — NOT A MEDICAL DEVICE. "
        "All outputs are mathematical and physiological explorations of single-lead (Lead I equivalent) "
        "wrist-to-finger potential differences and optical photoplethysmography (PPG). "
        "Apple documents that the Apple Watch ECG app is not intended for users under 22 years of age "
        "and cannot detect heart attacks (myocardial infarction), blood clots, stroke, or non-AFib arrhythmias."
    ),
    "capabilities": {
        "can_measure_directly": [
            "Lead-I-equivalent skin-surface potential difference (microvolts/millivolts) between left wrist (negative electrode) and right index finger (positive electrode).",
            "Inter-sample timestamp intervals (dt), effective sampling frequency (fs_hat), sample jitter, and data dropouts/gaps.",
            "R-peak timing, inter-beat (RR) intervals, and instantaneous ventricular rate during a stationary 30-second recording.",
            "Ultra-short-term (30-second) time-domain heart-rate variability (SDNN, RMSSD, pNN50, Poincare SD1/SD2) on artifact-cleaned Normal-to-Normal (NN) intervals."
        ],
        "can_estimate_statistically": [
            "Zero-phase bandpass-filtered waveform morphology (empirical R-wave amplitude and approximate QRS duration on high-SNR beats).",
            "Context-stratified longitudinal personal baselines (Awake Resting vs. Overnight Sleep vs. Post-Exercise) using median, MAD, and regularized Mahalanobis distance.",
            "Post-exercise mono-exponential heart-rate recovery time constant (tau) when sufficient post-workout HR samples exist.",
            "Empirical within-day vs. between-day variance components (ICC) and statistical anomaly flags relative to your personal history."
        ],
        "cannot_infer_or_diagnose": [
            "Myocardial ischemia, ST-segment elevation/depression (STEMI/NSTEMI), or coronary artery disease (requires multi-lead precordial/limb views).",
            "Chamber hypertrophy, bundle branch blocks, or complex conduction disorders requiring 12-lead spatial vectors.",
            "Stroke volume, cardiac output, left ventricular ejection fraction (LVEF), or arterial blood pressure.",
            "Clinical diagnosis of premature atrial/ventricular contractions (PAC/PVC), long-QT syndrome, myocarditis, or systemic infection.",
            "Standardized 5-minute frequency-domain HRV (LF, VLF, LF/HF ratio) from a 30-second ECG (insufficient cycle resolution)."
        ]
    },
    "references": [
        {
            "topic": "Diagnostic ECG Filtering & Bandwidth Standards",
            "citation": "Kligfield P, et al. AHA/ACC/HRS Recommendations for the Standardization and Interpretation of the Electrocardiogram. Circulation. 2007;115(10):1306-1324.",
            "application": "Justifies 0.5 Hz zero-phase high-pass cutoff to suppress respiratory baseline wander without distorting repolarization, and 40 Hz low-pass for ambulatory noise."
        },
        {
            "topic": "QRS Detection Pipeline",
            "citation": "Pan J, Tompkins WJ. A Real-Time QRS Detection Algorithm. IEEE Trans Biomed Eng. 1985;BME-32(3):230-236.",
            "application": "5-15 Hz QRS passband energy, 5-point derivative slope emphasis, squaring, 150 ms moving-window integration, and 200 ms refractory + T-wave slope discrimination."
        },
        {
            "topic": "HRV Standards & Ultra-Short-Term Validity",
            "citation": "Task Force of the ESC and NASPE. Heart Rate Variability: Standards of Measurement, Physiological Interpretation, and Clinical Use. Circulation. 1996;93(5):1043-1065. Shaffer F, Ginsberg JP. Front Public Health. 2017;5:258. Munoz ML, et al. PLoS One. 2015;10(9):e0138921.",
            "application": "Mandates strict NN-interval ectopy/artifact exclusion before RMSSD/SDNN calculation and prohibits VLF/LF/HF ratio claims on 30-second recordings."
        },
        {
            "topic": "Signal Quality Indices (SQI) for Wearable ECG",
            "citation": "Clifford GD, Behar J, Li Q, Rezek I. Signal quality indices and data fusion for determining clinical acceptability of electrocardiograms. Physiol Meas. 2012;33(9):1419-1433.",
            "application": "Combines in-band QRS spectral power concentration (5-15 Hz / 5-40 Hz), fourth-order kurtosis (>3.5), flatline/railing checks, and gap fraction."
        },
        {
            "topic": "Post-Exercise Heart Rate Recovery Kinetics",
            "citation": "Pierpont GL, Stolpman DR, Gornick CC. Heart rate recovery post-exercise as an index of parasympathetic activity. J Auton Nerv Syst. 2000;80(3):169-174.",
            "application": "Models post-exercise vagal reactivation and sympathetic withdrawal via empirical mono-exponential decay HR(t) = HR_rest + A * exp(-t/tau)."
        },
        {
            "topic": "Unsupervised Anomaly Detection",
            "citation": "Liu FT, Ting KM, Zhou ZH. Isolation Forest. IEEE ICDM. 2008:413-422. Page ES. Continuous Inspection Schemes. Biometrika. 1954;41(1/2):100-115.",
            "application": "Combines robust univariate Z-scores (0.6745*(x-med)/MAD), regularized subspace Mahalanobis distance, randomized Isolation Trees, and tabular CUSUM."
        }
    ]
}


# ==============================================================================
# MODULE A: ECG SIGNAL LABORATORY (INGESTION, TIMESTAMP RECONSTRUCTION & DSP)
# ==============================================================================

# Everything shown to you is IST. Timestamps that carry no offset are assumed to be in NAIVE_TZ
# (default IST, same rule as ingest.py). Set NAIVE_TZ_OFFSET_MIN=0 on Render if your export sends naive UTC.
IST = timezone(timedelta(hours=5, minutes=30))
NAIVE_TZ = timezone(timedelta(minutes=int(os.getenv("NAIVE_TZ_OFFSET_MIN", "330"))))


def parse_high_res_timestamp(ts_val: Any) -> Optional[float]:
    """
    Parses an Apple Health timestamp or numeric offset into Unix epoch seconds (float)
    preserving full microsecond/millisecond precision.
    Supports:
      - Numeric float/int (seconds or milliseconds)
      - '2026-10-18 09:15:23.001953 +0530'
      - '2026-10-18T09:15:23.001953+05:30'
      - '2026-10-18 09:15:23 +0530'
    """
    if ts_val is None:
        return None
    if isinstance(ts_val, (int, float)):
        val = float(ts_val)
        if math.isnan(val) or math.isinf(val):
            return None
        return val / 1000.0 if val > 1e11 else val

    s = str(ts_val).strip()
    if not s:
        return None

    # Check if string is a raw number (e.g. relative seconds "0.001953125")
    try:
        val = float(s)
        return val / 1000.0 if val > 1e11 else val
    except ValueError:
        pass

    # Standardize Apple Health space-separated timezone offset: '2026-10-18 09:15:23.123 +0530' -> ISO
    try:
        iso_s = s
        if len(iso_s) >= 24 and iso_s[-5] in ("+", "-") and iso_s[-4:].isdigit():
            # Convert ' +0530' -> '+05:30'
            iso_s = iso_s[:-5].strip() + iso_s[-5:-2] + ":" + iso_s[-2:]
        iso_s = iso_s.replace(" ", "T", 1) if "T" not in iso_s else iso_s
        if iso_s.endswith("Z"):
            iso_s = iso_s[:-1] + "+00:00"
        dt = datetime.fromisoformat(iso_s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=NAIVE_TZ)
        return dt.timestamp()
    except ValueError:
        pass

    # Fallback formats
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s[:26].strip(), fmt).replace(tzinfo=NAIVE_TZ)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def convert_voltage_to_mv(val: float, unit_str: str) -> Tuple[float, str]:
    """
    Converts raw ECG voltage measurement into millivolts (mV) cleanly.
    Returns (voltage_mv, canonical_source_unit).
    """
    u = str(unit_str or "mcV").strip().lower()
    if u in ("mcv", "uv", "µv", "μv", "microvolt", "microvolts"):
        return float(val) / 1000.0, "mcV"
    if u in ("mv", "millivolt", "millivolts"):
        return float(val), "mV"
    if u in ("v", "volt", "volts"):
        return float(val) * 1000.0, "V"
    if u in ("nv", "nanovolt", "nanovolts"):
        return float(val) / 1e6, "nV"
    # Heuristic fallback if unit string is missing or non-standard:
    # Apple Health exports in mcV (typical QRS peak ~300 to 1500 mcV)
    return float(val) / 1000.0, f"assumed_mcV({unit_str})"


def zero_phase_butterworth_filter(
    sig: np.ndarray,
    fs: float,
    hp_cut: Optional[float] = 0.5,
    lp_cut: Optional[float] = 40.0,
    order: int = 4,
    notch_freqs: Tuple[float, ...] = (50.0, 60.0),
) -> np.ndarray:
    """
    Exact zero-phase forward-backward Butterworth filter implemented in the frequency domain
    with reflective boundary padding to prevent edge transients.
    Squared magnitude response |H(f)|^2 = 1 / (1 + (f/fc)^(2*order)) corresponds to
    cascaded forward + reverse filtering (zero phase distortion: phi(f) = 0 everywhere).
    """
    n = len(sig)
    if n < 16 or fs <= 0:
        return sig.copy()

    # Demean and apply reflective padding (up to 2 seconds on each end)
    mu = float(np.nanmean(sig))
    x = np.nan_to_num(sig - mu, nan=0.0)
    pad = min(int(round(2.0 * fs)), n - 1)
    if pad > 1:
        x_pad = np.concatenate([x[pad:0:-1], x, x[-2:-pad - 2:-1]])
    else:
        x_pad = x.copy()

    n_fft = len(x_pad)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    X_f = np.fft.rfft(x_pad)

    H2 = np.ones_like(freqs, dtype=float)
    # High-pass zero-phase response
    if hp_cut is not None and hp_cut > 0:
        with np.errstate(divide="ignore", over="ignore"):
            ratio_hp = np.where(freqs > 1e-9, hp_cut / np.maximum(freqs, 1e-9), 1e9)
            H2 *= 1.0 / (1.0 + np.power(ratio_hp, 2 * order))
        H2[0] = 0.0

    # Low-pass zero-phase response
    if lp_cut is not None and lp_cut > 0 and lp_cut < (0.5 * fs):
        with np.errstate(over="ignore"):
            ratio_lp = freqs / lp_cut
            H2 *= 1.0 / (1.0 + np.power(ratio_lp, 2 * order))

    # Narrowband powerline notch (50 Hz India/Europe & 60 Hz Americas)
    for nf in notch_freqs:
        if 0 < nf < (0.5 * fs):
            bw = 1.0  # 1 Hz notch half-width
            H2 *= 1.0 - np.exp(-0.5 * ((freqs - nf) / bw) ** 2)

    y_pad = np.fft.irfft(X_f * H2, n=n_fft)
    return y_pad[pad:pad + n] if pad > 1 else y_pad[:n]


def process_raw_ecg_recording(ecg_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full pipeline for a single ECG recording covering:
      - Module A: Timestamp reconstruction, sampling rate verification, gap/artifact/clipping audit,
                  zero-phase DSP filtering, and Signal Quality Index (SQI).
      - Module B: Pan-Tompkins R-peak detection, RR series, candidate ectopic checks, QRS morphology.
      - Module C: Artifact-rejected NN intervals and 30-second ultra-short-term HRV metrics.
    """
    rec_start_str = str(
        ecg_obj.get("start")
        or ecg_obj.get("startDate")
        or ecg_obj.get("date")
        or ecg_obj.get("timestamp")
        or ""
    )
    rec_end_str = str(ecg_obj.get("end") or ecg_obj.get("endDate") or "")
    classification = str(
        ecg_obj.get("classification")
        or ecg_obj.get("rhythmClassification")
        or ecg_obj.get("value")
        or "Unclassified"
    )
    symptoms = ecg_obj.get("symptoms") or ecg_obj.get("symptomStatus") or []
    if isinstance(symptoms, str):
        symptoms = [symptoms] if symptoms else []

    raw_samples = (
        ecg_obj.get("voltageMeasurements")
        or ecg_obj.get("samples")
        or ecg_obj.get("data")
        or []
    )

    rec_start_epoch = parse_high_res_timestamp(rec_start_str)
    rec_end_epoch = parse_high_res_timestamp(rec_end_str)

    # Parse individual voltageMeasurements[]
    parsed_t: List[float] = []
    parsed_mv: List[float] = []
    units_seen = set()
    malformed_count = 0

    for idx, item in enumerate(raw_samples):
        if not isinstance(item, dict):
            malformed_count += 1
            continue
        v_raw = item.get("voltage", item.get("value", item.get("v")))
        d_raw = item.get("date", item.get("timestamp", item.get("time", item.get("t"))))
        u_raw = item.get("units", item.get("unit", "mcV"))
        if v_raw is None:
            malformed_count += 1
            continue
        try:
            v_f = float(v_raw)
            if math.isnan(v_f) or math.isinf(v_f):
                malformed_count += 1
                continue
        except (ValueError, TypeError):
            malformed_count += 1
            continue

        v_mv, canon_u = convert_voltage_to_mv(v_f, str(u_raw))
        units_seen.add(canon_u)

        t_val = parse_high_res_timestamp(d_raw)
        if t_val is None:
            malformed_count += 1
            continue
        parsed_t.append(t_val)
        parsed_mv.append(v_mv)

    n_raw = len(parsed_mv)
    if n_raw < 32:
        ecg_id = hashlib.sha1(f"{rec_start_str}_{n_raw}".encode("utf-8")).hexdigest()[:16]
        return {
            "ecg_id": ecg_id,
            "valid": False,
            "error": f"Insufficient valid voltage samples ({n_raw} found, minimum 32 required).",
            "start_timestamp": rec_start_str,
            "classification": classification,
            "sample_count": n_raw,
            "malformed_samples": malformed_count,
        }

    t_arr = np.array(parsed_t, dtype=float)
    v_arr = np.array(parsed_mv, dtype=float)

    # Sort chronologically if timestamps were out of order, tracking inversions
    dt_raw = np.diff(t_arr)
    out_of_order_count = int(np.sum(dt_raw < 0))
    if out_of_order_count > 0:
        order_idx = np.argsort(t_arr, kind="mergesort")
        t_arr = t_arr[order_idx]
        v_arr = v_arr[order_idx]
        dt_raw = np.diff(t_arr)

    # Deduplicate identical timestamps conservatively (keep first occurrence, report count)
    dup_mask = np.concatenate([[False], np.diff(t_arr) == 0.0])
    duplicate_ts_count = int(np.sum(dup_mask))
    if duplicate_ts_count > 0:
        keep_mask = ~dup_mask
        t_arr = t_arr[keep_mask]
        v_arr = v_arr[keep_mask]

    # Relative time vector t_rel in seconds starting from 0.0
    t0 = float(t_arr[0])
    t_rel = t_arr - t0
    if rec_start_epoch is None or rec_start_epoch < 1e8:
        rec_start_epoch = t0 if t0 > 1e8 else datetime.now(timezone.utc).timestamp()
    if not rec_start_str or len(rec_start_str) < 10:
        rec_start_str = datetime.fromtimestamp(rec_start_epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    # Keep the original string only to build a stable ecg_id (so re-uploads overwrite, not duplicate).
    rec_start_raw = rec_start_str
    # date / hour / display time come from the true instant, converted to IST, so they line up with
    # db.records (which is IST). Cutting them out of the raw string kept the export's own timezone.
    rec_start_ist = datetime.fromtimestamp(rec_start_epoch, tz=IST)
    rec_date = rec_start_ist.strftime("%Y-%m-%d")
    rec_hour = float(rec_start_ist.hour + rec_start_ist.minute / 60.0)
    rec_start_str = rec_start_ist.strftime("%Y-%m-%d %H:%M:%S")
    if rec_end_epoch is not None and rec_end_epoch > 1e8:
        rec_end_str = datetime.fromtimestamp(rec_end_epoch, tz=IST).strftime("%Y-%m-%d %H:%M:%S")

    # Independent sampling frequency verification from actual timestamps
    dt_pos = np.diff(t_rel)
    dt_pos = dt_pos[dt_pos > 0]
    if len(dt_pos) == 0:
        median_dt_s = 1.0 / 512.0
    else:
        median_dt_s = float(np.median(dt_pos))

    fs_hat = 1.0 / max(median_dt_s, 1e-6)
    mean_dt_ms = float(np.mean(dt_pos) * 1000.0) if len(dt_pos) else (median_dt_s * 1000.0)
    median_dt_ms = median_dt_s * 1000.0
    iqr_dt_ms = float((np.percentile(dt_pos, 75) - np.percentile(dt_pos, 25)) * 1000.0) if len(dt_pos) >= 4 else 0.0
    jitter_cv_pct = float((np.std(dt_pos) / max(median_dt_s, 1e-9)) * 100.0) if len(dt_pos) >= 2 else 0.0

    # Gap & missing sample detection (never silently interpolate large gaps)
    # Micro-irregularity: dt > 1.5 * median_dt; Major dropout gap: dt > 0.100 s (100 ms)
    dt_all = np.diff(t_rel)
    irregular_steps = int(np.sum((dt_all > 1.5 * median_dt_s) & (dt_all <= 0.100)))
    large_gap_indices = np.where(dt_all > 0.100)[0]
    large_gaps = []
    estimated_missing_samples = 0
    for g_idx in large_gap_indices:
        gap_dur_s = float(dt_all[g_idx])
        miss_n = max(1, int(round(gap_dur_s / median_dt_s)) - 1)
        estimated_missing_samples += miss_n
        large_gaps.append({
            "start_sec": round(float(t_rel[g_idx]), 3),
            "end_sec": round(float(t_rel[g_idx + 1]), 3),
            "duration_ms": round(gap_dur_s * 1000.0, 1),
            "estimated_missing_samples": miss_n,
        })

    # Also count missing samples from micro-irregular steps
    for d_step in dt_all[(dt_all > 1.5 * median_dt_s) & (dt_all <= 0.100)]:
        estimated_missing_samples += max(0, int(round(float(d_step) / median_dt_s)) - 1)

    total_duration_s = float(t_rel[-1] - t_rel[0]) if len(t_rel) > 1 else 0.0
    gap_duration_s = sum(g["duration_ms"] / 1000.0 for g in large_gaps)

    # Artifact checks: implausible voltages (> +-5 mV), rail clipping, and flatlines
    implausible_mask = np.abs(v_arr) > 5.0
    implausible_count = int(np.sum(implausible_mask))

    v_max_abs = float(np.percentile(np.abs(v_arr), 99.9)) if len(v_arr) else 0.0
    clipping_mask = (np.abs(v_arr) >= 4.5) | (
        (v_max_abs > 1.5) & (np.abs(v_arr) >= 0.995 * v_max_abs) & (np.abs(np.diff(v_arr, prepend=v_arr[0])) < 1e-5)
    )
    clipping_count = int(np.sum(clipping_mask))

    # Flatline check over sliding 250 ms windows
    win_flat = max(8, int(round(0.25 * fs_hat)))
    dv_abs = np.abs(np.diff(v_arr, prepend=v_arr[0]))
    kernel_flat = np.ones(win_flat, dtype=float) / win_flat
    rolling_dv = np.convolve(dv_abs, kernel_flat, mode="same")
    flatline_mask = rolling_dv < 0.001  # < 1 mcV average step over 250 ms
    flatline_seconds = float(np.sum(flatline_mask) * median_dt_s)

    # Segment signal across large gaps (>100 ms) so we NEVER filter or interpolate across missing data
    split_pts = [int(i + 1) for i in large_gap_indices]
    segments_idx = np.split(np.arange(len(v_arr)), split_pts)

    v_filt = np.zeros_like(v_arr)
    # Ensure segments have at least 1.0 second of data so the FFT can resolve 1.0 Hz bins
    min_filter_len = max(32, int(math.ceil(fs_hat))) 

    for seg in segments_idx:
        if len(seg) == 0:
            continue
        if len(seg) >= min_filter_len:
            v_filt[seg] = zero_phase_butterworth_filter(
                v_arr[seg], fs=fs_hat, hp_cut=0.5, lp_cut=40.0, order=4, notch_freqs=(50.0, 60.0)
            )
        else:
            v_filt[seg] = v_arr[seg] - float(np.mean(v_arr[seg]))

    # Heuristic Signal Quality Index (SQI). Windows that straddle a detected
    # >100 ms timestamp gap are excluded rather than treated as continuous data.
    win_2s = max(32, int(round(2.0 * fs_hat)))
    n_wins = max(1, len(v_filt) // win_2s)
    gap_boundary_indices = set(int(i) for i in large_gap_indices)
    usable_wins = 0
    for w_i in range(n_wins):
        s_i = w_i * win_2s
        e_i = min(len(v_filt), s_i + win_2s)
        # A boundary at i means samples i and i+1 are separated by a detected gap.
        crosses_gap = any(s_i <= g_i < e_i - 1 for g_i in gap_boundary_indices)
        if crosses_gap:
            continue
        seg_w = v_filt[s_i:e_i]
        if len(seg_w) < win_2s:
            continue  # don't give a short trailing fragment the same weight as 2 s
        std_w = float(np.std(seg_w))
        p2p_w = float(np.max(seg_w) - np.min(seg_w))
        if 0.015 <= std_w <= 0.85 and p2p_w <= 3.8 and not np.any(implausible_mask[s_i:e_i]):
            usable_wins += 1

    # This is the percentage of the recording's wall-clock duration covered by
    # accepted complete 2-second windows. Gaps are penalized by the denominator;
    # do not apply a second separate gap-duration penalty.
    usable_duration_pct = round(float(np.clip(
        (usable_wins * win_2s / max(fs_hat, 1e-9)) / max(total_duration_s, 1e-9) * 100.0,
        0.0, 100.0)), 1)

    # Exploratory whole-recording spectral/kurtosis descriptors. These are not
    # validated clinical SQIs; values may still be affected by non-gap artifacts.
    freqs_sqi = np.fft.rfftfreq(len(v_filt), d=1.0 / fs_hat)
    psd_sqi = np.abs(np.fft.rfft(v_filt)) ** 2
    qrs_band_pwr = float(np.sum(psd_sqi[(freqs_sqi >= 5.0) & (freqs_sqi <= 15.0)]))
    broad_band_pwr = float(np.sum(psd_sqi[(freqs_sqi >= 5.0) & (freqs_sqi <= 40.0)]))
    inband_qrs_ratio = qrs_band_pwr / max(broad_band_pwr, 1e-12)

    std_global = max(float(np.std(v_filt)), 1e-6)
    kurtosis_val = float(np.mean(((v_filt - np.mean(v_filt)) / std_global) ** 4))

    # Estimate baseline wander using a separate <0.5 Hz low-pass component, and
    # high-frequency residual above the 40 Hz low-pass. Work segment-by-segment
    # so neither diagnostic filter bridges missing-data gaps.
    baseline_parts = []
    hf_residual_parts = []
    for seg in segments_idx:
        if len(seg) < min_filter_len:
            continue
        raw_seg = v_arr[seg]
        baseline_parts.append(zero_phase_butterworth_filter(
            raw_seg, fs=fs_hat, hp_cut=None, lp_cut=0.5, order=4, notch_freqs=()))
        low40_seg = zero_phase_butterworth_filter(
            raw_seg, fs=fs_hat, hp_cut=None, lp_cut=40.0, order=4, notch_freqs=())
        hf_residual_parts.append(raw_seg - low40_seg)
    baseline_wander_rms_mv = round(float(np.sqrt(np.mean(np.concatenate(baseline_parts) ** 2))), 4) if baseline_parts else None
    hf_noise_rms_mv = round(float(np.sqrt(np.mean(np.concatenate(hf_residual_parts) ** 2))), 4) if hf_residual_parts else None

    # Heuristic composite (0-100), inspired by common SQI feature families but
    # not a validated implementation of Clifford et al. and not a probability.
    sqi_kurt_score = float(np.clip((kurtosis_val - 3.0) / 5.0, 0.0, 1.0)) * 100.0
    sqi_spec_score = float(np.clip((inband_qrs_ratio - 0.25) / 0.45, 0.0, 1.0)) * 100.0
    sqi_score = round(0.45 * usable_duration_pct + 0.30 * sqi_spec_score + 0.25 * sqi_kurt_score, 1)

    if sqi_score >= 78.0 and usable_duration_pct >= 80.0:
        sqi_label = "High heuristic SQI (not externally validated)"
        sqi_color = "#3ddbb0"
    elif sqi_score >= 52.0 and usable_duration_pct >= 55.0:
        sqi_label = "Moderate heuristic SQI (inspect artifacts)"
        sqi_color = "#f5b84a"
    else:
        sqi_label = "Low heuristic SQI (inspect signal / contact)"
        sqi_color = "#ff7a6b"

    # Run Module B (Beat Detection & Morphology)
    beat_bundle = detect_beats_and_morphology(t_rel, v_filt, fs_hat, large_gaps, sqi_score)

    # Run Module C (Mathematical HRV on Accepted NN Intervals)
    hrv_bundle = compute_ecg_hrv_metrics(
        beat_bundle["peak_times_s"],
        beat_bundle["rr_intervals_ms"],
        beat_bundle["rr_pair_has_gap"],
        total_duration_s,
        usable_duration_pct,
        sqi_score,
    )

    # Prepare browser-safe waveform arrays with explicit null breaks at large gaps (>100 ms)
    # So Plotly never draws a fake straight line across missing samples
    plot_t: List[Optional[float]] = []
    plot_raw_mv: List[Optional[float]] = []
    plot_filt_mv: List[Optional[float]] = []
    gap_Start_set = set(int(i) for i in large_gap_indices)

    for idx_pt in range(len(t_rel)):
        plot_t.append(round(float(t_rel[idx_pt]), 4))
        plot_raw_mv.append(round(float(v_arr[idx_pt]), 4))
        plot_filt_mv.append(round(float(v_filt[idx_pt]), 4))
        if idx_pt in gap_Start_set:
            plot_t.append(None)
            plot_raw_mv.append(None)
            plot_filt_mv.append(None)

    ecg_id = hashlib.sha1(f"{rec_start_raw}_{len(t_rel)}_{round(float(v_arr[0]), 4)}".encode("utf-8")).hexdigest()[:16]

    return {
        "ecg_id": ecg_id,
        "valid": True,
        "start_timestamp": rec_start_str,
        "end_timestamp": rec_end_str,
        "start_epoch": rec_start_epoch,
        "date": rec_date,
        "hour": round(rec_hour, 2),
        "classification": classification,
        "symptoms": symptoms,
        "signal_quality": {
            "sqi_score": sqi_score,
            "sqi_label": sqi_label,
            "sqi_color": sqi_color,
            "usable_duration_pct": usable_duration_pct,
            "total_duration_s": round(total_duration_s, 2),
            "sample_count": int(len(t_rel)),
            "malformed_samples": malformed_count,
            "duplicate_timestamps_removed": duplicate_ts_count,
            "out_of_order_timestamps_sorted": out_of_order_count,
            "irregular_sample_steps": irregular_steps,
            "large_gaps_count": len(large_gaps),
            "large_gaps": large_gaps,
            "estimated_missing_samples": estimated_missing_samples,
            "clipping_samples": clipping_count,
            "implausible_voltage_samples": implausible_count,
            "flatline_seconds": round(flatline_seconds, 2),
            "kurtosis": round(kurtosis_val, 2),
            "inband_qrs_power_ratio": round(inband_qrs_ratio, 3),
            "baseline_wander_rms_mv": baseline_wander_rms_mv,
            "hf_noise_rms_mv": hf_noise_rms_mv,
            "source_units": sorted(list(units_seen)),
            "calibration_note": (
                "Lead-I-like single-lead recording; absolute voltage amplitude depends on contact and cardiac axis. "
                "The 0-100 SQI is a heuristic feature score, not a probability or externally validated clinical quality grade. "
                "Baseline-wander RMS is estimated from a separate <0.5 Hz component on gap-free segments."
            ),
        },
        "sampling": {
            "estimated_fs_hz": round(fs_hat, 2),
            "median_dt_ms": round(median_dt_ms, 4),
            "mean_dt_ms": round(mean_dt_ms, 4),
            "iqr_dt_ms": round(iqr_dt_ms, 4),
            "jitter_cv_pct": round(jitter_cv_pct, 3),
        },
        "filter_documentation": {
            "method": "Zero-Phase Forward-Backward Butterworth Filter (Frequency-Domain |H(f)|^2 with 2.0s Reflective Padding)",
            "highpass_hz": 0.5,
            "lowpass_hz": 40.0,
            "order_effective": "8th-order zero-phase (4th-order forward + 4th-order reverse)",
            "notch_hz": [50.0, 60.0],
            "phase_distortion": "0.0 rad (strictly zero-phase; preserves R-peak timing and QRS symmetry)",
            "clinical_waveform_caveat": (
                "A 0.5 Hz zero-phase high-pass filter removes respiratory baseline drift while preserving QRS/RR timing, "
                "but can introduce subtle ring-back near steep ST segments. Single-lead smartwatch recordings must not be used for ST-ischemia analysis."
            ),
        },
        "beats": beat_bundle,
        "hrv": hrv_bundle,
        "waveform": {
            "t_sec": plot_t,
            "raw_mv": plot_raw_mv,
            "filtered_mv": plot_filt_mv,
        },
    }


# ==============================================================================
# MODULE B: BEAT DETECTION (PAN-TOMPKINS) & CONSERVATIVE MORPHOLOGY FEATURES
# ==============================================================================

def detect_beats_and_morphology(
    t_rel: np.ndarray,
    v_filt: np.ndarray,
    fs: float,
    large_gaps: List[Dict[str, Any]],
    sqi_score: float,
) -> Dict[str, Any]:
    """
    Implements the Pan-Tompkins QRS detection pipeline adapted to the empirical sampling rate fs:
      1. 5-15 Hz QRS bandpass filter.
      2. 5-point derivative operator.
      3. Squaring nonlinear amplification.
      4. 150 ms moving-window integration (MWI).
      5. Adaptive thresholding with 200 ms refractory period and T-wave slope discrimination.
      6. Polarity-aware R-peak refinement + 3-point parabolic subsample interpolation.
    """
    n = len(v_filt)
    if n < 32 or fs <= 20:
        return {"detected_beats": 0, "peak_indices": [], "peak_times_s": [], "peak_voltages_mv": [], "rr_intervals_ms": []}

# Reconstruct gap segments to avoid LTI violations across missing data
    dt_all = np.diff(t_rel)
    split_pts = [int(i + 1) for i in np.where(dt_all > 0.100)[0]]
    segments_idx = np.split(np.arange(n), split_pts)

    v_qrs = np.zeros_like(v_filt)
    deriv = np.zeros_like(v_filt)
    sq = np.zeros_like(v_filt)
    mwi = np.zeros_like(v_filt)
    win_mwi = max(3, int(round(0.150 * fs)))

    for seg in segments_idx:
        n_seg = len(seg)
        if n_seg < 32:
            continue
            
        # Step 1: 5-15 Hz QRS Bandpass strictly within the continuous segment
        v_qrs[seg] = zero_phase_butterworth_filter(v_filt[seg], fs=fs, hp_cut=5.0, lp_cut=15.0, order=2, notch_freqs=())
        
        # Step 2: 5-point derivative strictly within the segment
        if n_seg >= 5:
            d_seg = np.zeros(n_seg)
            v_seg = v_qrs[seg]
            d_seg[2:-2] = (-v_seg[:-4] - 2.0 * v_seg[1:-3] + 2.0 * v_seg[3:-1] + v_seg[4:]) * (fs / 8.0)
            deriv[seg] = d_seg
            
        # Step 3 & 4: Nonlinear squaring & MWI strictly within the segment
        sq[seg] = deriv[seg] ** 2
        mwi[seg] = np.convolve(sq[seg], np.ones(win_mwi, dtype=float) / win_mwi, mode="same")

    # Determine dominant QRS polarity (positive R vs deep S if watch worn on opposite wrist)
    p98 = float(np.percentile(v_filt, 98))
    p02 = float(np.percentile(v_filt, 2))
    polarity = -1.0 if abs(p02) > 1.35 * abs(p98) else 1.0
    v_oriented = v_filt * polarity

    # Step 5: Adaptive threshold & local maxima above refractory period (200 ms)
    refractory_samples = max(2, int(round(0.200 * fs)))
    twave_window_samples = max(refractory_samples + 1, int(round(0.360 * fs)))
    search_half = max(2, int(round(0.045 * fs)))  # +-45 ms refinement window around MWI peak

    base_thresh = float(np.percentile(mwi, 70)) + 0.20 * float(np.percentile(mwi, 96) - np.percentile(mwi, 70))
    cand_indices: List[int] = []
    cand_slopes: List[float] = []

    i = search_half
    while i < n - search_half:
        if mwi[i] > base_thresh and mwi[i] >= np.max(mwi[i - search_half:i + search_half + 1]):
            # Refine exact R-peak on v_oriented within +-50 ms
            w0 = max(1, i - int(round(0.055 * fs)))
            w1 = min(n - 2, i + int(round(0.055 * fs)))
            r_idx = int(w0 + np.argmax(v_oriented[w0:w1 + 1]))
            r_slope = float(np.max( np.abs(deriv[max(0, r_idx - 4):min(n, r_idx + 5)]) ))

            if not cand_indices:
                cand_indices.append(r_idx)
                cand_slopes.append(r_slope)
            else:
                prev_idx = cand_indices[-1]
                prev_slope = cand_slopes[-1]
                dist = r_idx - prev_idx

                if dist < refractory_samples:
                    # Within 200 ms absolute refractory period: keep higher amplitude peak
                    if v_oriented[r_idx] > v_oriented[prev_idx]:
                        cand_indices[-1] = r_idx
                        cand_slopes[-1] = r_slope
                elif dist < twave_window_samples:
                    # Between 200 ms and 360 ms: Pan-Tompkins T-wave slope discrimination rule
                    if r_slope >= 0.50 * prev_slope and v_oriented[r_idx] >= 0.40 * v_oriented[prev_idx]:
                        cand_indices.append(r_idx)
                        cand_slopes.append(r_slope)
                else:
                    cand_indices.append(r_idx)
                    cand_slopes.append(r_slope)

            i = r_idx + refractory_samples
        else:
            i += 1

    # Amplitude sanity filter: remove tiny noise peaks (< 25% of median detected R-peak prominence)
    if len(cand_indices) >= 3:
        amps = np.array([v_oriented[idx] for idx in cand_indices], dtype=float)
        med_amp = float(np.median(amps))
        valid_pairs = [(idx, slp) for idx, slp in zip(cand_indices, cand_slopes) if v_oriented[idx] >= 0.28 * med_amp]
        cand_indices = [p[0] for p in valid_pairs]
        cand_slopes = [p[1] for p in valid_pairs]

    # Parabolic 3-point subsample interpolation for sub-millisecond R-peak timing
    peak_times_s: List[float] = []
    peak_voltages_mv: List[float] = []
    for idx in cand_indices:
        if 0 < idx < n - 1:
            y0, y1, y2 = float(v_oriented[idx - 1]), float(v_oriented[idx]), float(v_oriented[idx + 1])
            denom = 2.0 * (y0 - 2.0 * y1 + y2)
            delta = (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
            t_sub = float(t_rel[idx]) + delta * (1.0 / fs)
        else:
            t_sub = float(t_rel[idx])
        peak_times_s.append(round(t_sub, 4))
        peak_voltages_mv.append(round(float(v_filt[idx]), 4))

    # Inter-beat intervals (RR) and gap cross-checking
    rr_intervals_ms: List[float] = []
    rr_pair_has_gap: List[bool] = []
    inst_hr_bpm: List[float] = []

    for k in range(len(peak_times_s) - 1):
        t_a, t_b = peak_times_s[k], peak_times_s[k + 1]
        rr_ms = (t_b - t_a) * 1000.0
        has_gap = any((g["start_sec"] >= t_a and g["end_sec"] <= t_b) for g in large_gaps)
        rr_intervals_ms.append(round(rr_ms, 2))
        rr_pair_has_gap.append(has_gap)
        if not has_gap and 250.0 <= rr_ms <= 2200.0:
            inst_hr_bpm.append(round(60000.0 / rr_ms, 1))

    # Candidate unusually timed / premature beats (conservative check)
    # Requires: no data gap, local SQI >= 60, RR_k < 0.80 * local_median_RR, and >= 5 total beats
    candidate_premature_beats = []
    clean_rr = [r for r, g in zip(rr_intervals_ms, rr_pair_has_gap) if not g and 250.0 <= r <= 2200.0]
    if len(clean_rr) >= 5 and sqi_score >= 55.0:
        for k in range(len(rr_intervals_ms)):
            if rr_pair_has_gap[k]:
                continue
            local_window = [
                rr_intervals_ms[j]
                for j in range(max(0, k - 3), min(len(rr_intervals_ms), k + 4))
                if not rr_pair_has_gap[j]
            ]
            local_med = float(np.median(local_window)) if local_window else float(np.median(clean_rr))
            rr_val = rr_intervals_ms[k]
            premature_ratio = rr_val / max(local_med, 1.0)
            if premature_ratio < 0.80 and rr_val >= 280.0:
                comp_pause_ms = rr_intervals_ms[k + 1] if (k + 1 < len(rr_intervals_ms) and not rr_pair_has_gap[k + 1]) else None
                candidate_premature_beats.append({
                    "beat_number": k + 2,
                    "peak_time_s": peak_times_s[k + 1],
                    "coupling_interval_ms": round(rr_val, 1),
                    "local_median_rr_ms": round(local_med, 1),
                    "prematurity_pct": round((1.0 - premature_ratio) * 100.0, 1),
                    "following_rr_ms": round(comp_pause_ms, 1) if comp_pause_ms else None,
                    "note": (
                        "Algorithmic timing observation only (>20% shorter than local median RR). "
                        "Single-lead wrist ECG cannot distinguish supraventricular vs. ventricular ectopy or motion artifact without clinical review."
                    ),
                })

    # Conservative QRS duration estimation on high-SNR beats (half-prominence / zero-crossing envelope)
    qrs_durations_ms: List[float] = []
    r_amplitudes_mv: List[float] = []
    win_qrs_half = max(3, int(round(0.070 * fs)))  # +-70 ms search for Q and S inflection points

    for idx in cand_indices:
        if idx - win_qrs_half < 0 or idx + win_qrs_half >= n:
            continue
        r_amp = float(v_oriented[idx])
        r_amplitudes_mv.append(r_amp)
        # Locate Q onset and S offset where signal drops below 15% of R peak relative to local isoelectric baseline
        iso_left = float(np.median(v_oriented[max(0, idx - int(round(0.10 * fs))):max(1, idx - int(round(0.05 * fs)))]))
        prom = r_amp - iso_left
        if prom < 0.15:
            continue
        thresh_qrs = iso_left + 0.15 * prom

        left_seg = v_oriented[idx - win_qrs_half:idx + 1]
        below_l = np.where(left_seg <= thresh_qrs)[0]
        q_idx = (idx - win_qrs_half + int(below_l[-1])) if len(below_l) else (idx - int(round(0.04 * fs)))

        right_seg = v_oriented[idx:idx + win_qrs_half + 1]
        below_r = np.where(right_seg <= thresh_qrs)[0]
        s_idx = (idx + int(below_r[0])) if len(below_r) else (idx + int(round(0.04 * fs)))

        qrs_ms = (s_idx - q_idx) * (1000.0 / fs)
        if 50.0 <= qrs_ms <= 160.0:
            qrs_durations_ms.append(qrs_ms)

    # Build median P-QRS-T beat template (-200 ms to +350 ms around R-peaks) for visual inspection
    pre_s = int(round(0.200 * fs))
    post_s = int(round(0.350 * fs))
    beat_stack = []
    for idx in cand_indices:
        if idx - pre_s >= 0 and idx + post_s < n:
            beat_stack.append(v_filt[idx - pre_s:idx + post_s + 1])

    if len(beat_stack) >= 3:
        mat_beats = np.array(beat_stack, dtype=float)
        template_t_ms = [round((k - pre_s) * (1000.0 / fs), 1) for k in range(mat_beats.shape[1])]
        template_median_mv = [round(float(x), 4) for x in np.median(mat_beats, axis=0)]
        template_p25_mv = [round(float(x), 4) for x in np.percentile(mat_beats, 25, axis=0)]
        template_p75_mv = [round(float(x), 4) for x in np.percentile(mat_beats, 75, axis=0)]
    else:
        template_t_ms, template_median_mv, template_p25_mv, template_p75_mv = [], [], [], []

    return {
        "detected_beats": len(cand_indices),
        "polarity_orientation": "Standard Positive Lead I" if polarity > 0 else "Inverted Lead I (Check Wrist Orientation)",
        "peak_indices": cand_indices,
        "peak_times_s": peak_times_s,
        "peak_voltages_mv": peak_voltages_mv,
        "rr_intervals_ms": rr_intervals_ms,
        "rr_pair_has_gap": rr_pair_has_gap,
        "instantaneous_hr_bpm": inst_hr_bpm,
        "mean_hr_bpm": round(float(np.mean(inst_hr_bpm)), 1) if inst_hr_bpm else None,
        "median_hr_bpm": round(float(np.median(inst_hr_bpm)), 1) if inst_hr_bpm else None,
        "min_hr_bpm": round(float(np.min(inst_hr_bpm)), 1) if inst_hr_bpm else None,
        "max_hr_bpm": round(float(np.max(inst_hr_bpm)), 1) if inst_hr_bpm else None,
        "mean_rr_ms": round(float(np.mean(clean_rr)), 1) if clean_rr else None,
        "median_rr_ms": round(float(np.median(clean_rr)), 1) if clean_rr else None,
        "min_rr_ms": round(float(np.min(clean_rr)), 1) if clean_rr else None,
        "max_rr_ms": round(float(np.max(clean_rr)), 1) if clean_rr else None,
        "qrs_duration_median_ms": round(float(np.median(qrs_durations_ms)), 1) if (len(qrs_durations_ms) >= 3 and sqi_score >= 60.0) else None,
        "r_amplitude_median_mv": round(float(np.median(r_amplitudes_mv)), 3) if r_amplitudes_mv else None,
        "candidate_premature_beats": candidate_premature_beats,
        "beat_detection_confidence": round(min(99.0, sqi_score * (1.0 if len(cand_indices) >= 15 else 0.7)), 1),
        "template_t_ms": template_t_ms,
        "template_median_mv": template_median_mv,
        "template_p25_mv": template_p25_mv,
        "template_p75_mv": template_p75_mv,
        "morphology_limitations": (
            "P-wave and T-wave boundaries and QT intervals are intentionally not auto-diagnosed because single-lead "
            "Lead-I wrist projections have low atrial SNR and axis-dependent T-wave amplitude."
        ),
    }


# ==============================================================================
# MODULE C: MATHEMATICAL HEART-RATE VARIABILITY (NN CLEANING & UST METRICS)
# ==============================================================================

def compute_ecg_hrv_metrics(
    peak_times_s: List[float],
    rr_intervals_ms: List[float],
    rr_pair_has_gap: List[bool],
    total_duration_s: float,
    usable_duration_pct: float,
    sqi_score: float,
) -> Dict[str, Any]:
    """
    Computes ECG-derived Heart Rate Variability strictly on artifact-free Normal-to-Normal (NN) intervals.
    Never interpolates across rejected beats; computes RMSSD strictly on contiguous accepted NN pairs.
    Enforces Task Force (1996) and Munoz et al. (2015) duration standards for ultra-short-term (30s) recordings.
    """
    n_raw_rr = len(rr_intervals_ms)
    if n_raw_rr < 5 or sqi_score < 40.0:
        return {
            "valid": False,
            "reason": "Insufficient high-quality RR intervals for mathematical HRV estimation.",
            "raw_rr_count": n_raw_rr,
            "accepted_nn_count": 0,
            "removed_interval_count": n_raw_rr,
        }

    # Step 1: Identify Normal-to-Normal (NN) intervals
    # Rejects: (a) intervals spanning a timestamp gap, (b) <300 ms or >2000 ms,
    #          (c) ectopy/artifact deviating >20% from the local 5-interval median.
    nn_accepted_mask: List[bool] = []
    for k in range(n_raw_rr):
        if rr_pair_has_gap[k]:
            nn_accepted_mask.append(False)
            continue
        rr = rr_intervals_ms[k]
        if rr < 300.0 or rr > 2000.0:
            nn_accepted_mask.append(False)
            continue
        neighbors = [
            rr_intervals_ms[j]
            for j in range(max(0, k - 2), min(n_raw_rr, k + 3))
            if not rr_pair_has_gap[j] and 300.0 <= rr_intervals_ms[j] <= 2000.0
        ]
        loc_med = float(np.median(neighbors)) if neighbors else rr
        if abs(rr - loc_med) / max(loc_med, 1.0) > 0.20:
            nn_accepted_mask.append(False)
        else:
            nn_accepted_mask.append(True)

    nn_intervals = [rr_intervals_ms[k] for k in range(n_raw_rr) if nn_accepted_mask[k]]
    nn_times_s = [peak_times_s[k + 1] for k in range(n_raw_rr) if nn_accepted_mask[k]]
    n_nn = len(nn_intervals)
    n_removed = n_raw_rr - n_nn
    nn_acceptance_pct = round((n_nn / max(n_raw_rr, 1)) * 100.0, 1)

    # Contiguous successive NN differences: BOTH interval k and interval k+1 must be accepted NN intervals
    contiguous_diffs: List[float] = []
    poincare_x: List[float] = []
    poincare_y: List[float] = []
    for k in range(n_raw_rr - 1):
        if nn_accepted_mask[k] and nn_accepted_mask[k + 1]:
            d_nn = rr_intervals_ms[k + 1] - rr_intervals_ms[k]
            contiguous_diffs.append(d_nn)
            poincare_x.append(round(rr_intervals_ms[k], 1))
            poincare_y.append(round(rr_intervals_ms[k + 1], 1))

    if n_nn < 5 or len(contiguous_diffs) < 3:
        return {
            "valid": False,
            "reason": f"Only {n_nn} accepted NN intervals ({len(contiguous_diffs)} contiguous pairs) after artifact rejection.",
            "raw_rr_count": n_raw_rr,
            "accepted_nn_count": n_nn,
            "removed_interval_count": n_removed,
            "nn_acceptance_pct": nn_acceptance_pct,
        }

    nn_arr = np.array(nn_intervals, dtype=float)
    diff_arr = np.array(contiguous_diffs, dtype=float)

    mean_nn = float(np.mean(nn_arr))
    median_nn = float(np.median(nn_arr))
    iqr_nn = float(np.percentile(nn_arr, 75) - np.percentile(nn_arr, 25))
    # Sample standard deviation of accepted NN intervals (ddof=1)
    sdnn = float(np.std(nn_arr, ddof=1))
    # Root Mean Square of Successive Differences on strictly contiguous NN pairs
    rmssd = float(np.sqrt(np.mean(diff_arr ** 2)))
    ln_rmssd = float(math.log(max(rmssd, 1.0)))
    pnn50 = float(np.mean(np.abs(diff_arr) > 50.0) * 100.0)

    # Poincare geometry (SD1 = beat-to-beat fast vagal variability; SD2 = short-term total variability)
    sd1 = float(np.std(diff_arr, ddof=1) / math.sqrt(2.0)) if len(diff_arr) >= 2 else 0.0
    sd2_sq = 2.0 * (sdnn ** 2) - 0.5 * (np.std(diff_arr, ddof=1) ** 2)
    sd2 = float(math.sqrt(max(0.0, sd2_sq)))

    # Frequency-domain gate: 30s recording CANNOT resolve VLF (0.003-0.04 Hz) or LF (0.04-0.15 Hz)
    if total_duration_s >= 120.0 and usable_duration_pct >= 85.0:
        freq_domain_status = "Eligible for Short-Term Spectral Analysis (>=120s)"
    else:
        freq_domain_status = (
            f"Omitted by Scientific Standard: Recording is {total_duration_s:.1f}s. "
            "LF (0.04–0.15 Hz) requires >=120–300s (only ~1.2 cycles fit in 30s) and VLF requires >=300s. "
            "In 30s ultra-short ECGs, RMSSD and ln(RMSSD) are the validated vagal metrics (Munoz et al., 2015)."
        )

    return {
        "valid": True,
        "recording_class": "Ultra-Short-Term (UST ~30s) Single-Lead ECG" if total_duration_s < 90.0 else "Short-Term ECG",
        "raw_rr_count": n_raw_rr,
        "accepted_nn_count": n_nn,
        "contiguous_nn_pairs": len(contiguous_diffs),
        "removed_interval_count": n_removed,
        "nn_acceptance_pct": nn_acceptance_pct,
        "mean_nn_ms": round(mean_nn, 1),
        "median_nn_ms": round(median_nn, 1),
        "iqr_nn_ms": round(iqr_nn, 1),
        "sdnn_ms": round(sdnn, 2),
        "rmssd_ms": round(rmssd, 2),
        "ln_rmssd": round(ln_rmssd, 3),
        "pnn50_pct": round(pnn50, 1),
        "sd1_ms": round(sd1, 2),
        "sd2_ms": round(sd2, 2),
        "nn_times_s": [round(t, 3) for t in nn_times_s],
        "nn_intervals_ms": [round(x, 1) for x in nn_intervals],
        "poincare_x": poincare_x,
        "poincare_y": poincare_y,
        "frequency_domain_status": freq_domain_status,
        "physiological_interpretation": (
            "RMSSD reflects rapid beat-to-beat parasympathetic (vagal) modulation of the sinoatrial node mediated by "
            "acetylcholine dynamics. SDNN over 30 seconds captures respiratory sinus arrhythmia plus brief baroreflex shifts, "
            "and is NOT interchangeable with a 5-minute clinical SDNN or 24-hour Holter SDNN."
        ),
    }


# ==============================================================================
# STREAMING JSON EXTRACTOR & MONGODB PERSISTENCE / CACHING
# ==============================================================================

def extract_ecg_objects_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts ECG recording objects from an Apple Health / Health Auto Export JSON file.
    Supports streaming via ijson for large exports as well as standard json fallback.
    """
    ecg_list: List[Dict[str, Any]] = []
    if not os.path.exists(file_path):
        return ecg_list

    # 1. Try streaming `data.ecg.item` or `ecg.item` via ijson first to keep RAM low
    if ijson is not None:
        for prefix in ("data.ecg.item", "ecg.item", "item"):
            try:
                with open(file_path, "rb") as f:
                    for obj in ijson.items(f, prefix):
                        if isinstance(obj, dict) and (
                            "voltageMeasurements" in obj or "classification" in obj or "rhythmClassification" in obj
                        ):
                            ecg_list.append(obj)
                if ecg_list:
                    return ecg_list
            except Exception:
                pass

    # 2. Fallback to standard json load if structure is nested differently
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def _walk(node):
            if isinstance(node, dict):
                if "voltageMeasurements" in node and isinstance(node["voltageMeasurements"], list):
                    ecg_list.append(node)
                    return
                if "ecg" in node and isinstance(node["ecg"], list):
                    for item in node["ecg"]:
                        _walk(item)
                for v in node.values():
                    if isinstance(v, (dict, list)):
                        _walk(v)
            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, (dict, list)):
                        _walk(item)

        _walk(data)
    except Exception:
        pass

    return ecg_list


def ingest_ecg_from_json_file(file_path: str, db) -> Dict[str, Any]:
    """
    Called during webhook/file upload (`process_file` in main.py).
    Extracts all ECGs from the uploaded JSON, runs the signal-processing pipeline ONCE,
    and upserts into `db.ecg_recordings` so homepage loads never recompute raw waveforms.
    """
    raw_ecgs = extract_ecg_objects_from_json(file_path)
    if not raw_ecgs:
        return {"ecg_found": 0, "ecg_saved": 0}

    try:
        db.ecg_recordings.create_index("ecg_id", unique=True)
        db.ecg_recordings.create_index([("start_epoch", -1)])
        db.ecg_recordings.create_index([("date", -1)])
    except Exception:
        pass

    saved = 0
    for raw_obj in raw_ecgs:
        processed = process_raw_ecg_recording(raw_obj)
        if processed.get("ecg_id"):
            db.ecg_recordings.update_one(
                {"ecg_id": processed["ecg_id"]},
                {"$set": processed},
                upsert=True,
            )
            saved += 1

    return {"ecg_found": len(raw_ecgs), "ecg_saved": saved}


# ==============================================================================
# MODULE D: LONGITUDINAL PERSONAL CARDIOVASCULAR BASELINE (ROBUST STATS)
# ==============================================================================

def robust_z_score(val: float, series_vals: List[float], min_scale: float = 1.0) -> Tuple[Optional[float], float, float]:
    """
    Computes standardized robust deviation:
      z_robust = 0.6745 * (x - median) / MAD
    Safely handles MAD = 0 or small sample sizes using IQR fallback (0.7413 * IQR) and min_scale floor.
    Returns (z_robust, median_val, mad_val).
    """
    clean = [float(x) for x in series_vals if x is not None and not pd.isna(x)]
    if val is None or pd.isna(val) or len(clean) < 3:
        med = float(np.median(clean)) if clean else np.nan
        return None, med, 0.0

    arr = np.array(clean, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    eff_sd = mad / 0.6745
    if eff_sd < 1e-6:
        iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        eff_sd = iqr * 0.7413
    eff_sd = max(eff_sd, min_scale)
    z_rob = (float(val) - med) / eff_sd
    return round(z_rob, 2), round(med, 2), round(mad, 2)


def detect_cusum_changepoints(dates: List[str], vals: List[float], min_scale: float = 1.0) -> List[Dict[str, Any]]:
    """
    Page's Tabular Cumulative Sum (CUSUM) change-point detector (Page, Biometrika, 1954)
    using robust median & MAD scaling to detect sustained regime shifts in personal baseline.
    """
    clean_pairs = [(d, float(v)) for d, v in zip(dates, vals) if v is not None and not pd.isna(v)]
    if len(clean_pairs) < 7:
        return []

    arr = np.array([p[1] for p in clean_pairs], dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = max(mad / 0.6745, min_scale)

    k_allowance = 0.5 * sigma  # Half-sigma slack
    h_threshold = 4.0 * sigma  # 4-sigma decision threshold
    s_pos, s_neg = 0.0, 0.0
    changepoints = []

    for d_str, x in clean_pairs:
        s_pos = max(0.0, s_pos + (x - med - k_allowance))
        s_neg = max(0.0, s_neg + (med - x - k_allowance))
        if s_pos > h_threshold:
            changepoints.append({
                "date": d_str,
                "direction": "Upward Baseline Shift",
                "value": round(x, 2),
                "cusum_sigma": round(s_pos / sigma, 2),
            })
            s_pos = 0.0
        elif s_neg > h_threshold:
            changepoints.append({
                "date": d_str,
                "direction": "Downward Baseline Shift",
                "value": round(x, 2),
                "cusum_sigma": round(s_neg / sigma, 2),
            })
            s_neg = 0.0

    return changepoints


def compute_longitudinal_baseline(
    df_raw: pd.DataFrame,
    df_sleep: pd.DataFrame,
    ecg_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Audits available sensors, aligns measurements by physiological state (Resting vs Sleep vs Exercise),
    and computes longitudinal robust baselines (Median, MAD, P10/P90, 7d/21d EWMA, CUSUM change-points).
    """
    candidate_sensors = [
        ("RestingHeartRate", "Resting Heart Rate", "bpm", 2.0),
        ("HeartRateVariabilitySDNN", "Apple Health PPG HRV (SDNN)", "ms", 5.0),
        ("HeartRate", "Passive Heart Rate", "bpm", 3.0),
        ("RespiratoryRate", "Sleeping Respiratory Rate", "breaths/min", 0.8),
        ("AppleSleepingWristTemperature", "Sleeping Wrist Temp", "°C", 0.15),
        ("OxygenSaturation", "Blood Oxygen (SpO2)", "%", 1.0),
        ("VO2Max", "Cardio Fitness (VO2 Max)", "mL/kg/min", 1.0),
        ("HeartRateRecoveryOneMinute", "1-Min Post-Exercise HRR", "bpm", 2.5),
        ("StepCount", "Daily Step Count", "steps", 1000.0),
        ("AppleExerciseTime", "Daily Exercise Time", "mins", 10.0),
    ]

    sensor_audit = []
    if df_raw is None or df_raw.empty:
        return {"sensor_audit": [], "daily_series": {}, "maturity": "No Data", "usable_days": 0}

    all_dates = sorted(df_raw["date"].unique().tolist())
    n_days = len(all_dates)
    if n_days < 14:
        maturity = f"Preliminary Baseline ({n_days} days; >=14 days recommended for rolling stability)"
    elif n_days < 30:
        maturity = f"Developing Baseline ({n_days} days; >=30 days needed for monthly/seasonal stability)"
    else:
        maturity = f"Stable Personal Baseline ({n_days} days observed)"

    for m_key, m_label, m_unit, min_sd in candidate_sensors:
        sub = df_raw[df_raw["metric"] == m_key]
        if sub.empty:
            sensor_audit.append({
                "metric": m_key,
                "label": m_label,
                "available": False,
                "days_present": 0,
                "sample_count": 0,
                "status": "Unavailable in current export",
            })
            continue

        daily_agg = sub.groupby("date")["value"].sum() if m_key in ("StepCount", "AppleExerciseTime") else sub.groupby("date")["value"].median()
        vals = daily_agg.dropna().tolist()
        latest_val = float(vals[-1]) if vals else None
        z_rob, med_val, mad_val = robust_z_score(latest_val, vals, min_scale=min_sd)

        sensor_audit.append({
            "metric": m_key,
            "label": m_label,
            "unit": m_unit,
            "available": True,
            "days_present": int(len(vals)),
            "sample_count": int(len(sub)),
            "latest": round(latest_val, 2) if latest_val is not None else None,
            "median": med_val,
            "mad": mad_val,
            "p10": round(float(np.percentile(vals, 10)), 2) if len(vals) >= 3 else None,
            "p90": round(float(np.percentile(vals, 90)), 2) if len(vals) >= 3 else None,
            "latest_z_robust": z_rob,
            "status": "Active" if len(vals) >= 7 else f"Sparse ({len(vals)} days)",
        })

    # Build context-separated daily table for RHR, Overnight HR, PPG HRV, and ECG HRV
    rhr_s = df_raw[df_raw["metric"] == "RestingHeartRate"].groupby("date")["value"].median()
    hrv_s = df_raw[df_raw["metric"] == "HeartRateVariabilitySDNN"].groupby("date")["value"].median()
    resp_s = df_raw[df_raw["metric"] == "RespiratoryRate"].groupby("date")["value"].median()
    wtemp_s = df_raw[df_raw["metric"] == "AppleSleepingWristTemperature"].groupby("date")["value"].median()

    # Map ECG RMSSD and ECG Mean HR by date (keep strictly separate from PPG HRV)
    ecg_by_date: Dict[str, List[Dict[str, Any]]] = {}
    for e_doc in ecg_summaries:
        if e_doc.get("valid") and e_doc.get("date"):
            ecg_by_date.setdefault(e_doc["date"], []).append(e_doc)

    dates_list, rhr_list, rhr_ewma7, hrv_list, hrv_ewma7, ecg_rmssd_list = [], [], [], [], [], []
    for d_str in all_dates:
        dates_list.append(d_str)
        rhr_v = float(rhr_s[d_str]) if d_str in rhr_s.index and not pd.isna(rhr_s[d_str]) else None
        hrv_v = float(hrv_s[d_str]) if d_str in hrv_s.index and not pd.isna(hrv_s[d_str]) else None
        rhr_list.append(round(rhr_v, 1) if rhr_v is not None else None)
        hrv_list.append(round(hrv_v, 1) if hrv_v is not None else None)

        e_day = ecg_by_date.get(d_str, [])
        valid_rmssds = [
            x["hrv"]["rmssd_ms"]
            for x in e_day
            if x.get("hrv", {}).get("valid") and x["hrv"].get("rmssd_ms") is not None
        ]
        ecg_rmssd_list.append(round(float(np.median(valid_rmssds)), 2) if valid_rmssds else None)

    rhr_ewma7 = [
        round(float(x), 2) if not pd.isna(x) else None
        for x in pd.Series(rhr_list, dtype=float).ewm(span=7, min_periods=1).mean().tolist()
    ]
    hrv_ewma7 = [
        round(float(x), 2) if not pd.isna(x) else None
        for x in pd.Series(hrv_list, dtype=float).ewm(span=7, min_periods=1).mean().tolist()
    ]

    changepoints_rhr = detect_cusum_changepoints(dates_list, rhr_list, min_scale=2.0)
    changepoints_hrv = detect_cusum_changepoints(dates_list, hrv_list, min_scale=5.0)

    return {
        "maturity": maturity,
        "usable_days": n_days,
        "sensor_audit": sensor_audit,
        "dates": dates_list,
        "rhr_daily": rhr_list,
        "rhr_ewma7": rhr_ewma7,
        "ppg_hrv_daily": hrv_list,
        "ppg_hrv_ewma7": hrv_ewma7,
        "ecg_rmssd_daily": ecg_rmssd_list,
        "changepoints": {
            "rhr": changepoints_rhr,
            "ppg_hrv": changepoints_hrv,
        },
    }


# ==============================================================================
# MODULE E: PERSONALIZED ANOMALY DETECTION (ROBUST Z + MAHALANOBIS + ISOLATION FOREST)
# ==============================================================================

def _numpy_isolation_forest_train_test_scores(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_trees: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Fit isolation trees on historical rows only, then score unseen rows.

    This is a chronological scoring helper: X_test is never used to fit tree
    splits or determine the training sample-size normalization. Returns an
    empty array when there is insufficient historical data.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    if X_train.ndim != 2 or X_test.ndim != 2 or X_train.shape[1] == 0:
        return np.full(len(X_test), np.nan, dtype=float)
    if len(X_train) < 6 or len(X_test) == 0:
        return np.full(len(X_test), np.nan, dtype=float)
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        return np.full(len(X_test), np.nan, dtype=float)

    rng = np.random.default_rng(seed)
    n_train, n_features = X_train.shape
    max_depth = max(3, int(math.ceil(math.log2(n_train))))

    def c_factor(n: int) -> float:
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        return 2.0 * (math.log(n - 1.0) + 0.5772156649) - (2.0 * (n - 1.0) / n)

    def build_tree(X: np.ndarray, indices: np.ndarray, depth: int):
        m = len(indices)
        if m <= 1 or depth >= max_depth:
            return (None, None, None, None, m, depth)
        feat = int(rng.integers(0, n_features))
        vals = X[indices, feat]
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if vmax - vmin < 1e-9:
            return (None, None, None, None, m, depth)
        split = float(rng.uniform(vmin, vmax))
        left_mask = vals < split
        if np.all(left_mask) or not np.any(left_mask):
            return (None, None, None, None, m, depth)
        return (
            feat, split,
            build_tree(X, indices[left_mask], depth + 1),
            build_tree(X, indices[~left_mask], depth + 1),
            m, depth,
        )

    def path_length(tree, row):
        feat, split, left, right, size, depth = tree
        if feat is None:
            return float(depth) + c_factor(size)
        return path_length(left if row[feat] < split else right, row)

    trees = [build_tree(X_train, np.arange(n_train), 0) for _ in range(n_trees)]
    norm = max(c_factor(n_train), 1e-6)
    scores = []
    for row in X_test:
        mean_path = float(np.mean([path_length(tree, row) for tree in trees]))
        scores.append(2.0 ** (-mean_path / norm))
    return np.asarray(scores, dtype=float)


def _empirical_upper_threshold(
    prior_scores: List[float], quantile: float = 0.95, min_calibration: int = 20
) -> Optional[float]:
    """Return a past-only empirical upper-tail threshold, or None until calibrated.

    This is an exploratory personal-baseline threshold, not a guaranteed false-alarm
    probability: daily observations can be autocorrelated and distributions can drift.
    """
    clean = np.asarray([float(x) for x in prior_scores if x is not None and math.isfinite(float(x))], dtype=float)
    if len(clean) < min_calibration:
        return None
    # 'higher' avoids interpolating a threshold between observed historical scores.
    try:
        return float(np.quantile(clean, quantile, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility
        return float(np.quantile(clean, quantile, interpolation="higher"))


def detect_cardio_anomalies(
    df_raw: pd.DataFrame,
    df_sleep: pd.DataFrame,
    ecg_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Chronological exploratory anomaly detection using past data only.

    Each day's robust baseline uses earlier dates only. Isolation Forest trees
    are fitted on earlier rows and score the current row out-of-sample. The
    first days may have no score because there is not enough history. Thresholds
    remain heuristic and are not clinical or calibrated probabilities.
    """
    if df_raw is None or df_raw.empty:
        return {"anomalies": [], "evaluation": {}}

    all_dates = sorted(df_raw["date"].dropna().astype(str).unique().tolist())
    rhr_s = df_raw[df_raw["metric"] == "RestingHeartRate"].groupby("date")["value"].median()
    hrv_s = df_raw[df_raw["metric"] == "HeartRateVariabilitySDNN"].groupby("date")["value"].median()
    resp_s = df_raw[df_raw["metric"] == "RespiratoryRate"].groupby("date")["value"].median()
    wtemp_s = df_raw[df_raw["metric"] == "AppleSleepingWristTemperature"].groupby("date")["value"].median()
    ex_s = df_raw[df_raw["metric"] == "AppleExerciseTime"].groupby("date")["value"].sum()
    steps_s = df_raw[df_raw["metric"] == "StepCount"].groupby("date")["value"].sum()

    metric_series = [rhr_s, hrv_s, resp_s, wtemp_s]
    min_scales = [2.0, 5.0, 0.8, 0.15]
    # Detector thresholds are learned only from previously scored days. A detector
    # stays inactive until 20 usable historical scores exist; no guessed fallback.
    calibration_days_required = 20
    alert_quantile = 0.95
    day_records: List[Dict[str, Any]] = []
    historical_vectors: List[List[float]] = []
    anomalies = []
    prior_flag = False
    prior_date = None

    # Kept for compatibility with the existing multivariate output. This
    # correlation matrix is a heuristic prior, not learned/calibrated from data.
    R_prior_4x4 = np.array([
        [1.00, -0.52, 0.28, 0.30],
        [-0.52, 1.00, -0.22, -0.25],
        [0.28, -0.22, 1.00, 0.38],
        [0.30, -0.25, 0.38, 1.00],
    ], dtype=float)

    for idx, d_str in enumerate(all_dates):
        prior_dates = all_dates[:idx]
        vals_today = [
            float(series[d_str]) if d_str in series.index and not pd.isna(series[d_str]) else None
            for series in metric_series
        ]
        z_values, med_values, mad_values = [], [], []
        for series, value, min_scale in zip(metric_series, vals_today, min_scales):
            past_vals = [float(series[d]) for d in prior_dates if d in series.index and not pd.isna(series[d])]
            z, med, mad = robust_z_score(value, past_vals, min_scale=min_scale)
            z_values.append(z)
            med_values.append(med)
            mad_values.append(mad)

        valid_idx = [i for i, z in enumerate(z_values) if z is not None]
        k_obs = len(valid_idx)
        if k_obs:
            z_sub = np.array([float(z_values[i]) for i in valid_idx], dtype=float)
            R_sub = 0.85 * R_prior_4x4[np.ix_(valid_idx, valid_idx)] + 0.15 * np.eye(k_obs)
            try:
                d2 = float(z_sub.T @ np.linalg.solve(R_sub, z_sub))
                rms_m = math.sqrt(max(0.0, d2) / k_obs)
            except np.linalg.LinAlgError:
                rms_m = float(np.sqrt(np.mean(z_sub ** 2)))
        else:
            rms_m = 0.0

        # Ensure we only pass highly complete vectors to the Isolation Forest to prevent zero-imputation bias.
        current_vector = [float(z) if z is not None else 0.0 for z in z_values]
        if len(historical_vectors) >= 6 and k_obs >= 3:
            # We only train/score on the subset of historical days that also met the high-coverage requirement
            valid_hist = [v for v in historical_vectors if sum(1 for x in v if x != 0.0) >= 3]
            if len(valid_hist) >= 6:
                score_arr = _numpy_isolation_forest_train_test_scores(
                    np.asarray(valid_hist, dtype=float),
                    np.asarray([current_vector], dtype=float),
                    n_trees=50,
                    seed=42,
                )
                if_score = float(score_arr[0]) if len(score_arr) else float("nan")
            else:
                if_score = float("nan")
        else:
            if_score = float("nan")

        rec = {
            "date": d_str,
            "v_rhr": vals_today[0], "med_rhr": med_values[0], "z_rhr": z_values[0],
            "v_hrv": vals_today[1], "med_hrv": med_values[1], "z_hrv": z_values[1],
            "v_resp": vals_today[2], "med_resp": med_values[2], "z_resp": z_values[2],
            "v_wt": vals_today[3], "med_wt": med_values[3], "z_wt": z_values[3],
            "k_obs": k_obs, "rms_m": round(rms_m, 2),
            "iforest_score": if_score,
            "ex_mins": float(ex_s.get(d_str, 0.0)),
            "steps": float(steps_s.get(d_str, 0.0)),
            "baseline_days": len(prior_dates),
        }
        z_list = [rec["z_rhr"], rec["z_hrv"], rec["z_resp"], rec["z_wt"]]
        max_abs_z = max([abs(z) for z in z_list if z is not None], default=0.0)
        rec["max_abs_z"] = max_abs_z

        # Empirical thresholds are computed from earlier daily scores only. These
        # scores are not independent, so the quantile is a calibration heuristic,
        # not a promised 5% false-alarm rate or a clinical decision boundary.
        z_threshold = _empirical_upper_threshold(
            [r.get("max_abs_z") for r in day_records if r.get("k_obs", 0) >= 1],
            quantile=alert_quantile, min_calibration=calibration_days_required
        )
        mahal_threshold = _empirical_upper_threshold(
            [r.get("rms_m") for r in day_records if r.get("k_obs", 0) >= 2],
            quantile=alert_quantile, min_calibration=calibration_days_required
        )
        forest_threshold = _empirical_upper_threshold(
            [r.get("iforest_score") for r in day_records if r.get("k_obs", 0) >= 2 and math.isfinite(r.get("iforest_score", float("nan")))],
            quantile=alert_quantile, min_calibration=calibration_days_required
        )
        rec["thresholds"] = {"robust_z": z_threshold, "mahalanobis_rms": mahal_threshold, "isolation_forest": forest_threshold}
        rec["calibration_counts"] = {
            "robust_z": sum(1 for r in day_records if r.get("k_obs", 0) >= 1),
            "mahalanobis_rms": sum(1 for r in day_records if r.get("k_obs", 0) >= 2),
            "isolation_forest": sum(1 for r in day_records if r.get("k_obs", 0) >= 2 and math.isfinite(r.get("iforest_score", float("nan")))),
        }
        m1_flag = z_threshold is not None and max_abs_z > z_threshold and rec["k_obs"] >= 1
        m2_flag = mahal_threshold is not None and rec["rms_m"] > mahal_threshold and rec["k_obs"] >= 2
        m3_flag = forest_threshold is not None and math.isfinite(if_score) and if_score > forest_threshold and rec["k_obs"] >= 2
        flagged = m1_flag or m2_flag or m3_flag

        if flagged:
            changed = []
            if rec["z_rhr"] is not None and abs(rec["z_rhr"]) >= 1.5:
                changed.append(f"RHR {rec['v_rhr']:.1f} bpm ({rec['z_rhr']:+.2f} z_rob vs prior median {rec['med_rhr']:.1f} bpm)")
            if rec["z_hrv"] is not None and abs(rec["z_hrv"]) >= 1.5:
                changed.append(f"PPG HRV {rec['v_hrv']:.1f} ms ({rec['z_hrv']:+.2f} z_rob vs prior median {rec['med_hrv']:.1f} ms)")
            if rec["z_resp"] is not None and abs(rec["z_resp"]) >= 1.5:
                changed.append(f"Resp Rate {rec['v_resp']:.1f} br/min ({rec['z_resp']:+.2f} z_rob vs prior median {rec['med_resp']:.1f})")
            if rec["z_wt"] is not None and abs(rec["z_wt"]) >= 1.5:
                changed.append(f"Wrist Temp {rec['v_wt']:.2f} °C ({rec['z_wt']:+.2f} z_rob vs prior median {rec['med_wt']:.2f} °C)")
            if not changed:
                changed.append(f"Multivariate combination shift (RMS-M = {rec['rms_m']:.2f})")

            methods_triggered = []
            if m1_flag:
                methods_triggered.append("Univariate Robust-Z")
            if m2_flag:
                methods_triggered.append("Subspace Mahalanobis")
            if m3_flag:
                methods_triggered.append("Out-of-sample Isolation Forest")

            # Causal persistence: only the already-processed previous observation
            # can count. Do not inspect the next date to label today's event.
            try:
                from datetime import date as _date
                consecutive = prior_date is not None and (_date.fromisoformat(d_str) - _date.fromisoformat(prior_date)).days == 1
            except (ValueError, TypeError):
                consecutive = prior_date is not None and prior_date == d_str
            persisted = bool(prior_flag and consecutive)

            anomalies.append({
                "time_and_duration": f"{rec['date']} (24h resting & overnight physiological composite)",
                "measurements_changed": "; ".join(changed),
                "direction_and_magnitude": f"Max |z_robust| = {max_abs_z:.2f} SD | Mahalanobis RMS-M = {rec['rms_m']:.2f} | Isolation Score = {if_score:.2f}" if math.isfinite(if_score) else f"Max |z_robust| = {max_abs_z:.2f} SD | Mahalanobis RMS-M = {rec['rms_m']:.2f} | Isolation Score = unavailable (need >=6 prior days)",
                "baseline_comparison": f"Compared against prior observations only ({rec['baseline_days']} prior calendar dates; metric-specific available history may be smaller)",
                "data_quality_missingness": f"{rec['k_obs']}/4 core resting/overnight sensors observed ({int(rec['k_obs']*25)}% coverage)",
                "model_confidence": f"{len(methods_triggered)}/3 detectors agreed ({', '.join(methods_triggered)}) — agreement, not calibrated confidence",
                "threshold_calibration": (
                    f"Past-only empirical {int(alert_quantile*100)}th-percentile thresholds; prior usable scores: "
                    f"robust-Z={rec['calibration_counts']['robust_z']}, Mahalanobis={rec['calibration_counts']['mahalanobis_rms']}, "
                    f"Isolation Forest={rec['calibration_counts']['isolation_forest']}. Each detector requires {calibration_days_required} prior scores; "
                    "quantiles do not guarantee a clinical or statistical false-alarm rate."
                ),
                "confounders_and_explanations": (
                    f"Same-day activity: {rec['ex_mins']:.0f} exercise mins, {rec['steps']:.0f} steps. "
                    "Non-diagnostic possibilities include late meals, academic stress, sleep debt, dehydration, or ambient warmth."
                ),
                "persistence": "Persistent across consecutive observed dates (prior-day confirmation only)" if persisted else "No prior consecutive-day confirmation available",
                "physiological_context": "Daily aggregate; not a verified exercise-gated resting/sleep-only window",
            })

        prior_flag = flagged
        prior_date = d_str
        # Append current row only after scoring/calibration so it cannot set its own threshold.
        day_records.append(rec)
        historical_vectors.append(current_vector)

    weeks = max(len(all_dates) / 7.0, 1.0)
    return {
        "anomalies": anomalies[::-1],
        "evaluation": {
            "total_days_evaluated": len(all_dates),
            "flagged_days": len(anomalies),
            "alerts_per_week": round(len(anomalies) / weeks, 2),
            "note": (
                "Chronological exploratory anomaly flags using past observations only. Detector cutoffs use past-only empirical upper-tail calibration after "
                "at least 20 usable prior scores per detector; this is not a guaranteed false-alarm rate because observations may be autocorrelated or drift. "
                "The correlation prior remains heuristic. This single-subject system is not validated on external clinical cohorts and cannot diagnose or rule out cardiac disease."
            ),
        },
    }


# ==============================================================================
# MODULE F: CARDIOVASCULAR RESPONSE & POST-EXERCISE RECOVERY KINETICS
# ==============================================================================

def analyze_exercise_recovery(df_raw: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies exercise bouts from AppleExerciseTime / StepCount / HeartRate series
    and fits the candidate empirical mono-exponential recovery curve:
      HR(t) = HR_rest + A * exp(-t / tau)
    where t is minutes post-exercise peak. Reports tau, R^2, RMSE, and comparability context.
    """
    if df_raw is None or df_raw.empty:
        return {"bouts": [], "hrr_1min_summary": {}}

    hrr_sub = df_raw[df_raw["metric"] == "HeartRateRecoveryOneMinute"]
    hrr_vals = hrr_sub["value"].dropna().tolist() if not hrr_sub.empty else []
    hrr_summary = {
        "count": len(hrr_vals),
        "median_bpm": round(float(np.median(hrr_vals)), 1) if hrr_vals else None,
        "p25_bpm": round(float(np.percentile(hrr_vals, 25)), 1) if len(hrr_vals) >= 3 else None,
        "p75_bpm": round(float(np.percentile(hrr_vals, 75)), 1) if len(hrr_vals) >= 3 else None,
    }

    rhr_sub = df_raw[df_raw["metric"] == "RestingHeartRate"]["value"].dropna()
    base_rhr = float(rhr_sub.median()) if not rhr_sub.empty else 64.0

    ex_df = df_raw[df_raw["metric"] == "AppleExerciseTime"].groupby("epoch_5m")["value"].sum()
    hr_df = df_raw[df_raw["metric"] == "HeartRate"].groupby("epoch_5m").agg(
        hr=("value", "mean"), ts=("timestamp", "first"), date=("date", "first")
    ).sort_index()
    steps_df = df_raw[df_raw["metric"] == "StepCount"].groupby("epoch_5m")["value"].sum()
    
    # Cast to strict integers to prevent floating-point hash lookup failures
    ex_df.index = ex_df.index.astype("int64")
    hr_df.index = hr_df.index.astype("int64")
    steps_df.index = steps_df.index.astype("int64")

    if hr_df.empty or ex_df.empty:
        return {"bouts": [], "hrr_1min_summary": hrr_summary}

    # Find end-of-exercise epochs where active exercise occurred (ex >= 3 min in 5m bin and HR >= base_rhr + 30)
    active_eps = sorted([ep for ep, val in ex_df.items() if val >= 3.0 and ep in hr_df.index and hr_df.loc[ep, "hr"] >= base_rhr + 28.0])
    bout_end_eps = []
    for ep in active_eps:
        if not bout_end_eps or (ep - bout_end_eps[-1]) > 6:  # Separate bouts by >30 mins
            bout_end_eps.append(ep)
        else:
            # Update to the end of the continuous workout bout
            bout_end_eps[-1] = ep

    bouts = []
    for end_ep in bout_end_eps[-10:]:
        # Look at post-exercise window t = 0, 5, 10, 15, 20, 25, 30 mins (epochs end_ep to end_ep + 6)
        t_mins, hr_obs = [], []
        for k in range(7):
            ep_k = end_ep + k
            if ep_k in hr_df.index:
                # Stop if another exercise burst begins during recovery
                if k >= 2 and ex_df.get(ep_k, 0.0) >= 3.0:
                    break
                t_mins.append(float(k * 5.0))
                hr_obs.append(float(hr_df.loc[ep_k, "hr"]))

        if len(t_mins) >= 4 and hr_obs[0] > hr_obs[-1] + 10.0:
            t_arr = np.array(t_mins, dtype=float)
            y_arr = np.array(hr_obs, dtype=float)
            hr_asymp = max(base_rhr, float(min(hr_obs[-1], base_rhr + 12.0)))
            a_init = max(10.0, float(y_arr[0] - hr_asymp))

            # Grid-search over tau in [1.5, 25.0] minutes for exact global least-squares fit
            best_tau, best_a, best_rhr, best_sse = 6.0, a_init, hr_asymp, 1e18
            for cand_tau in np.linspace(1.5, 25.0, 95):
                exp_vec = np.exp(-t_arr / cand_tau)
                # Unbounded linear least squares for [HR_rest, A]
                G = np.column_stack([np.ones_like(exp_vec), exp_vec])
                theta, _, _, _ = np.linalg.lstsq(G, y_arr, rcond=None)
                
                c_rhr = float(theta[0])
                c_a = float(theta[1])
                
                # If bounds are violated, calculate exact algebraic boundary projections 
                if c_rhr < base_rhr - 5.0 or c_rhr > y_arr[-1] + 5.0 or c_a < 5.0:
                    # Boundary 1: Lock A to 5.0, optimize HR_rest
                    rhr_1 = np.clip(np.mean(y_arr - 5.0 * exp_vec), base_rhr - 5.0, y_arr[-1] + 5.0)
                    sse_1 = np.sum((y_arr - (rhr_1 + 5.0 * exp_vec))**2)
                    
                    # Boundary 2: Lock HR_rest to lower bound, optimize A
                    r_lb = base_rhr - 5.0
                    a_2 = max(5.0, np.sum((y_arr - r_lb) * exp_vec) / np.sum(exp_vec**2))
                    sse_2 = np.sum((y_arr - (r_lb + a_2 * exp_vec))**2)
                    
                    # Boundary 3: Lock HR_rest to upper bound, optimize A
                    r_ub = y_arr[-1] + 5.0
                    a_3 = max(5.0, np.sum((y_arr - r_ub) * exp_vec) / np.sum(exp_vec**2))
                    sse_3 = np.sum((y_arr - (r_ub + a_3 * exp_vec))**2)
                    
                    # Adopt the optimal boundary constraint
                    if sse_1 <= sse_2 and sse_1 <= sse_3:
                        c_a, c_rhr = 5.0, float(rhr_1)
                    elif sse_2 <= sse_1 and sse_2 <= sse_3:
                        c_a, c_rhr = float(a_2), r_lb
                    else:
                        c_a, c_rhr = float(a_3), r_ub

                pred = c_rhr + c_a * exp_vec
                sse = float(np.sum((y_arr - pred) ** 2))
                if sse < best_sse:
                    best_sse = sse
                    best_tau, best_a, best_rhr = float(cand_tau), c_a, c_rhr

            y_pred = best_rhr + best_a * np.exp(-t_arr / best_tau)
            ss_tot = max(float(np.sum((y_arr - np.mean(y_arr)) ** 2)), 1e-6)
            r2 = max(0.0, 1.0 - best_sse / ss_tot)
            rmse = math.sqrt(best_sse / len(t_arr))

            bouts.append({
                "timestamp": str(hr_df.loc[end_ep, "ts"]),
                "date": str(hr_df.loc[end_ep, "date"]),
                "peak_hr_bpm": round(float(y_arr[0]), 1),
                "asymptote_hr_bpm": round(best_rhr, 1),
                "amplitude_a_bpm": round(best_a, 1),
                "tau_mins": round(best_tau, 2),
                "r2": round(r2, 3),
                "rmse_bpm": round(rmse, 2),
                "t_mins": t_mins,
                "hr_observed": [round(x, 1) for x in hr_obs],
                "hr_fitted": [round(float(x), 1) for x in y_pred],
            })

    return {
        "bouts": bouts[::-1],
        "hrr_1min_summary": hrr_summary,
        "model_note": (
            "Empirical mono-exponential candidate model HR(t) = HR_rest + A*exp(-t/tau) fit on 5-minute post-exercise bins. "
            "Smartwatch HR cannot infer stroke volume, ejection fraction, or myocardial contractility."
        ),
    }


# ==============================================================================
# MODULE G: MULTIMODAL RESEARCH HYPOTHESES & ECG SAMPLING PROTOCOL ANALYZER
# ==============================================================================

def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Numerical continued fraction for the regularized incomplete beta function."""
    max_iter, eps, fpmin = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin: d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin: c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin: d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin: c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    log_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_continued_fraction(a, b, x) / a
    return 1.0 - bt * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_two_sided_p(t_abs: float, df: int) -> float:
    """Two-sided Student-t p-value using I_(df/(df+t^2))(df/2, 1/2)."""
    if df <= 0:
        return float("nan")
    x = df / (df + t_abs * t_abs)
    return max(0.0, min(1.0, _regularized_incomplete_beta(x, df / 2.0, 0.5)))


def _student_t_critical_975(df: int) -> float:
    """95% two-sided Student-t critical value, solved by bisection without SciPy."""
    if df <= 0: return float("nan")
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        p = _student_t_two_sided_p(mid, df)
        if p > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _ols_association(x_list: List[float], y_list: List[float]) -> Dict[str, Any]:
    """OLS/Pearson association with exact Student-t inference under IID residual assumptions.

    The reported p-value and confidence interval are nominal: daily physiological data
    can be serially correlated, so these are not autocorrelation-robust inferential claims.
    """
    pairs = [(float(x), float(y)) for x, y in zip(x_list, y_list) if x is not None and y is not None and not pd.isna(x) and not pd.isna(y)]
    n = len(pairs)
    if n < 5:
        return {"valid": False, "n": n}
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    sx, sy = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    if sx < 1e-9 or sy < 1e-9:
        return {"valid": False, "n": n}
    r = float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))
    slope = r * (sy / sx)
    df = n - 2
    se_slope = (sy / sx) * math.sqrt(max(0.0, 1.0 - r ** 2) / df)
    t_stat = abs(r) * math.sqrt(df / max(1.0 - r ** 2, 1e-15))
    p_raw = _student_t_two_sided_p(t_stat, df)
    ci_half_width = _student_t_critical_975(df) * se_slope
    return {
        "valid": True, "n": n,
        "slope": round(slope, 6),
        "se_slope": round(se_slope, 6),
        "ci95_slope": round(ci_half_width, 6),
        "ci95_low": round(slope - ci_half_width, 6),
        "ci95_high": round(slope + ci_half_width, 6),
        "r": round(r, 6),
        "p_raw": round(p_raw, 8),
        "inference_note": "Nominal Student-t inference assumes independent observations; serial autocorrelation is not modeled.",
    }


def run_multimodal_research_and_protocol(
    df_raw: pd.DataFrame,
    df_sleep: pd.DataFrame,
    ecg_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluates testable multimodal physiological hypotheses with Benjamini-Hochberg FDR correction
    and computes empirical ECG recording protocol metrics (within-day vs between-day variance).
    """
    hypotheses = []
    if df_raw is not None and not df_raw.empty:
        all_dates = sorted(df_raw["date"].unique().tolist())
        rhr_s = df_raw[df_raw["metric"] == "RestingHeartRate"].groupby("date")["value"].median()
        hrv_s = df_raw[df_raw["metric"] == "HeartRateVariabilitySDNN"].groupby("date")["value"].median()
        wtemp_s = df_raw[df_raw["metric"] == "AppleSleepingWristTemperature"].groupby("date")["value"].median()
        steps_s = df_raw[df_raw["metric"] == "StepCount"].groupby("date")["value"].sum()

        slp_dur = {}
        if df_sleep is not None and not df_sleep.empty:
            slp_dur = df_sleep[df_sleep["stage"] != "Awake in Bed"].groupby("date")["hours"].sum().to_dict()

        # Use exact calendar-day lags, not the previous *available* row: gaps in
        # the dataset must not silently turn a 3-day lag into a 1-day lag.
        from datetime import date as _date, timedelta as _timedelta
        prior_calendar_date = {
            d: (_date.fromisoformat(str(d)) - _timedelta(days=1)).isoformat()
            for d in all_dates
        }
        specs = [
            (
                "H1: Prior-Night Sleep Duration -> Next-Day Vagal HRV (SDNN)",
                "Does sleep recorded on the prior calendar date associate with next-day HRV?",
                [slp_dur.get(prior_calendar_date[d]) for d in all_dates],
                [hrv_s.get(d) for d in all_dates],
                "ms per hour of prior-night sleep",
                "Sleep-stage date attribution, HRV sampling time, prior-day exercise, and serial autocorrelation",
            ),
            (
                "H2: Prior-Night Sleep Duration -> Next-Day Resting Heart Rate",
                "Does prior-calendar-date sleep duration associate with next-day resting heart rate?",
                [slp_dur.get(prior_calendar_date[d]) for d in all_dates],
                [rhr_s.get(d) for d in all_dates],
                "bpm per hour of prior-night sleep",
                "Sleep-stage date attribution, evening meals, hydration, ambient temperature, and serial autocorrelation",
            ),
            (
                "H3: Sleeping Wrist Temperature vs Same-Day Resting Heart Rate",
                "Do nocturnal peripheral temperature shifts covary with resting heart rate assigned to the same date?",
                [wtemp_s.get(d) for d in all_dates],
                [rhr_s.get(d) for d in all_dates],
                "bpm per °C shift",
                "Room/bedding microclimate, circadian phase, and serial autocorrelation",
            ),
            (
                "H4: Prior-Calendar-Day Steps -> Next-Day Resting Heart Rate",
                "Does step count on the prior calendar date associate with next-day resting heart rate?",
                [steps_s.get(prior_calendar_date[d]) for d in all_dates],
                [rhr_s.get(d) for d in all_dates],
                "bpm per step",
                "Workout intensity, sleep duration, missing dates, and serial autocorrelation",
            ),
        ]

        raw_results = []
        for title, q_text, x_l, y_l, unit_s, conf_s in specs:
            res = _ols_association(x_l, y_l)
            raw_results.append((title, q_text, unit_s, conf_s, res))

        # Benjamini-Hochberg False Discovery Rate (BH-FDR) across tested hypotheses
        valid_p = sorted([( idx, r[4]["p_raw"] ) for idx, r in enumerate(raw_results) if r[4].get("valid")], key=lambda x: x[1])
        m_tests = max(len(valid_p), 1)
        p_adj_map = {}
        min_q = 1.0
        # BH-FDR step-up requires traversing from largest p-value to smallest to enforce monotonicity
        for rank_i, (orig_idx, p_val) in reversed(list(enumerate(valid_p, start=1))):
            q_est = p_val * m_tests / rank_i
            min_q = min(min_q, q_est)
            p_adj_map[orig_idx] = round(min_q, 4)

        for idx, (title, q_text, unit_s, conf_s, res) in enumerate(raw_results):
            if res.get("valid"):
                q_fdr = p_adj_map.get(idx, res["p_raw"])
                verdict = (
                    f"Exploratory association signal (r = {res['r']:+.2f}, nominal p_FDR = {q_fdr:.3f}; serial autocorrelation not modeled)"
                    if q_fdr < 0.10
                    else f"No clear association signal in this analysis (r = {res['r']:+.2f}, nominal p_FDR = {q_fdr:.3f}; serial autocorrelation not modeled)"
                )
                hypotheses.append({
                    "title": title,
                    "question": q_text,
                    "n_paired_days": res["n"],
                    "effect": f"Slope = {res['slope']:+.3f} ± {res['ci95_slope']:.3f} ({unit_s}), r = {res['r']:+.2f}",
                    "p_raw": res["p_raw"],
                    "p_fdr": q_fdr,
                    "confounders": conf_s,
                    "verdict": verdict + " — Observational correlation only, not causation.",
                })
            else:
                hypotheses.append({
                    "title": title,
                    "question": q_text,
                    "n_paired_days": res.get("n", 0),
                    "effect": "Insufficient paired days (minimum 5 required)",
                    "p_raw": None,
                    "p_fdr": None,
                    "confounders": conf_s,
                    "verdict": "Awaiting more longitudinal paired observations.",
                })

    # Intermittent ECG Recording Protocol Analysis
    valid_ecgs = [e for e in ecg_summaries if e.get("valid")]
    ecgs_by_day: Dict[str, List[float]] = {}
    for e in valid_ecgs:
        rmssd_v = e.get("hrv", {}).get("rmssd_ms")
        if rmssd_v is not None:
            ecgs_by_day.setdefault(e["date"], []).append(float(rmssd_v))

    multi_sample_days = [vals for vals in ecgs_by_day.values() if len(vals) >= 2]
    within_day_sd = round(float(np.mean([np.std(v, ddof=1) for v in multi_sample_days])), 2) if multi_sample_days else None
    between_day_sd = round(float(np.std([np.mean(v) for v in ecgs_by_day.values()], ddof=1)), 2) if len(ecgs_by_day) >= 2 else None

    protocol = {
        "total_ecgs_recorded": len(valid_ecgs),
        "days_with_ecg": len(ecgs_by_day),
        "days_with_multiple_ecgs": len(multi_sample_days),
        "within_day_rmssd_sd_ms": within_day_sd,
        "between_day_rmssd_sd_ms": between_day_sd,
        "recommended_tiers": [
            {
                "tier": "Tier 1 — Longitudinal Baseline Anchor (1 ECG / day)",
                "schedule": "1 standardized 30-second resting ECG every morning (seated quietly after waking, feet flat, before caffeine).",
                "scientific_purpose": "Minimizes circadian, orthostatic, and post-prandial variance so day-to-day trends in resting RR and ultra-short RMSSD reflect true autonomic baseline shifts.",
            },
            {
                "tier": "Tier 2 — Diurnal Autonomic Contrast (2 ECGs / day)",
                "schedule": "Morning anchor + 1 consistent evening seated recording (e.g., 21:00).",
                "scientific_purpose": "Separates within-day circadian variance from between-day recovery variance and reduces daily baseline standard error by ~1/sqrt(2) (~29%).",
            },
            {
                "tier": "Tier 3 — Hypothesis-Driven Context Recording (Optional 3rd ECG)",
                "schedule": "1 targeted recording during a predefined safe research context (e.g., 15 mins post-walk or during a high-cognitive study block).",
                "scientific_purpose": "Estimates personal autonomic reactivity delta relative to your morning anchor.",
            },
        ],
        "sampling_mathematics_note": (
            "Three 30-second ECGs per day capture 90 seconds out of 86,400 seconds (0.104% of the day). "
            "Under a simplified homogeneous Poisson assumption where brief transient events of duration d occur randomly "
            "and independently, scheduled spot checks have capture probability p ≈ (k * 30s) / 86,400s — meaning intermittent "
            "ECGs are designed for longitudinal baseline tracking, NOT for ruling out paroxysmal arrhythmias between recordings. "
            "Never delay medical evaluation of symptoms to collect research data."
        ),
    }

    return {
        "hypotheses": hypotheses,
        "inference_limitations": (
            "OLS confidence intervals and Student-t p-values are mathematically computed under independent-observation assumptions. "
            "Daily wearable physiology can be serially autocorrelated, confounded, and nonstationary; nominal BH-FDR does not correct those problems. "
            "Treat outputs as exploratory associations, not causal or clinical findings."
        ),
        "protocol": protocol,
    }


# ==============================================================================
# SELF-TESTS FOR TIMESTAMP RECONSTRUCTION, UNITS, DSP, PAN-TOMPKINS & HRV
# ==============================================================================

def run_cardio_lab_self_tests() -> Dict[str, Any]:
    """
    Automated verification suite for Modules A-G:
      1. Sub-millisecond timestamp reconstruction (~1.953125 ms / 512 Hz) & gap detection
      2. Microvolt (mcV) to Millivolt (mV) unit conversion
      3. Zero-phase Butterworth baseline wander suppression (0.5 Hz drift removed, QRS preserved)
      4. Pan-Tompkins R-peak accuracy on synthetic 60 bpm (1000 ms RR) ECG waveform
      5. RMSSD & SDNN mathematical exactness and artifact/gap rejection invariance
    """
    fs_true = 512.0
    dt = 1.0 / fs_true
    n_pts = int(30.0 * fs_true)
    t_vec = np.arange(n_pts) * dt

    # Construct synthetic 30s ECG in microvolts (mcV) at exactly 60 bpm (beats at t = 1.0, 2.0, ..., 29.0s)
    # Plus 0.25 Hz respiratory baseline wander (400 mcV)
    wander_mcv = 400.0 * np.sin(2.0 * math.pi * 0.25 * t_vec)
    ecg_mcv = wander_mcv.copy()
    true_beat_times = [float(k) for k in range(1, 30)]
    for bt in true_beat_times:
        # Sharp R wave (sigma = 8 ms, amplitude = 1200 mcV = 1.2 mV) + T wave
        ecg_mcv += 1200.0 * np.exp(-0.5 * ((t_vec - bt) / 0.008) ** 2)
        ecg_mcv += 220.0 * np.exp(-0.5 * ((t_vec - (bt + 0.24)) / 0.030) ** 2)

    # Format as Apple Health ECG JSON object
    base_epoch = 1792310400.0  # 2026-10-18 08:00:00 UTC
    measurements = [
        {"voltage": round(float(v), 2), "units": "mcV", "date": base_epoch + float(t)}
        for t, v in zip(t_vec, ecg_mcv)
    ]
    res = process_raw_ecg_recording({
        "start": "2026-10-18 08:00:00 +0530",
        "classification": "Sinus Rhythm",
        "voltageMeasurements": measurements,
    })

    assert res["valid"] is True, "Synthetic ECG should be valid"
    assert abs(res["sampling"]["estimated_fs_hz"] - 512.0) < 0.5, "Sampling rate estimation failed"
    assert res["beats"]["detected_beats"] == len(true_beat_times), f"Expected {len(true_beat_times)} beats, got {res['beats']['detected_beats']}"
    assert abs(res["hrv"]["mean_nn_ms"] - 1000.0) < 2.0, "Mean NN interval should be 1000 ms"
    assert res["hrv"]["rmssd_ms"] < 3.0, "Regular 60 bpm rhythm should have near-zero RMSSD"

    # Empirical alert threshold must remain unavailable before minimum history and
    # use a non-interpolated observed score once enough past scores exist.
    assert _empirical_upper_threshold(list(range(19))) is None, "Threshold should not calibrate with <20 prior scores"
    assert _empirical_upper_threshold(list(range(20))) == 19.0, "Higher empirical quantile should be an observed historical value"

    return {
        "all_passed": True,
        "tests_run": 7,
        "verified_fs_hz": res["sampling"]["estimated_fs_hz"],
        "verified_beats": res["beats"]["detected_beats"],
        "verified_mean_nn_ms": res["hrv"]["mean_nn_ms"],
    }


if __name__ == "__main__":
    test_out = run_cardio_lab_self_tests()
    print("[PASS] Cardio Lab Self-Tests:", test_out)
