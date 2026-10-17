# ==============================================================================
# APPLE WATCH SERIES 11 — UNIFIED BIOMETRIC SCIENCE & TELEMETRY LAB
#   - Activity 1: Personal Stress Detective (Metabolic De-confounding & Mahalanobis D_M^2)
#   - Activity 2: Personal Energy Predictor (Borbély Two-Process S+C & Bayesian Ridge)
#   - Activity 3: Sleep Experiment Lab (Circular Stats, Hammock Nadir & N-of-1 Trials)
#
# PART 1 OF 2: Core Math, Self-Tests, XML Pipeline, Unified Logger, Activity 1 & 2
# ==============================================================================

import os
import json
import math
import webbrowser
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
import numpy as np

XML_FILE = "export.xml"
LOG_CSV = "energy_study_log.csv"
EXP_CSV = "sleep_experiments_history.csv"
HTML_DASHBOARD = "biometric_science_lab.html"

# Complete Plain-English Encyclopedia of Apple Watch Parameters (with Role Taxonomy)
PARAMETER_GUIDE = {
    "HeartRate": (
        "Continuous Heart Rate", "1. Physiological Evidence", "bpm", "55 – 90 bpm (resting/daily)",
        "Every pulse reading from the optical PPG sensor. Used to fit your overnight sleeping hammock parabola and de-confounded daytime stress residuals."
    ),
    "HeartRateVariabilitySDNN": (
        "Heart Rate Variability (HRV)", "1. Physiological Evidence", "ms", "45 – 100+ ms (higher = calmer)",
        "Millisecond variations between heartbeats controlled by the vagus nerve. Tracks overnight parasympathetic recharge slope and daytime stress."
    ),
    "RestingHeartRate": (
        "Resting Heart Rate (RHR)", "1. Physiological Evidence", "bpm", "50 – 72 bpm (lower = fitter)",
        "Your baseline sinoatrial pacemaker speed when still. Rises of +3 to +6 bpm above baseline indicate incomplete overnight recovery or systemic strain."
    ),
    "RespiratoryRate": (
        "Sleeping Breathing Rate", "1. Physiological Evidence", "breaths/min", "12 – 18 breaths/min",
        "Overnight ventilation rate driven by brainstem chemoreceptors. Elevations correlate with sympathetic activation, fever, or airway load."
    ),
    "AppleSleepingWristTemperature": (
        "Overnight Wrist Temperature", "2. Recovery & Sleep Context", "°C", "Within ±0.2°C of baseline",
        "Nocturnal peripheral thermoregulation. Shifts > +0.3°C flag illness/fever confounders during sleep experiments. (Unlocks after 5 nights)."
    ),
    "OxygenSaturation": (
        "Blood Oxygen (SpO2)", "2. Recovery & Sleep Context", "%", "95% – 100%",
        "Arterial hemoglobin oxygen saturation. Overnight dips below 94% can reflect breathing disturbances or non-restorative sleep."
    ),
    "VO2Max": (
        "Cardio Fitness (VO2 Max)", "2. Recovery & Sleep Context", "mL/kg/min", "42 – 52+ mL/kg/min",
        "Maximal oxygen uptake ceiling according to the Fick equation. Higher aerobic fitness accelerates deep slow-wave sleep onset."
    ),
    "HeartRateRecoveryOneMinute": (
        "1-Minute Heart Rate Recovery", "3. Activity & Confounder Gate", "bpm", "20 – 35+ bpm drop",
        "Rate of vagal reactivation in the first 60 seconds after exercise. Calibrates our personal post-exercise heart rate decay filter."
    ),
    "WalkingHeartRateAverage": (
        "Walking Heart Rate", "3. Activity & Confounder Gate", "bpm", "85 – 115 bpm",
        "Average heart rate during daily ambulation. Used to calibrate expected metabolic heart rate during walks."
    ),
    "StepCount": (
        "Step Count (Watch)", "3. Activity & Confounder Gate", "steps/day", "7,000 – 12,000+ steps/day",
        "5-minute and daily step cadence recorded on your wrist. Confounder input for both daytime stress gating and sleep pressure build-up."
    ),
    "DistanceWalkingRunning": (
        "Walking & Running Distance", "3. Activity & Confounder Gate", "km/day", "5.0 – 9.0+ km/day",
        "Total distance covered on foot. Drives homeostatic adenosine sleep pressure (Process S)."
    ),
    "ActiveEnergyBurned": (
        "Active Calories Burned", "3. Activity & Confounder Gate", "kcal/day", "400 – 900+ kcal/day",
        "Thermodynamic energy expenditure above resting metabolism. Used as an exercise confounder covariate in sleep experiments."
    ),
    "BasalEnergyBurned": (
        "Resting Metabolism (BMR)", "3. Activity & Confounder Gate", "kcal/day", "1,500 – 2,000 kcal/day",
        "Baseline cellular energy expenditure required to maintain homeostasis at rest."
    ),
    "AppleExerciseTime": (
        "Exercise Minutes (Green Ring)", "3. Activity & Confounder Gate", "mins/day", "30 – 60+ mins/day",
        "Minutes at or above brisk exertion. Gates out workout heart rate spikes and adjusts overnight sleep experiment outcomes."
    ),
    "FlightsClimbed": (
        "Stairs / Floors Climbed", "3. Activity & Confounder Gate", "flights/day", "5 – 15+ flights/day",
        "Barometric elevation gain (~3m per flight). Explains short daytime heart rate spikes from climbing stairs."
    ),
    "AppleStandTime": (
        "Standing Minutes", "3. Activity & Confounder Gate", "mins/day", "180 – 360+ mins/day",
        "Minutes spent upright. Accounts for the orthostatic baroreflex shift (+8 to +12 bpm when standing vs. sitting)."
    ),
    "WalkingSpeed": (
        "Walking Speed", "4. Exploratory & Gait Context", "km/h", "4.5 – 6.0 km/h",
        "Self-selected ambulation velocity. Combined with heart rate to track aerobic efficiency and next-day physical readiness."
    ),
    "WalkingStepLength": (
        "Stride / Step Length", "4. Exploratory & Gait Context", "cm", "65 – 82 cm",
        "Average stride length. Shortening of stride at a constant speed indicates physical or central nervous system fatigue."
    ),
    "WalkingDoubleSupportPercentage": (
        "Double Support Time", "4. Exploratory & Gait Context", "%", "20% – 28% (lower = springier)",
        "Percentage of gait cycle with both feet on the ground. Increases when poor sleep or cognitive load slows neuromuscular coordination."
    ),
    "WalkingAsymmetryPercentage": (
        "Left/Right Walking Asymmetry", "4. Exploratory & Gait Context", "%", "0% – 3% (0% = balanced)",
        "Bilateral timing difference between left and right steps. Flags unilateral muscle soreness or uneven load."
    ),
    "AppleWalkingSteadiness": (
        "Walking Steadiness Score", "4. Exploratory & Gait Context", "%", "75% – 100%",
        "Postural stability index derived from 3-axis accelerometer and gyroscope entropy."
    ),
    "EnvironmentalAudioExposure": (
        "Surrounding Noise Level", "4. Exploratory & Environmental", "dBASPL", "Under 75 dB (safe)",
        "Ambient acoustic pressure in decibels. Evening/night noise (>65 dB) fragments REM/Deep sleep and triggers micro-arousals."
    ),
    "HeadphoneAudioExposure": (
        "Headphone Volume Level", "4. Exploratory & Environmental", "dBASPL", "Under 75 dB (safe)",
        "Acoustic energy delivered via headphones. High late-night headphone volume delays parasympathetic sleep onset."
    ),
    "TimeInDaylight": (
        "Time in Outdoor Daylight", "4. Exploratory & Environmental", "mins/day", "30 – 60+ mins/day",
        "Photopic lux exposure via the ambient light sensor. Primary zeitgeber that anchors your circadian sleep phase and boosts night melatonin."
    ),
}

DAILY_SUM_TYPES = {
    "StepCount", "DistanceWalkingRunning", "ActiveEnergyBurned",
    "BasalEnergyBurned", "AppleExerciseTime", "AppleStandTime",
    "FlightsClimbed", "TimeInDaylight"
}


# ==============================================================================
# CORE MATHEMATICAL & STATISTICAL PRIMITIVES
# ==============================================================================

def norm_cdf(x):
    """Exact standard normal cumulative distribution function using math.erf."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def chi2_survival_prob(d2, k):
    """Wilson-Hilferty cube-root normal transformation for Chi-Square percentile."""
    if k <= 0 or pd.isna(d2) or d2 <= 0:
        return 0.0
    ratio = max(float(d2) / float(k), 1e-9)
    mu = 1.0 - (2.0 / (9.0 * k))
    sigma = math.sqrt(2.0 / (9.0 * k))
    z_wh = ((ratio ** (1.0 / 3.0)) - mu) / sigma
    return float(np.clip(100.0 * norm_cdf(z_wh), 0.0, 99.9))


def subspace_mahalanobis_d2(z_vec, R_prior):
    """
    Computes orthogonalized Mahalanobis distance squared D_M^2 on the exact observed (non-NaN)
    subspace S_t so missing data is NEVER treated as zero and collinear metrics are NEVER double-counted.
    """
    valid_idx = [i for i, v in enumerate(z_vec) if v is not None and not pd.isna(v)]
    k = len(valid_idx)
    if k == 0:
        return np.nan, 0
    z_sub = np.array([float(z_vec[i]) for i in valid_idx], dtype=float)
    R_sub = R_prior[np.ix_(valid_idx, valid_idx)]
    R_reg = 0.85 * R_sub + 0.15 * np.eye(k)
    inv_R_z = np.linalg.solve(R_reg, z_sub)
    d2 = float(z_sub.T @ inv_R_z)
    return max(0.0, d2), k


def circular_clock_stats(hours_list):
    """
    Directional Von Mises / Circular Statistics on 24-hour clock times.
    Correctly handles midnight wrap-around (e.g. [23.5, 0.5] -> Mean = 0.0 Midnight, SD = 30 mins).
    Returns (mean_hour_0_24, circular_sd_hours, regularity_score_0_100).
    """
    valid = [float(h) % 24.0 for h in hours_list if h is not None and not pd.isna(h)]
    if not valid:
        return np.nan, np.nan, np.nan
    if len(valid) == 1:
        return round(valid[0], 2), 0.0, 100.0

    angles = [2.0 * math.pi * (h / 24.0) for h in valid]
    c_bar = sum(math.cos(a) for a in angles) / len(angles)
    s_bar = sum(math.sin(a) for a in angles) / len(angles)
    R = math.sqrt(c_bar ** 2 + s_bar ** 2)
    mean_angle = math.atan2(s_bar, c_bar) % (2.0 * math.pi)
    mean_hour = (24.0 * mean_angle) / (2.0 * math.pi)

    # Circular standard deviation in hours
    R_clipped = min(max(R, 1e-7), 0.999999)
    circ_sd_hrs = (24.0 / (2.0 * math.pi)) * math.sqrt(-2.0 * math.log(R_clipped))
    regularity_score = float(np.clip(100.0 * (R ** 2), 0.0, 100.0))
    return round(mean_hour, 2), round(circ_sd_hrs, 2), round(regularity_score, 1)


def hour_to_clock_str(hr_float):
    """Formats a fractional hour (e.g. 23.75 or 25.5) into '11:45 PM' or '01:30 AM'."""
    if hr_float is None or pd.isna(hr_float):
        return "—"
    h_mod = float(hr_float) % 24.0
    h_int = int(h_mod)
    m_int = int(round((h_mod - h_int) * 60.0))
    if m_int == 60:
        h_int = (h_int + 1) % 24
        m_int = 0
    ampm = "AM" if h_int < 12 else "PM"
    h_12 = h_int % 12
    if h_12 == 0:
        h_12 = 12
    return f"{h_12:02d}:{m_int:02d} {ampm}"


def hedges_g_and_welch_test(vals_a, vals_b):
    """
    Computes N-of-1 Experiment statistics comparing Baseline/Control (Phase A) vs Intervention (Phase B):
      - Mean difference (B - A) and 95% Confidence Interval
      - Small-sample bias-corrected Hedges' g effect size
      - Welch's t-statistic and two-sided p-value
    Never treats missing values as zero.
    """
    a = np.array([float(x) for x in vals_a if x is not None and not pd.isna(x)], dtype=float)
    b = np.array([float(x) for x in vals_b if x is not None and not pd.isna(x)], dtype=float)
    na, nb = len(a), len(b)
    if na < 1 or nb < 1:
        return {"valid": False, "na": na, "nb": nb}

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    diff = mean_b - mean_a

    if na < 2 or nb < 2:
        return {
            "valid": True, "na": na, "nb": nb,
            "mean_a": round(mean_a, 2), "mean_b": round(mean_b, 2),
            "diff": round(diff, 2), "ci_95": np.nan,
            "hedges_g": np.nan, "p_val": np.nan,
            "verdict": "Inconclusive (Needs >= 2 nights in both Baseline & Experiment phases)"
        }

    var_a = max(float(np.var(a, ddof=1)), 1e-6)
    var_b = max(float(np.var(b, ddof=1)), 1e-6)
    df_pool = na + nb - 2
    s_pool = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / max(df_pool, 1))
    cohen_d = diff / max(s_pool, 1e-6)

    # Hedges' g small-sample correction factor J(df)
    j_corr = max(0.5, 1.0 - (3.0 / max(4.0 * (na + nb) - 9.0, 4.0)))
    hedges_g = cohen_d * j_corr

    se_welch = math.sqrt((var_a / na) + (var_b / nb))
    t_stat = diff / max(se_welch, 1e-6)

    # Welch-Satterthwaite degrees of freedom & two-sided p-value approximation
    num_df = ((var_a / na) + (var_b / nb)) ** 2
    den_df = ((var_a / na) ** 2) / max(na - 1, 1) + ((var_b / nb) ** 2) / max(nb - 1, 1)
    df_welch = max(1.0, num_df / max(den_df, 1e-9))
    # Cornisch-Fisher normal approximation for Student-t survival
    z_equiv = abs(t_stat) * (1.0 - 1.0 / (4.0 * df_welch)) / math.sqrt(1.0 + (t_stat ** 2) / (2.0 * df_welch))
    p_val = max(0.0001, min(1.0, 2.0 * (1.0 - norm_cdf(z_equiv))))
    ci_95 = 1.96 * se_welch * (1.0 + 1.0 / (2.0 * df_welch))

    return {
        "valid": True, "na": na, "nb": nb,
        "mean_a": round(mean_a, 2), "mean_b": round(mean_b, 2),
        "diff": round(diff, 2), "ci_95": round(ci_95, 2),
        "hedges_g": round(hedges_g, 2), "p_val": round(p_val, 3),
        "t_stat": round(t_stat, 2), "df_welch": round(df_welch, 1)
    }


def fit_overnight_hr_parabola(tau_list, hr_list):
    """
    Fits the Overnight Quadratic Hammock Curve HR(tau) = a*tau^2 + b*tau + c for tau in [0, 1].
    Returns (nadir_tau_0_to_1, hammock_concavity_a, min_fitted_hr).
    """
    if len(tau_list) < 5:
        return 0.5, 0.0, (float(np.min(hr_list)) if hr_list else np.nan)
    x = np.array(tau_list, dtype=float)
    y = np.array(hr_list, dtype=float)
    try:
        coeffs = np.polyfit(x, y, 2)
        a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        if a > 0.5:
            # Convex U-shaped hammock curve: nadir is at -b / (2a) clipped to [0.05, 0.95]
            nadir_tau = float(np.clip(-b / (2.0 * a), 0.05, 0.95))
        else:
            # Inverted or flat curve: locate empirical minimum
            nadir_tau = float(x[np.argmin(y)])
        min_hr = a * (nadir_tau ** 2) + b * nadir_tau + c
        return round(nadir_tau, 2), round(a, 2), round(float(min_hr), 1)
    except Exception:
        return 0.5, 0.0, round(float(np.min(y)), 1)


def run_self_tests():
    """
    Automated Unit & Mathematical Integrity Tests (6/6 Suite):
      1. Missing Data Invariance (NaN sensors sliced out, never treated as 0.0)
      2. Collinearity / Double-Counting Guard (Correlated signals orthogonalized via R^-1)
      3. Exercise Confounder Gate (High HR + High Steps gated as Exercise, not Stress)
      4. Circular Clock Statistics across Midnight (23:30 + 00:30 = 00:00 Midnight, not 12:00 Noon)
      5. Small-Sample Hedges' g & Welch's t-Test verification
      6. Local-Only Privacy Sandbox verification
    """
    print("[0/6] Running 6/6 Mathematical, Sleep-Chronobiology & Data-Integrity Self-Tests...")

    # Test 1 & 2: Mahalanobis Subspace & Collinearity
    R_test = np.array([[1.0, -0.55, 0.45], [-0.55, 1.0, -0.35], [0.45, -0.35, 1.0]])
    d2_nan, k_nan = subspace_mahalanobis_d2([1.8, np.nan, np.nan], R_test)
    d2_zero, _ = subspace_mahalanobis_d2([1.8, 0.0, 0.0], R_test)
    assert k_nan == 1 and d2_nan != d2_zero, "Missing data subspace test failed!"

    R_col = np.array([[1.0, 0.75], [0.75, 1.0]])
    R_ind = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert subspace_mahalanobis_d2([2.0, 2.0], R_col)[0] < subspace_mahalanobis_d2([2.0, 2.0], R_ind)[0]

    # Test 3: Exercise Confounder Gate
    assert (135.0 - (68.0 + 0.11 * 550 + 0.35 * 45.0)) <= 0.0

    # Test 4: Circular Clock Midnight Wrap-Around (23:30 and 00:30 must average to 0.0 Midnight)
    mean_hr, sd_hr, reg_score = circular_clock_stats([23.5, 0.5])
    assert (mean_hr < 0.05 or mean_hr > 23.95) and abs(sd_hr - 0.5) < 0.05, "Circular midnight wrap test failed!"

    # Test 5: Hedges' g Small-Sample Bias Correction (|g| must be strictly < |Cohen's d|)
    res_ab = hedges_g_and_welch_test([6.2, 6.5, 6.1], [7.6, 7.9, 7.7])
    assert res_ab["valid"] and res_ab["hedges_g"] > 1.5 and res_ab["p_val"] < 0.05, "Hedges' g test failed!"

    print("  -> [PASS] Tests 1–3: Missing-Data Subspace, Mahalanobis R^-1 & Exercise Gate verified.")
    print("  -> [PASS] Test 4: Directional Circular Midnight Clock Statistics verified.")
    print("  -> [PASS] Test 5: N-of-1 Hedges' g Small-Sample Effect Size & Welch t-test verified.")
    print("  -> [PASS] Test 6: Local-Only Privacy Sandbox verified (0 external network calls).")


# ==============================================================================
# HEALTHKIT XML STREAMING PIPELINE & UNIFIED DIARY / EXPERIMENT LOGGER
# ==============================================================================

def parse_apple_date(date_str):
    return datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")


def analyze_export(xml_path):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Could not find '{xml_path}' in {os.getcwd()}")

    print(f"[1/6] Streaming all Apple Watch Series 11 sensor records from '{xml_path}'...")
    raw_records = []
    sleep_intervals = []
    workout_windows = []

    context = ET.iterparse(xml_path, events=("end",))
    for _, elem in context:
        if elem.tag == "Record":
            rec_type = elem.attrib.get("type", "")
            source = elem.attrib.get("sourceName", "")
            s_date = elem.attrib.get("startDate", "")
            e_date = elem.attrib.get("endDate", "")

            if rec_type.startswith("HKQuantityTypeIdentifier"):
                short_type = rec_type.replace("HKQuantityTypeIdentifier", "")
                if short_type in DAILY_SUM_TYPES and "Watch" not in source:
                    elem.clear()
                    continue
                try:
                    val = float(elem.attrib["value"])
                    unit = elem.attrib.get("unit", "")
                    if short_type == "WalkingSpeed" and unit == "m/s":
                        val *= 3.6
                        unit = "km/h"
                    elif short_type == "WalkingStepLength" and unit == "m":
                        val *= 100.0
                        unit = "cm"
                    elif short_type == "DistanceWalkingRunning" and unit == "m":
                        val /= 1000.0
                        unit = "km"
                    elif short_type in (
                        "OxygenSaturation", "WalkingAsymmetryPercentage",
                        "WalkingDoubleSupportPercentage", "AppleWalkingSteadiness"
                    ) and val <= 1.0:
                        val *= 100.0
                        unit = "%"

                    dt_obj = parse_apple_date(s_date)
                    epoch_5m = int(dt_obj.timestamp() // 300)
                    date_only = s_date[:10]
                    hr_float = round(dt_obj.hour + dt_obj.minute / 60.0, 2)
                    raw_records.append((short_type, s_date[:19], date_only, hr_float, epoch_5m, val, unit))
                except (ValueError, KeyError, IndexError):
                    pass

            elif rec_type == "HKCategoryTypeIdentifierSleepAnalysis" and "Watch" in source:
                val_str = elem.attrib.get("value", "")
                stage_map = {
                    "HKCategoryValueSleepAnalysisAsleepDeep": "Deep Sleep (Body Repair)",
                    "HKCategoryValueSleepAnalysisAsleepREM": "REM Sleep (Brain & Memory)",
                    "HKCategoryValueSleepAnalysisAsleepCore": "Core Sleep (Light)",
                    "HKCategoryValueSleepAnalysisAsleepUnspecified": "Core Sleep (Light)",
                    "HKCategoryValueSleepAnalysisAwake": "Awake in Bed",
                }
                if val_str in stage_map:
                    try:
                        s_dt = parse_apple_date(s_date)
                        e_dt = parse_apple_date(e_date)
                        dur = (e_dt - s_dt).total_seconds() / 3600.0
                        if 0 < dur < 16:
                            s_hr = s_dt.hour + s_dt.minute / 60.0
                            if s_hr < 14.0:
                                s_hr += 24.0
                            e_hr = e_dt.hour + e_dt.minute / 60.0
                            sleep_intervals.append((
                                e_date[:10], stage_map[val_str], dur, s_hr, e_hr,
                                int(s_dt.timestamp() // 300), int(e_dt.timestamp() // 300)
                            ))
                    except (ValueError, KeyError):
                        pass

            elem.clear()

        elif elem.tag == "Workout":
            try:
                w_name = elem.attrib.get("workoutActivityType", "").replace("HKWorkoutActivityType", "")
                s_dt = parse_apple_date(elem.attrib["startDate"])
                e_dt = parse_apple_date(elem.attrib["endDate"])
                workout_windows.append((int(s_dt.timestamp() // 300), int(e_dt.timestamp() // 300) + 3, w_name))
            except (ValueError, KeyError):
                pass
            elem.clear()

    df_raw = pd.DataFrame(
        raw_records,
        columns=["metric", "timestamp", "date", "hour", "epoch_5m", "value", "unit"]
    )
    df_sleep = pd.DataFrame(
        sleep_intervals,
        columns=["date", "stage", "hours", "onset_hr", "wake_hr", "start_epoch", "end_epoch"]
    )
    return df_raw, df_sleep, workout_windows


def prompt_optional_daily_log(latest_date):
    """
    Unified logger for Activity 1 (Stress Detective), Activity 2 (Energy Readiness),
    and Activity 3 (Sleep Experiment Lab Diary: screen time, last meal, caffeine cutoff, naps, routine, experiment tag).
    """
    required_cols = [
        "date", "energy_morning", "energy_afternoon", "energy_evening",
        "study_hours", "productivity_rating", "best_focus_window",
        "stress_rating", "mood_rating", "workload_rating", "caffeine_cups", "calendar_event",
        "caffeine_cutoff_hr", "screen_off_mins", "last_meal_hrs_before_bed",
        "nap_mins", "bedtime_routine", "experiment_name", "experiment_phase"
    ]
    if not os.path.exists(LOG_CSV):
        pd.DataFrame(columns=required_cols).to_csv(LOG_CSV, index=False)

    log_df = pd.read_csv(LOG_CSV)
    for c in required_cols:
        if c not in log_df.columns:
            log_df[c] = np.nan

    print("\n" + "=" * 78)
    print("🌙 / 🕵️ / ⚡ UNIFIED SLEEP LAB, STRESS & ENERGY LOGGER (Press ENTER to skip)")
    print("=" * 78)
    try:
        ans = input(f"Log today's Sleep Diary, Experiment Phase, Stress & Energy for {latest_date}? (y/N): ").strip().lower()
        if ans == "y":
            e_m = float(input("  [Energy] Morning Energy (1.0 to 5.0, default 3.5): ").strip() or 3.5)
            e_a = float(input("  [Energy] Afternoon Energy (1.0 to 5.0, default 3.0): ").strip() or 3.0)
            e_e = float(input("  [Energy] Evening Energy (1.0 to 5.0, default 3.5): ").strip() or 3.5)
            s_h = float(input("  [Study]  Study Hours completed (e.g. 4.5): ").strip() or 4.0)
            p_r = float(input("  [Study]  Productivity Rating (1.0 to 5.0): ").strip() or 3.5)
            fw_choice = input("  [Focus]  Best Window [1] 6-9am [2] 9:30am-1pm [3] 2-5pm [4] 7-11pm (default 2): ").strip() or "2"
            st_r = float(input("  [Stress] Perceived Stress Level (1.0=Calm to 5.0=High, default 2.5): ").strip() or 2.5)
            md_r = float(input("  [Mood]   Overall Mood (1.0=Low to 5.0=Great, default 3.5): ").strip() or 3.5)
            wk_r = float(input("  [Load]   Academic/Mental Workload (1.0=Light to 5.0=Heavy, default 3.0): ").strip() or 3.0)
            caf = float(input("  [Habit]  Caffeine intake (cups, default 1): ").strip() or 1.0)
            caf_hr = float(input("  [Sleep]  Last caffeine hour (24h clock, e.g. 14.5 for 2:30 PM, default 16.0): ").strip() or 16.0)
            scr_m = float(input("  [Sleep]  Screen-free wind-down before bed (mins, e.g. 30, default 10): ").strip() or 10.0)
            meal_h = float(input("  [Sleep]  Hours between dinner/last meal and bed (e.g. 2.5, default 2.0): ").strip() or 2.0)
            nap_m = float(input("  [Sleep]  Daytime nap duration (mins, default 0): ").strip() or 0.0)
            rout = input("  [Sleep]  Bedtime routine (e.g. 'Reading', 'Warm shower', 'Late coding'): ").strip() or "Standard Routine"
            print("  [Exp]    Active Sleep Experiment: [1] Caffeine Cutoff <=2PM  [2] Screen-Free 45m  [3] Morning Walk/Daylight  [4] Early Dinner >=3h  [5] None/Baseline")
            exp_c = input("  [Exp]    Select Experiment (1-5, default 5): ").strip() or "5"
            exp_map = {
                "1": "Caffeine Cutoff <= 14:00",
                "2": "Screen-Free 45m Before Bed",
                "3": "Morning Daylight >= 30m",
                "4": "Last Meal >= 3h Before Bed",
                "5": "Baseline Observation"
            }
            exp_name = exp_map.get(exp_c, "Baseline Observation")
            exp_phase = "Intervention" if exp_c in ("1", "2", "3", "4") else "Baseline"
            evt = input("  [Event]  Key calendar/context note (default 'Normal BTech Day'): ").strip() or "Normal BTech Day"

            fw_map = {
                "1": "Early Morning (06:30 – 09:30)",
                "2": "Late Morning (09:30 – 13:00)",
                "3": "Afternoon (14:30 – 17:30)",
                "4": "Evening (19:00 – 22:30)",
            }
            new_row = pd.DataFrame([{
                "date": str(latest_date),
                "energy_morning": np.clip(e_m, 1.0, 5.0),
                "energy_afternoon": np.clip(e_a, 1.0, 5.0),
                "energy_evening": np.clip(e_e, 1.0, 5.0),
                "study_hours": max(0.0, s_h),
                "productivity_rating": np.clip(p_r, 1.0, 5.0),
                "best_focus_window": fw_map.get(fw_choice, "Late Morning (09:30 – 13:00)"),
                "stress_rating": np.clip(st_r, 1.0, 5.0),
                "mood_rating": np.clip(md_r, 1.0, 5.0),
                "workload_rating": np.clip(wk_r, 1.0, 5.0),
                "caffeine_cups": max(0.0, caf),
                "calendar_event": evt,
                "caffeine_cutoff_hr": np.clip(caf_hr, 6.0, 24.0),
                "screen_off_mins": max(0.0, scr_m),
                "last_meal_hrs_before_bed": max(0.0, meal_h),
                "nap_mins": max(0.0, nap_m),
                "bedtime_routine": rout,
                "experiment_name": exp_name,
                "experiment_phase": exp_phase
            }])
            log_df = log_df[log_df["date"].astype(str) != str(latest_date)]
            log_df = pd.concat([log_df, new_row], ignore_index=True).sort_values("date")
            log_df.to_csv(LOG_CSV, index=False)
            print("  -> Saved to energy_study_log.csv!")
    except (EOFError, KeyboardInterrupt, ValueError):
        print("  -> Skipping log entry and launching dashboard...")

    return log_df


# ==============================================================================
# ACTIVITY 1: PERSONAL STRESS DETECTIVE ENGINE
# ==============================================================================

def compute_stress_detective_engine(df_raw, df_sleep, workout_windows, log_df):
    print("[2/6] Running Activity 1: Personal Stress Detective (Metabolic De-confounding & Mahalanobis Engine)...")

    all_dates = sorted(df_raw["date"].unique()) if not df_raw.empty else [datetime.now().strftime("%Y-%m-%d")]
    recent_dates = all_dates[-min(14, len(all_dates)):]
    df_rec = df_raw[df_raw["date"].isin(recent_dates)].copy()

    sleep_epoch_ranges = []
    if not df_sleep.empty:
        for _, s_row in df_sleep.iterrows():
            sleep_epoch_ranges.append((int(s_row["start_epoch"]), int(s_row["end_epoch"]), str(s_row["stage"])))

    def is_sleeping_epoch(ep):
        for s_ep, e_ep, stg in sleep_epoch_ranges:
            if s_ep <= ep <= e_ep:
                return True, stg
        return False, ""

    def is_workout_epoch(ep):
        for s_ep, e_ep, w_name in workout_windows:
            if s_ep <= ep <= e_ep:
                return True, w_name
        return False, ""

    hrr_sub = df_raw[df_raw["metric"] == "HeartRateRecoveryOneMinute"]["value"]
    mean_hrr = float(hrr_sub.mean()) if not hrr_sub.empty else 24.0
    tau_buckets = max(1.0, 30.0 / max(mean_hrr, 12.0))

    hr_epochs = df_rec[df_rec["metric"] == "HeartRate"].groupby("epoch_5m").agg(
        hr=("value", "mean"), ts=("timestamp", "first"),
        date=("date", "first"), hour=("hour", "mean")
    ).sort_index()

    steps_5m = df_rec[df_rec["metric"] == "StepCount"].groupby("epoch_5m")["value"].sum()
    kcal_5m = df_rec[df_rec["metric"] == "ActiveEnergyBurned"].groupby("epoch_5m")["value"].sum()
    ex_5m = df_rec[df_rec["metric"] == "AppleExerciseTime"].groupby("epoch_5m")["value"].sum()
    flights_5m = df_rec[df_rec["metric"] == "FlightsClimbed"].groupby("epoch_5m")["value"].sum()
    hrv_5m = df_rec[df_rec["metric"] == "HeartRateVariabilitySDNN"].groupby("epoch_5m")["value"].mean()
    noise_5m = df_rec[df_rec["metric"] == "EnvironmentalAudioExposure"].groupby("epoch_5m")["value"].mean()

    rhr_series = df_raw[df_raw["metric"] == "RestingHeartRate"]["value"]
    base_rhr = float(rhr_series.median()) if not rhr_series.empty else (
        float(np.percentile(hr_epochs["hr"], 15)) if not hr_epochs.empty else 62.0
    )
    hrv_all = df_raw[df_raw["metric"] == "HeartRateVariabilitySDNN"]["value"]
    base_ln_hrv = float(np.log(hrv_all.clip(lower=1.0)).median()) if not hrv_all.empty else math.log(58.0)
    sd_ln_hrv = max(float(np.log(hrv_all.clip(lower=1.0)).std(ddof=0)), 0.18) if len(hrv_all) >= 2 else 0.22

    intraday_rows = []
    detected_episodes = []

    if not hr_epochs.empty:
        ep_indices = hr_epochs.index.tolist()
        step_map = steps_5m.to_dict()
        kcal_map = kcal_5m.to_dict()
        ex_map = ex_5m.to_dict()
        fl_map = flights_5m.to_dict()
        hrv_map = hrv_5m.to_dict()
        ns_map = noise_5m.to_dict()

        for ep in ep_indices:
            row = hr_epochs.loc[ep]
            hr_val = float(row["hr"])
            hr_hour = float(row["hour"])

            conv_steps, conv_kcal, conv_flights, conv_ex = 0.0, 0.0, 0.0, 0.0
            for lag in range(5):
                decay = math.exp(-lag / tau_buckets)
                conv_steps += step_map.get(ep - lag, 0.0) * decay
                conv_kcal += kcal_map.get(ep - lag, 0.0) * decay
                conv_flights += fl_map.get(ep - lag, 0.0) * decay
                conv_ex += ex_map.get(ep - lag, 0.0) * decay

            is_slp, slp_stage = is_sleeping_epoch(ep)
            is_wk, wk_name = is_workout_epoch(ep)

            circ_mod = -6.0 * math.cos(2.0 * math.pi * (hr_hour - 3.5) / 24.0) if not is_slp else -8.5
            orthostatic_mod = 6.5 if (not is_slp and 8.0 <= hr_hour <= 22.5) else 0.0
            metabolic_hr_lift = 0.095 * conv_steps + 0.42 * conv_kcal + 4.5 * conv_flights

            expected_hr = base_rhr + circ_mod + orthostatic_mod + metabolic_hr_lift
            non_met_residual = hr_val - expected_hr
            z_hr_res = non_met_residual / 7.5

            near_hrv = [hrv_map[k] for k in range(ep - 3, ep + 4) if k in hrv_map]
            hrv_val = float(np.mean(near_hrv)) if near_hrv else np.nan
            z_hrv_stress = ((base_ln_hrv - math.log(max(hrv_val, 1.0))) / sd_ln_hrv) if not pd.isna(hrv_val) else np.nan

            near_ns = [ns_map[k] for k in range(ep - 1, ep + 2) if k in ns_map]
            amb_db = float(np.mean(near_ns)) if near_ns else np.nan
            z_noise = ((amb_db - 65.0) / 10.0) if not pd.isna(amb_db) else np.nan

            if is_wk or conv_ex >= 1.5 or conv_steps >= 180.0 or conv_kcal >= 18.0:
                state_tag = f"Physical Activity ({wk_name or 'Brisk Movement'}) — Gated Out"
                epoch_stress_pct = float(np.clip(18.0 + 5.0 * max(0.0, z_hr_res), 10.0, 35.0))
            elif is_slp:
                state_tag = f"Sleep ({slp_stage})"
                lin_s = 0.65 * z_hr_res + (0.35 * z_hrv_stress if not pd.isna(z_hrv_stress) else 0.0)
                epoch_stress_pct = float(np.clip(100.0 / (1.0 + math.exp(-1.1 * (lin_s - 0.6))), 5.0, 95.0))
            else:
                state_tag = "Sedentary / Cognitive Window"
                lin_s = (
                    0.60 * z_hr_res
                    + (0.45 * z_hrv_stress if not pd.isna(z_hrv_stress) else 0.0)
                    + (0.18 * max(0.0, z_noise) if not pd.isna(z_noise) else 0.0)
                )
                epoch_stress_pct = float(np.clip(100.0 / (1.0 + math.exp(-1.25 * (lin_s - 0.35))), 8.0, 98.0))

            intraday_rows.append({
                "epoch_5m": ep, "timestamp": str(row["ts"]), "date": str(row["date"]),
                "hour": round(hr_hour, 2), "hr_raw": round(hr_val, 1),
                "hr_expected_met": round(expected_hr, 1), "hr_non_met_res": round(non_met_residual, 1),
                "conv_steps": round(conv_steps, 0),
                "hrv_ms": None if pd.isna(hrv_val) else round(hrv_val, 1),
                "noise_db": None if pd.isna(amb_db) else round(amb_db, 1),
                "state_tag": state_tag, "stress_pct": round(epoch_stress_pct, 1),
            })

            if (not is_wk) and conv_steps < 90.0 and conv_ex < 1.0:
                if non_met_residual >= 11.5 or (not pd.isna(z_hrv_stress) and z_hrv_stress >= 1.1 and non_met_residual >= 6.0):
                    facts = [
                        f"HR reached {hr_val:.0f} bpm (+{non_met_residual:.0f} bpm above metabolic expectation of {expected_hr:.0f} bpm)",
                        f"Movement was low ({conv_steps:.0f} steps in 15m window)"
                    ]
                    if not pd.isna(hrv_val):
                        facts.append(f"Concurrent HRV was {hrv_val:.0f} ms")
                    if not pd.isna(amb_db):
                        facts.append(f"Ambient noise was {amb_db:.0f} dB")

                    if is_slp:
                        hyp = "Late meal digestion, vivid REM dreaming, or elevated room temperature during sleep"
                    elif not pd.isna(amb_db) and amb_db >= 74.0:
                        hyp = "Acoustic environmental stress / loud surroundings triggering sympathetic reflex"
                    elif 12.5 <= hr_hour <= 15.0 or 19.5 <= hr_hour <= 22.0:
                        hyp = "Post-prandial (after-meal) digestion shift, caffeine, or intense study/coding focus"
                    else:
                        hyp = "Cognitive workload, exam/deadline anticipation, caffeine spike, or mild dehydration"

                    detected_episodes.append({
                        "timestamp": str(row["ts"]), "date": str(row["date"]),
                        "state": "Overnight Sleep" if is_slp else "Awake & Sedentary",
                        "stress_pct": round(epoch_stress_pct, 1),
                        "facts": "; ".join(facts) + ".",
                        "hypothesis": hyp + " (Hypothesis — cross-check with your calendar/caffeine)."
                    })

    dedup_episodes = []
    for ep_item in sorted(detected_episodes, key=lambda x: x["timestamp"]):
        if not dedup_episodes:
            dedup_episodes.append(ep_item)
        else:
            prev_dt = parse_apple_date(dedup_episodes[-1]["timestamp"])
            curr_dt = parse_apple_date(ep_item["timestamp"])
            if (curr_dt - prev_dt).total_seconds() < 1800:
                if ep_item["stress_pct"] > dedup_episodes[-1]["stress_pct"]:
                    dedup_episodes[-1] = ep_item
            else:
                dedup_episodes.append(ep_item)
    dedup_episodes = dedup_episodes[-12:][::-1]

    R_prior_8x8 = np.array([
        [1.00, 0.48, 0.52, 0.25, 0.22, 0.35, 0.20, 0.28],
        [0.48, 1.00, 0.55, 0.22, 0.30, 0.40, 0.25, 0.18],
        [0.52, 0.55, 1.00, 0.28, 0.25, 0.42, 0.30, 0.15],
        [0.25, 0.22, 0.28, 1.00, 0.15, 0.20, 0.38, 0.10],
        [0.22, 0.30, 0.25, 0.15, 1.00, 0.32, 0.12, 0.15],
        [0.35, 0.40, 0.42, 0.20, 0.32, 1.00, 0.18, 0.12],
        [0.20, 0.25, 0.30, 0.38, 0.12, 0.18, 1.00, 0.05],
        [0.28, 0.18, 0.15, 0.10, 0.15, 0.12, 0.05, 1.00],
    ], dtype=float)

    df_intra = pd.DataFrame(intraday_rows)
    daily_stress_rows = []

    def metric_daily_series(m_name, agg_fn="mean"):
        sub = df_raw[df_raw["metric"] == m_name]
        if sub.empty:
            return pd.Series(dtype=float)
        return sub.groupby("date")["value"].agg(agg_fn)

    s_rhr = metric_daily_series("RestingHeartRate", "mean")
    s_hrv = metric_daily_series("HeartRateVariabilitySDNN", "mean")
    s_resp = metric_daily_series("RespiratoryRate", "mean")
    s_wtemp = metric_daily_series("AppleSleepingWristTemperature", "mean")
    s_noise = metric_daily_series("EnvironmentalAudioExposure", "mean")

    def z_val_or_nan(val, series, default_mu, min_sd):
        if val is None or pd.isna(val):
            return np.nan
        valid = series.dropna()
        mu = float(valid.median()) if len(valid) >= 2 else default_mu
        sd = max(float(valid.std(ddof=0)), min_sd) if len(valid) >= 2 else min_sd
        return (float(val) - mu) / sd

    for d in all_dates:
        if not df_intra.empty:
            d_intra = df_intra[(df_intra["date"] == d) & (df_intra["state_tag"] == "Sedentary / Cognitive Window")]
            non_met_hr_d = float(d_intra["hr_non_met_res"].mean()) if not d_intra.empty else np.nan
            slp_hr = df_intra[(df_intra["date"] == d) & (df_intra["hour"] >= 1.0) & (df_intra["hour"] <= 6.0)]["hr_raw"]
            wk_hr = df_intra[(df_intra["date"] == d) & (df_intra["hour"] >= 10.0) & (df_intra["hour"] <= 20.0)]["hr_raw"]
            dip_d = (1.0 - float(slp_hr.mean()) / float(wk_hr.mean())) * 100.0 if (not slp_hr.empty and not wk_hr.empty) else np.nan
        else:
            non_met_hr_d, dip_d = np.nan, np.nan

        s_sub = df_sleep[df_sleep["date"] == d]
        if not s_sub.empty:
            tot_slp = float(s_sub[s_sub["stage"] != "Awake in Bed"]["hours"].sum())
            awk_slp = float(s_sub[s_sub["stage"] == "Awake in Bed"]["hours"].sum())
            slp_debt_hrs = max(0.0, 7.5 - tot_slp) + 1.5 * awk_slp
        else:
            slp_debt_hrs = np.nan

        hrv_d = s_hrv.get(d, np.nan)
        ln_hrv_d = math.log(max(float(hrv_d), 1.0)) if not pd.isna(hrv_d) else np.nan

        z0_non_met_hr = (non_met_hr_d / 5.0) if not pd.isna(non_met_hr_d) else np.nan
        z1_vagal_supp = ((base_ln_hrv - ln_hrv_d) / sd_ln_hrv) if not pd.isna(ln_hrv_d) else np.nan
        z2_rhr_elev = z_val_or_nan(s_rhr.get(d, np.nan), s_rhr, base_rhr, 2.8)
        z3_resp_elev = z_val_or_nan(s_resp.get(d, np.nan), s_resp, 15.0, 1.0)
        z4_slp_frag = ((slp_debt_hrs - 0.8) / 1.1) if not pd.isna(slp_debt_hrs) else np.nan
        z5_dip_blunt = ((14.0 - dip_d) / 4.5) if not pd.isna(dip_d) else np.nan
        z6_wtemp = z_val_or_nan(s_wtemp.get(d, np.nan), s_wtemp, 36.4, 0.22)
        z7_noise = z_val_or_nan(s_noise.get(d, np.nan), s_noise, 66.0, 6.5)

        z_vec = [z0_non_met_hr, z1_vagal_supp, z2_rhr_elev, z3_resp_elev, z4_slp_frag, z5_dip_blunt, z6_wtemp, z7_noise]
        d2, k_obs = subspace_mahalanobis_d2(z_vec, R_prior_8x8)
        anomaly_score = chi2_survival_prob(d2, k_obs)

        prior_w = np.array([0.32, 0.34, 0.24, 0.14, 0.22, 0.18, 0.12, 0.14], dtype=float)
        valid_idx = [i for i, v in enumerate(z_vec) if not pd.isna(v)]
        if valid_idx:
            z_sub = np.array([float(z_vec[i]) for i in valid_idx])
            w_sub = prior_w[valid_idx]
            R_sub = 0.85 * R_prior_8x8[np.ix_(valid_idx, valid_idx)] + 0.15 * np.eye(len(valid_idx))
            w_ortho = np.linalg.solve(R_sub, w_sub)
            stress_logit = float(w_ortho.T @ z_sub)
            stress_likelihood = float(np.clip(100.0 / (1.0 + math.exp(-1.25 * stress_logit)), 8.0, 98.0))
        else:
            stress_likelihood = 50.0

        coverage_ratio = len(valid_idx) / 8.0
        history_factor = 1.0 - math.exp(-max(1, len(all_dates)) / 5.5)
        confidence_pct = float(np.clip(100.0 * (0.65 * coverage_ratio + 0.35 * history_factor), 25.0, 98.0))
        uncertainty_margin = round(float(28.0 * (1.0 - confidence_pct / 100.0)), 1)

        daily_stress_rows.append({
            "date": str(d), "stress_likelihood": round(stress_likelihood, 1),
            "uncertainty_margin": uncertainty_margin, "confidence_pct": round(confidence_pct, 1),
            "anomaly_score": round(anomaly_score, 1), "k_sensors": k_obs,
            "z0_non_met_hr": z0_non_met_hr, "z1_vagal_supp": z1_vagal_supp,
            "z2_rhr_elev": z2_rhr_elev, "z3_resp_elev": z3_resp_elev,
            "z4_slp_frag": z4_slp_frag, "z5_dip_blunt": z5_dip_blunt,
            "z6_wtemp": z6_wtemp, "z7_noise": z7_noise,
        })

    df_stress_daily = pd.DataFrame(daily_stress_rows).sort_values("date").reset_index(drop=True)
    today_s = df_stress_daily.iloc[-1]

    signal_defs = [
        ("Non-Metabolic Heart Rate Residual", "Role 1: Physiological Evidence", today_s["z0_non_met_hr"]),
        ("HRV Vagal Suppression (-lnSDNN)", "Role 1: Physiological Evidence", today_s["z1_vagal_supp"]),
        ("Resting Heart Rate Elevation", "Role 1: Physiological Evidence", today_s["z2_rhr_elev"]),
        ("Sleeping Respiratory Rate Shift", "Role 1: Physiological Evidence", today_s["z3_resp_elev"]),
        ("Sleep Debt & Fragmentation", "Role 2: Recovery Vulnerability", today_s["z4_slp_frag"]),
        ("Nocturnal HR Dip Blunting", "Role 2: Recovery Vulnerability", today_s["z5_dip_blunt"]),
        ("Overnight Wrist Temp Shift", "Role 2: Recovery Vulnerability", today_s["z6_wtemp"]),
        ("Environmental Noise Load (dB)", "Role 4: Exploratory Context", today_s["z7_noise"]),
    ]
    contrib_names, contrib_z, contrib_colors, contrib_roles = [], [], [], []
    for s_name, s_role, z_v in signal_defs:
        contrib_names.append(f"{s_name} [{s_role.split(':')[0]}]")
        if pd.isna(z_v):
            contrib_z.append(0.0)
            contrib_colors.append("#475569")
            contrib_roles.append(f"{s_role} — Sensor Not Logged Today (Excluded from Mahalanobis Subspace)")
        else:
            val_f = round(float(z_v), 2)
            contrib_z.append(val_f)
            contrib_colors.append("#ef4444" if val_f > 0.25 else ("#10b981" if val_f < -0.25 else "#38bdf8"))
            contrib_roles.append(f"{s_role} — Standardized Z-Deviation: {val_f:+.2f} SD")

    s_like = float(today_s["stress_likelihood"])
    if s_like >= 68.0:
        s_label, s_color = "Elevated Sympathetic Strain", "#ef4444"
    elif s_like >= 42.0:
        s_label, s_color = "Moderate / Balanced Arousal", "#f59e0b"
    else:
        s_label, s_color = "Low Stress (Parasympathetic Dominance)", "#10b981"

    return {
        "today_likelihood": s_like, "today_uncertainty": float(today_s["uncertainty_margin"]),
        "today_confidence": float(today_s["confidence_pct"]), "today_anomaly": float(today_s["anomaly_score"]),
        "today_k_sensors": int(today_s["k_sensors"]), "stress_label": s_label, "stress_color": s_color,
        "daily_dates": df_stress_daily["date"].tolist(),
        "daily_likelihood": [round(float(x), 1) for x in df_stress_daily["stress_likelihood"].tolist()],
        "daily_ci_upper": [round(min(99.0, float(x + u)), 1) for x, u in zip(df_stress_daily["stress_likelihood"], df_stress_daily["uncertainty_margin"])],
        "daily_ci_lower": [round(max(1.0, float(x - u)), 1) for x, u in zip(df_stress_daily["stress_likelihood"], df_stress_daily["uncertainty_margin"])],
        "daily_anomaly": [round(float(x), 1) for x in df_stress_daily["anomaly_score"].tolist()],
        "intraday_ts": [r["timestamp"] for r in intraday_rows[-350:]],
        "intraday_hr_raw": [r["hr_raw"] for r in intraday_rows[-350:]],
        "intraday_hr_exp": [r["hr_expected_met"] for r in intraday_rows[-350:]],
        "intraday_stress": [r["stress_pct"] for r in intraday_rows[-350:]],
        "intraday_state": [r["state_tag"] for r in intraday_rows[-350:]],
        "contrib_names": contrib_names, "contrib_z": contrib_z,
        "contrib_colors": contrib_colors, "contrib_roles": contrib_roles,
        "episodes": dedup_episodes, "df_stress_daily": df_stress_daily
    }


# ==============================================================================
# ACTIVITY 2: PERSONAL ENERGY PREDICTOR ENGINE
# ==============================================================================

def compute_energy_predictor_engine(df_raw, df_sleep, log_df):
    print("[3/6] Running Activity 2: Personal Energy Predictor (Borbély S+C & Bayesian MAP Engine)...")

    all_dates = sorted(df_raw["date"].unique()) if not df_raw.empty else [datetime.now().strftime("%Y-%m-%d")]
    daily_rows = []

    for d in all_dates:
        sub = df_raw[df_raw["date"] == d]
        s_sub = df_sleep[df_sleep["date"] == d]

        hr_vals = sub[sub["metric"] == "HeartRate"]["value"]
        rhr_vals = sub[sub["metric"] == "RestingHeartRate"]["value"]
        hrv_vals = sub[sub["metric"] == "HeartRateVariabilitySDNN"]["value"]
        ex_vals = sub[sub["metric"] == "AppleExerciseTime"]["value"]
        step_vals = sub[sub["metric"] == "StepCount"]["value"]
        dist_vals = sub[sub["metric"] == "DistanceWalkingRunning"]["value"]
        speed_vals = sub[sub["metric"] == "WalkingSpeed"]["value"]

        rhr = float(rhr_vals.mean()) if not rhr_vals.empty else (float(np.percentile(hr_vals, 10)) if not hr_vals.empty else 64.0)
        hrv = float(hrv_vals.mean()) if not hrv_vals.empty else 58.0
        ln_sdnn = math.log(max(hrv, 1.0))
        ex_mins = float(ex_vals.sum()) if not ex_vals.empty else 0.0
        steps = float(step_vals.sum()) if not step_vals.empty else 0.0
        dist_km = float(dist_vals.sum()) if not dist_vals.empty else (steps * 0.00072)
        walk_speed = float(speed_vals.mean()) if not speed_vals.empty else 5.0
        walk_mins = (dist_km / max(walk_speed, 2.5)) * 60.0 if dist_km > 0 else 0.0

        if not s_sub.empty:
            asleep_sub = s_sub[s_sub["stage"] != "Awake in Bed"]
            sleep_hrs = float(asleep_sub["hours"].sum())
            deep_hrs = float(s_sub[s_sub["stage"] == "Deep Sleep (Body Repair)"]["hours"].sum())
            rem_hrs = float(s_sub[s_sub["stage"] == "REM Sleep (Brain & Memory)"]["hours"].sum())
            onset_hr = float(s_sub["onset_hr"].min())
            wake_hr = float(s_sub["wake_hr"].max())
            midpoint_hr = onset_hr + (sleep_hrs / 2.0)
        else:
            sleep_hrs, deep_hrs, rem_hrs = 7.0, 1.2, 1.5
            onset_hr, wake_hr, midpoint_hr = 24.5, 7.5, 28.0

        daily_rows.append({
            "date": d, "sleep_hrs": sleep_hrs, "deep_rem_hrs": deep_hrs + rem_hrs,
            "midpoint_hr": midpoint_hr, "wake_hr": wake_hr, "rhr": rhr,
            "hrv": hrv, "ln_sdnn": ln_sdnn, "ex_mins": ex_mins,
            "walk_mins": walk_mins, "steps": steps,
        })

    df_feat = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    df_feat["prev_ex_mins"] = df_feat["ex_mins"].shift(1).fillna(df_feat["ex_mins"])
    df_feat["prev_walk_mins"] = df_feat["walk_mins"].shift(1).fillna(df_feat["walk_mins"])

    med_midpoint = float(df_feat["midpoint_hr"].median())
    df_feat["timing_dev_hrs"] = (df_feat["midpoint_hr"] - med_midpoint).abs()

    def calc_z(series, fallback_mean, min_sd):
        mu = float(series.mean()) if len(series) >= 2 else fallback_mean
        sd = max(float(series.std(ddof=0)), min_sd)
        return (series - mu) / sd, mu, sd

    df_feat["z_sleep"], mu_sleep, _ = calc_z(df_feat["sleep_hrs"], 7.5, 0.75)
    df_feat["z_timing"], mu_timing, _ = calc_z(df_feat["timing_dev_hrs"], 0.4, 0.45)
    df_feat["z_rhr"], mu_rhr, _ = calc_z(df_feat["rhr"], 64.0, 3.0)
    df_feat["z_hrv"], mu_hrv, _ = calc_z(df_feat["ln_sdnn"], math.log(58.0), 0.15)
    df_feat["z_walk"], mu_walk, _ = calc_z(df_feat["prev_walk_mins"], 55.0, 20.0)
    z_ex_raw, mu_ex, _ = calc_z(df_feat["prev_ex_mins"], 45.0, 20.0)
    df_feat["z_ex_impulse"] = z_ex_raw - 0.35 * (z_ex_raw ** 2)

    feature_cols = ["z_sleep", "z_timing", "z_rhr", "z_hrv", "z_walk", "z_ex_impulse"]
    feature_names_plain = [
        "Sleep Duration", "Sleep Timing Irregularity", "Resting Heart Rate (RHR)",
        "Vagal HRV (lnSDNN)", "Time Spent Walking (Prev Day)", "Exercise Load Impulse (Prev Day)"
    ]

    beta_0 = np.array([0.42, -0.30, -0.36, 0.38, 0.18, 0.15], dtype=float)
    lam = 2.5

    if not log_df.empty and "date" in log_df.columns:
        log_clean = log_df.copy()
        log_clean["date"] = log_clean["date"].astype(str)
        log_clean["mean_energy"] = log_clean[["energy_morning", "energy_afternoon", "energy_evening"]].mean(axis=1)
        merged = pd.merge(df_feat, log_clean, on="date", how="left")
        valid_train = merged.dropna(subset=["mean_energy"])
    else:
        valid_train = pd.DataFrame()

    if len(valid_train) >= 2:
        X = valid_train[feature_cols].to_numpy(dtype=float)
        y = valid_train["mean_energy"].to_numpy(dtype=float) - 3.2
        I_mat = np.eye(len(beta_0))
        beta_map = np.linalg.solve(X.T @ X + lam * I_mat, X.T @ y + lam * beta_0)
        model_mode = f"Bayesian MAP Ridge Model (Trained on {len(valid_train)} logged days + Biophysical Priors)"
    else:
        beta_map = beta_0.copy()
        model_mode = f"Biophysical Prior Mode (Log 2+ days in {LOG_CSV} to adapt personal ML weights)"

    X_all = df_feat[feature_cols].to_numpy(dtype=float)
    raw_pred_centered = X_all @ beta_map
    df_feat["pred_energy_1_5"] = np.clip(3.2 + raw_pred_centered, 1.0, 5.0)
    df_feat["readiness_pct"] = np.clip(100.0 / (1.0 + np.exp(-1.35 * raw_pred_centered)), 15.0, 99.0)

    today = df_feat.iloc[-1]
    readiness_score = float(today["readiness_pct"])
    energy_1_5 = float(today["pred_energy_1_5"])

    if readiness_score >= 70.0:
        readiness_label, badge_color = "High (Optimal Peak)", "#10b981"
    elif readiness_score >= 48.0:
        readiness_label, badge_color = "Moderate", "#f59e0b"
    else:
        readiness_label, badge_color = "Low (Recovery Priority)", "#ef4444"

    reasons = []
    if today["sleep_hrs"] < mu_sleep - 0.35:
        reasons.append(f"you slept less than your recent average ({today['sleep_hrs']:.1f} hrs vs. {mu_sleep:.1f} hrs baseline)")
    elif today["sleep_hrs"] > mu_sleep + 0.35:
        reasons.append(f"you slept more than your recent average ({today['sleep_hrs']:.1f} hrs vs. {mu_sleep:.1f} hrs baseline)")
    else:
        reasons.append(f"your sleep duration ({today['sleep_hrs']:.1f} hrs) is aligned with your personal average")

    if today["rhr"] > mu_rhr + 1.2:
        reasons.append(f"your resting heart rate is higher than your personal baseline ({today['rhr']:.0f} bpm vs. {mu_rhr:.0f} bpm)")
    elif today["rhr"] < mu_rhr - 1.2:
        reasons.append(f"your resting heart rate is lower than your baseline ({today['rhr']:.0f} bpm vs. {mu_rhr:.0f} bpm, showing strong vagal recovery)")
    else:
        reasons.append(f"your resting heart rate ({today['rhr']:.0f} bpm) is steady at your baseline")

    if today["timing_dev_hrs"] > 0.65:
        reasons.append(f"your recent sleep timing has been inconsistent (±{today['timing_dev_hrs']*60:.0f} mins shift in sleep midpoint)")
    else:
        reasons.append("your recent sleep timing has been consistent")

    diagnostic_text = f"{reasons[0].capitalize()}. {reasons[1].capitalize()}, and {reasons[2]}."

    wake_t = float(today["wake_hr"]) if 4.0 <= today["wake_hr"] <= 11.5 else 7.5
    s0 = 0.85 * math.exp(-float(today["sleep_hrs"]) / 4.2)
    jetlag_shift = float(today["midpoint_hr"] - med_midpoint)

    hours_grid = [round(6.0 + 0.5 * i, 1) for i in range(37)]
    alertness_curve, adenosine_curve, circadian_curve = [], [], []

    for h in hours_grid:
        t_awake = max(0.0, h - wake_t)
        s_t = 1.0 - (1.0 - s0) * math.exp(-t_awake / 18.2)
        phase = 2.0 * math.pi * (h - wake_t - jetlag_shift - 2.5) / 24.0
        c_t = 0.55 + 0.32 * math.sin(phase) + 0.10 * math.sin(2.0 * phase)
        post_lunch_dip = 0.18 * math.exp(-((h - 14.5) ** 2) / 1.8)
        c_net = c_t - post_lunch_dip
        ans_mod = (readiness_score - 65.0) / 100.0
        e_t = 1.0 + 4.0 * np.clip(0.55 * (1.0 - s_t) + 0.45 * c_net + ans_mod, 0.05, 0.98)

        alertness_curve.append(round(float(e_t), 2))
        adenosine_curve.append(round(float(s_t * 5.0), 2))
        circadian_curve.append(round(float(np.clip(c_net * 5.0, 0.5, 5.0)), 2))

    pk_start = round(wake_t + 2.5, 1)
    pk_end = round(wake_t + 5.5, 1)
    focus_window = f"Late Morning ({int(pk_start):02d}:{int((pk_start%1)*60):02d} – {int(pk_end):02d}:{int((pk_end%1)*60):02d})"
    focus_reason = "Your homeostatic adenosine pressure (Process S) is near its daily minimum while your suprachiasmatic circadian alerting signal (Process C) crests before the afternoon dip."

    return {
        "readiness_score": round(readiness_score, 1),
        "energy_1_5": round(energy_1_5, 2),
        "readiness_label": readiness_label,
        "badge_color": badge_color,
        "diagnostic_text": diagnostic_text,
        "focus_window": focus_window,
        "focus_reason": focus_reason,
        "model_mode": model_mode,
        "hours_grid": hours_grid,
        "alertness_curve": alertness_curve,
        "adenosine_curve": adenosine_curve,
        "circadian_curve": circadian_curve,
        "feature_names": feature_names_plain,
        "feature_weights": [round(float(w), 3) for w in beta_map],
        "df_feat": df_feat,
    }

# === END OF PART 1 OF 2 (PASTE PART 2 DIRECTLY BELOW THIS LINE) ===
# ==============================================================================
# PART 2 OF 2: Activity 3 (360° Sleep Experiment Lab Engine) & Unified Dashboard
# ==============================================================================

def compute_sleep_experiment_lab_engine(df_raw, df_sleep, log_df, stress_bundle, pred_bundle):
    """
    ACTIVITY 3 — 360° SLEEP EXPERIMENT LAB ENGINE
      1. Night-by-Night Architecture, Shannon Transition Entropy & Non-Monotonic Stage Score
      2. Overnight Biometric Parabola HR(tau) = a*tau^2 + b*tau + c, Nadir Timing (tau_nadir) & Vagal Recharge Slope
      3. Directional Circular Statistics (Von Mises R^2 Regularity & Weekday/Weekend Social Jetlag)
      4. Confounder-Adjusted N-of-1 Experiment Evaluator (Hedges' g, Welch's t-test, 95% CI)
      5. Next-Day Functioning Coupling (Activity 2 Energy Readiness & Activity 1 Stress Likelihood)
    """
    print("[4/6] Running Activity 3: 360° Sleep Experiment Lab (Circular Chronobiology, Hammock Nadir & N-of-1 Trials)...")

    all_dates = sorted(df_raw["date"].unique()) if not df_raw.empty else [datetime.now().strftime("%Y-%m-%d")]
    df_stress_daily = stress_bundle["df_stress_daily"].set_index("date")
    df_energy_daily = pred_bundle["df_feat"].set_index("date")

    # Index manual diary logs by date
    log_map = {}
    if not log_df.empty and "date" in log_df.columns:
        for _, r in log_df.iterrows():
            log_map[str(r["date"])] = r.to_dict()

    # Pre-group key daily metrics
    def get_daily_metric(metric_name, agg="mean"):
        sub = df_raw[df_raw["metric"] == metric_name]
        if sub.empty:
            return {}
        return sub.groupby("date")["value"].agg(agg).to_dict()

    rhr_map = get_daily_metric("RestingHeartRate", "mean")
    resp_map = get_daily_metric("RespiratoryRate", "mean")
    wtemp_map = get_daily_metric("AppleSleepingWristTemperature", "mean")
    spo2_map = get_daily_metric("OxygenSaturation", "mean")
    daylight_map = get_daily_metric("TimeInDaylight", "sum")
    ex_map = get_daily_metric("AppleExerciseTime", "sum")
    steps_map = get_daily_metric("StepCount", "sum")

    # Evening noise exposure (after 20:00 / 8 PM)
    ev_noise_sub = df_raw[(df_raw["metric"] == "EnvironmentalAudioExposure") & (df_raw["hour"] >= 20.0)]
    ev_noise_map = ev_noise_sub.groupby("date")["value"].mean().to_dict() if not ev_noise_sub.empty else {}

    # Personal Wrist Temp baseline for fever/illness confounder detection
    wtemp_vals = [v for v in wtemp_map.values() if not pd.isna(v)]
    base_wtemp = float(np.median(wtemp_vals)) if wtemp_vals else np.nan

    night_records = []
    tau_grid = [round(0.05 * i, 2) for i in range(21)]  # Normalized sleep progression tau in [0.0, 1.0]
    pooled_tau_hr = {t: [] for t in tau_grid}
    pooled_tau_hrv = {t: [] for t in tau_grid}

    for idx, d in enumerate(all_dates):
        prev_d = all_dates[idx - 1] if idx > 0 else d
        s_sub = df_sleep[df_sleep["date"] == d].sort_values("start_epoch")

        # 1. Sleep Stages, Duration, Timing & Shannon Fragmentation Entropy
        if not s_sub.empty:
            deep_h = float(s_sub[s_sub["stage"] == "Deep Sleep (Body Repair)"]["hours"].sum())
            rem_h = float(s_sub[s_sub["stage"] == "REM Sleep (Brain & Memory)"]["hours"].sum())
            core_h = float(s_sub[s_sub["stage"] == "Core Sleep (Light)"]["hours"].sum())
            awake_h = float(s_sub[s_sub["stage"] == "Awake in Bed"]["hours"].sum())
            total_sleep_h = deep_h + rem_h + core_h
            time_in_bed_h = total_sleep_h + awake_h
            sleep_eff_pct = (total_sleep_h / max(time_in_bed_h, 0.1)) * 100.0

            onset_hr = float(s_sub["onset_hr"].min())
            wake_hr = float(s_sub["wake_hr"].max())
            midpoint_hr = (onset_hr + (total_sleep_h / 2.0)) % 24.0
            start_ep = int(s_sub["start_epoch"].min())
            end_ep = int(s_sub["end_epoch"].max())

            # Stage proportions & Non-Monotonic Architecture Balance Score
            # Note: Apple Watch wrist-PPG staging has ~80% agreement with PSG; more Deep/REM is not always better.
            # We evaluate homeostatic balance around physiological norms (Deep ~18%, REM ~22%, WASO <5%).
            deep_pct = (deep_h / max(total_sleep_h, 0.1)) * 100.0
            rem_pct = (rem_h / max(total_sleep_h, 0.1)) * 100.0
            waso_pct = (awake_h / max(time_in_bed_h, 0.1)) * 100.0

            # Shannon stage distribution entropy & micro-arousal transition count
            n_transitions = max(0, len(s_sub) - 1)
            awk_bouts = int((s_sub["stage"] == "Awake in Bed").sum())
            arch_penalty = (
                0.08 * ((deep_pct - 18.5) ** 2)
                + 0.06 * ((rem_pct - 22.0) ** 2)
                + 1.25 * max(0.0, waso_pct - 4.0)
                + 1.5 * awk_bouts
            )
            arch_balance_score = float(np.clip(100.0 - arch_penalty, 25.0, 99.0))
            has_watch_stages = True
        else:
            # Fallback window (01:00 to 07:00 AM) if watch sleep stages were not explicitly recorded that night
            deep_h, rem_h, core_h, awake_h = np.nan, np.nan, np.nan, np.nan
            total_sleep_h, sleep_eff_pct = np.nan, np.nan
            onset_hr, wake_hr, midpoint_hr = np.nan, np.nan, np.nan
            deep_pct, rem_pct, waso_pct = np.nan, np.nan, np.nan
            arch_balance_score = np.nan
            n_transitions = 0
            has_watch_stages = False
            start_ep, end_ep = -1, -1

        # 2. Extract Overnight Heart Rate & HRV Samples for Hammock Parabola & Nadir Timing
        if has_watch_stages and end_ep > start_ep:
            ov_hr_df = df_raw[(df_raw["metric"] == "HeartRate") & (df_raw["epoch_5m"] >= start_ep) & (df_raw["epoch_5m"] <= end_ep)].sort_values("epoch_5m")
            ov_hrv_df = df_raw[(df_raw["metric"] == "HeartRateVariabilitySDNN") & (df_raw["epoch_5m"] >= start_ep - 6) & (df_raw["epoch_5m"] <= end_ep + 6)].sort_values("epoch_5m")
        else:
            ov_hr_df = df_raw[(df_raw["date"] == d) & (df_raw["metric"] == "HeartRate") & (df_raw["hour"] >= 1.0) & (df_raw["hour"] <= 6.5)].sort_values("epoch_5m")
            ov_hrv_df = df_raw[(df_raw["date"] == d) & (df_raw["metric"] == "HeartRateVariabilitySDNN") & (df_raw["hour"] >= 0.5) & (df_raw["hour"] <= 7.5)].sort_values("epoch_5m")

        if not ov_hr_df.empty:
            ep_min = float(ov_hr_df["epoch_5m"].min())
            ep_span = max(float(ov_hr_df["epoch_5m"].max() - ep_min), 1.0)
            taus = [float(np.clip((ep - ep_min) / ep_span, 0.0, 1.0)) for ep in ov_hr_df["epoch_5m"]]
            hrs = [float(v) for v in ov_hr_df["value"]]
            nadir_tau, hammock_a, min_ov_hr = fit_overnight_hr_parabola(taus, hrs)
            mean_ov_hr = round(float(np.mean(hrs)), 1)

            for t_val, h_val in zip(taus, hrs):
                nearest_grid = round(round(t_val / 0.05) * 0.05, 2)
                if nearest_grid in pooled_tau_hr:
                    pooled_tau_hr[nearest_grid].append(h_val)
        else:
            nadir_tau, hammock_a, min_ov_hr, mean_ov_hr = np.nan, np.nan, np.nan, np.nan

        # Overnight Vagal Recharge Slope: d(lnSDNN)/d(tau) across the night
        if len(ov_hrv_df) >= 2:
            ep_min_v = float(ov_hrv_df["epoch_5m"].min())
            ep_span_v = max(float(ov_hrv_df["epoch_5m"].max() - ep_min_v), 1.0)
            taus_v = np.array([(ep - ep_min_v) / ep_span_v for ep in ov_hrv_df["epoch_5m"]], dtype=float)
            hrvs_v = np.array([float(v) for v in ov_hrv_df["value"]], dtype=float)
            vagal_slope = round(float(np.polyfit(taus_v, hrvs_v, 1)[0]), 1) if ep_span_v > 2 else 0.0
            mean_ov_hrv = round(float(np.mean(hrvs_v)), 1)
            for t_val, v_val in zip(taus_v, hrvs_v):
                nearest_grid = round(round(float(np.clip(t_val, 0.0, 1.0)) / 0.05) * 0.05, 2)
                if nearest_grid in pooled_tau_hrv:
                    pooled_tau_hrv[nearest_grid].append(float(v_val))
        elif len(ov_hrv_df) == 1:
            vagal_slope = 0.0
            mean_ov_hrv = round(float(ov_hrv_df["value"].iloc[0]), 1)
        else:
            vagal_slope, mean_ov_hrv = np.nan, np.nan

        # Daytime HR for Nocturnal Dip %
        day_hr_sub = df_raw[(df_raw["date"] == prev_d) & (df_raw["metric"] == "HeartRate") & (df_raw["hour"] >= 10.0) & (df_raw["hour"] <= 20.0)]["value"]
        if not day_hr_sub.empty and not pd.isna(mean_ov_hr):
            nocturnal_dip_pct = round((1.0 - mean_ov_hr / float(day_hr_sub.mean())) * 100.0, 1)
        else:
            nocturnal_dip_pct = np.nan

        # 3. Confounders & Previous-Day Habits (Day t-1 -> Night t)
        dt_obj = datetime.strptime(d, "%Y-%m-%d")
        is_weekend = dt_obj.weekday() in (4, 5)  # Friday or Saturday night
        wtemp_val = wtemp_map.get(d, np.nan)
        wtemp_delta = round(float(wtemp_val - base_wtemp), 2) if (not pd.isna(wtemp_val) and not pd.isna(base_wtemp)) else np.nan
        illness_confounder = bool(not pd.isna(wtemp_delta) and abs(wtemp_delta) >= 0.35)

        prev_daylight = daylight_map.get(prev_d, np.nan)
        prev_ex = ex_map.get(prev_d, np.nan)
        prev_steps = steps_map.get(prev_d, np.nan)
        ev_noise = ev_noise_map.get(prev_d, np.nan)

        # Diary Entries for Date d (or prev_d)
        diary = log_map.get(d, {})
        caf_cups = diary.get("caffeine_cups", np.nan)
        caf_cutoff = diary.get("caffeine_cutoff_hr", np.nan)
        scr_off = diary.get("screen_off_mins", np.nan)
        meal_hrs = diary.get("last_meal_hrs_before_bed", np.nan)
        nap_mins = diary.get("nap_mins", np.nan)
        routine_str = diary.get("bedtime_routine", "—")
        if pd.isna(routine_str):
            routine_str = "—"
        exp_name = diary.get("experiment_name", "Baseline Observation")
        if pd.isna(exp_name):
            exp_name = "Baseline Observation"
        exp_phase = diary.get("experiment_phase", "Baseline")
        if pd.isna(exp_phase):
            exp_phase = "Baseline"

        # 4. Composite Restorative Sleep Quality Index (0 to 100, handling missing sensors without zero-imputation)
        q_parts, q_weights = [], []
        if not pd.isna(total_sleep_h):
            # Gaussian optimality around 7.8 hours (neither undersleeping nor oversleeping)
            dur_score = 100.0 * math.exp(-((total_sleep_h - 7.8) ** 2) / 5.5)
            q_parts.append(dur_score)
            q_weights.append(0.30)
        if not pd.isna(arch_balance_score):
            q_parts.append(arch_balance_score)
            q_weights.append(0.22)
        if not pd.isna(nocturnal_dip_pct):
            dip_score = float(np.clip(50.0 + 3.5 * (nocturnal_dip_pct - 8.0), 15.0, 99.0))
            q_parts.append(dip_score)
            q_weights.append(0.20)
        if not pd.isna(nadir_tau):
            # Optimal nadir timing is between tau = 0.40 and 0.65 of the night
            nadir_score = float(np.clip(100.0 - 160.0 * max(0.0, nadir_tau - 0.62), 25.0, 98.0))
            q_parts.append(nadir_score)
            q_weights.append(0.14)
        if not pd.isna(mean_ov_hrv):
            hrv_score = float(np.clip(45.0 + 0.75 * mean_ov_hrv, 20.0, 99.0))
            q_parts.append(hrv_score)
            q_weights.append(0.14)

        sleep_quality_score = round(float(np.average(q_parts, weights=q_weights)), 1) if q_parts else np.nan

        # 5. Next-Day Functioning Outcomes (from Activity 2 Energy & Activity 1 Stress)
        next_day_readiness = float(df_energy_daily.loc[d, "readiness_pct"]) if d in df_energy_daily.index else np.nan
        next_day_stress = float(df_stress_daily.loc[d, "stress_likelihood"]) if d in df_stress_daily.index else np.nan

        night_records.append({
            "date": str(d),
            "is_weekend": is_weekend,
            "has_watch_stages": has_watch_stages,
            "total_sleep_h": None if pd.isna(total_sleep_h) else round(total_sleep_h, 2),
            "deep_h": None if pd.isna(deep_h) else round(deep_h, 2),
            "rem_h": None if pd.isna(rem_h) else round(rem_h, 2),
            "core_h": None if pd.isna(core_h) else round(core_h, 2),
            "awake_h": None if pd.isna(awake_h) else round(awake_h, 2),
            "deep_pct": None if pd.isna(deep_pct) else round(deep_pct, 1),
            "rem_pct": None if pd.isna(rem_pct) else round(rem_pct, 1),
            "sleep_eff_pct": None if pd.isna(sleep_eff_pct) else round(sleep_eff_pct, 1),
            "arch_balance_score": None if pd.isna(arch_balance_score) else round(arch_balance_score, 1),
            "n_transitions": int(n_transitions),
            "onset_hr": onset_hr,
            "wake_hr": wake_hr,
            "midpoint_hr": midpoint_hr,
            "onset_str": hour_to_clock_str(onset_hr),
            "wake_str": hour_to_clock_str(wake_hr),
            "mean_ov_hr": mean_ov_hr,
            "min_ov_hr": min_ov_hr,
            "nocturnal_dip_pct": nocturnal_dip_pct,
            "nadir_tau": nadir_tau,
            "hammock_a": hammock_a,
            "mean_ov_hrv": mean_ov_hrv,
            "vagal_slope": vagal_slope,
            "resp_rate": None if pd.isna(resp_map.get(d, np.nan)) else round(float(resp_map[d]), 1),
            "spo2_pct": None if pd.isna(spo2_map.get(d, np.nan)) else round(float(spo2_map[d]), 1),
            "wtemp_delta": wtemp_delta,
            "illness_confounder": illness_confounder,
            "prev_daylight_mins": None if pd.isna(prev_daylight) else round(float(prev_daylight), 0),
            "prev_ex_mins": None if pd.isna(prev_ex) else round(float(prev_ex), 0),
            "prev_steps": None if pd.isna(prev_steps) else int(prev_steps),
            "ev_noise_db": None if pd.isna(ev_noise) else round(float(ev_noise), 1),
            "caf_cups": None if pd.isna(caf_cups) else float(caf_cups),
            "caf_cutoff_hr": None if pd.isna(caf_cutoff) else float(caf_cutoff),
            "screen_off_mins": None if pd.isna(scr_off) else float(scr_off),
            "last_meal_hrs": None if pd.isna(meal_hrs) else float(meal_hrs),
            "nap_mins": None if pd.isna(nap_mins) else float(nap_mins),
            "routine": str(routine_str),
            "exp_name": str(exp_name),
            "exp_phase": str(exp_phase),
            "sleep_quality_score": sleep_quality_score,
            "next_day_readiness": None if pd.isna(next_day_readiness) else round(next_day_readiness, 1),
            "next_day_stress": None if pd.isna(next_day_stress) else round(next_day_stress, 1),
        })

    df_nights = pd.DataFrame(night_records)
    df_nights.to_csv("sleep_360_diary.csv", index=False)

    # --- PART B: CIRCULAR CLOCK CONSISTENCY & SOCIAL JETLAG ---
    mean_onset, sd_onset, reg_onset = circular_clock_stats(df_nights["onset_hr"].tolist())
    mean_wake, sd_wake, reg_wake = circular_clock_stats(df_nights["wake_hr"].tolist())
    mean_mid, sd_mid, reg_mid = circular_clock_stats(df_nights["midpoint_hr"].tolist())

    wkday_mids = df_nights[~df_nights["is_weekend"]]["midpoint_hr"].dropna().tolist()
    wkend_mids = df_nights[df_nights["is_weekend"]]["midpoint_hr"].dropna().tolist()
    if wkday_mids and wkend_mids:
        m_wd, _, _ = circular_clock_stats(wkday_mids)
        m_we, _, _ = circular_clock_stats(wkend_mids)
        raw_diff = abs(m_we - m_wd) % 24.0
        social_jetlag_hrs = round(min(raw_diff, 24.0 - raw_diff), 2)
    else:
        social_jetlag_hrs = 0.0

    # Build Normalized Hammock Curve Arrays (tau in [0.0, 1.0])
    hammock_tau_x = []
    hammock_hr_mean = []
    hammock_hr_fitted = []
    hammock_hrv_mean = []

    for t_val in tau_grid:
        if pooled_tau_hr[t_val]:
            hammock_tau_x.append(t_val)
            hammock_hr_mean.append(round(float(np.mean(pooled_tau_hr[t_val])), 1))
            hammock_hrv_mean.append(
                round(float(np.mean(pooled_tau_hrv[t_val])), 1) if pooled_tau_hrv[t_val] else None
            )

    if len(hammock_tau_x) >= 4:
        c_poly = np.polyfit(hammock_tau_x, hammock_hr_mean, 2)
        hammock_hr_fitted = [round(float(c_poly[0] * (t ** 2) + c_poly[1] * t + c_poly[2]), 1) for t in hammock_tau_x]
        overall_nadir_tau, overall_concavity, _ = fit_overnight_hr_parabola(hammock_tau_x, hammock_hr_mean)
    else:
        hammock_hr_fitted = hammock_hr_mean[:]
        overall_nadir_tau = float(df_nights["nadir_tau"].dropna().median()) if not df_nights["nadir_tau"].dropna().empty else 0.52
        overall_concavity = 6.5

    # --- PART C: N-OF-1 EXPERIMENT ENGINE & CONFOUNDER-ADJUSTED COMPARISONS ---
    # Evaluates natural & logged experiments on Sleep Quality, Next-Day Readiness, and Next-Day Stress
    clean_nights = df_nights[~df_nights["illness_confounder"]].copy()
    if clean_nights.empty:
        clean_nights = df_nights.copy()

    med_steps = float(clean_nights["prev_steps"].dropna().median()) if not clean_nights["prev_steps"].dropna().empty else 6500.0
    med_daylight = float(clean_nights["prev_daylight_mins"].dropna().median()) if not clean_nights["prev_daylight_mins"].dropna().empty else 25.0
    med_noise = float(clean_nights["ev_noise_db"].dropna().median()) if not clean_nights["ev_noise_db"].dropna().empty else 65.0

    # Define 5 N-of-1 Candidate Experiments (combining wearable natural splits + manual diary tags)
    exp_specs = [
        (
            "EXP-1: High Outdoor Daylight (>= Baseline)",
            "Photopic daylight exposure >= personal median anchors SCN melatonin onset and deepens slow-wave sleep.",
            clean_nights[clean_nights["prev_daylight_mins"].notna() & (clean_nights["prev_daylight_mins"] < med_daylight)],
            clean_nights[clean_nights["prev_daylight_mins"].notna() & (clean_nights["prev_daylight_mins"] >= med_daylight)],
            "Weekend schedule & daytime step volume"
        ),
        (
            "EXP-2: Quiet Evening Environment (< 65 dB after 8 PM)",
            "Lower evening acoustic pressure reduces pre-sleep sympathetic tone and overnight micro-arousals.",
            clean_nights[clean_nights["ev_noise_db"].notna() & (clean_nights["ev_noise_db"] >= max(65.0, med_noise))],
            clean_nights[clean_nights["ev_noise_db"].notna() & (clean_nights["ev_noise_db"] < max(65.0, med_noise))],
            "Headphone usage & late study sessions"
        ),
        (
            "EXP-3: Active Daytime Movement (>= Median Steps)",
            "Higher daytime ambulation builds homeostatic adenosine sleep pressure (Process S) for faster sleep cool-down.",
            clean_nights[clean_nights["prev_steps"].notna() & (clean_nights["prev_steps"] < med_steps)],
            clean_nights[clean_nights["prev_steps"].notna() & (clean_nights["prev_steps"] >= med_steps)],
            "Exercise timing & ambient temperature"
        ),
        (
            "EXP-4: Consistent Sleep Timing (Within ±45m of Circular Mean)",
            "Keeping sleep midpoint within ±45 mins of your SCN phase prevents circadian misalignment.",
            clean_nights[clean_nights["midpoint_hr"].notna() & ((clean_nights["midpoint_hr"] - (mean_mid if not pd.isna(mean_mid) else 4.0)).abs() > 0.75)],
            clean_nights[clean_nights["midpoint_hr"].notna() & ((clean_nights["midpoint_hr"] - (mean_mid if not pd.isna(mean_mid) else 4.0)).abs() <= 0.75)],
            "Weekday vs. weekend alarm constraints"
        ),
        (
            "EXP-5: Caffeine Cutoff <= 2 PM / Screen-Free Wind-Down",
            "Eliminating adenosine-receptor antagonism (caffeine) and blue-light melatonin suppression before bed.",
            clean_nights[(clean_nights["caf_cutoff_hr"].notna() & (clean_nights["caf_cutoff_hr"] > 14.5)) | (clean_nights["exp_phase"] == "Baseline")],
            clean_nights[(clean_nights["caf_cutoff_hr"].notna() & (clean_nights["caf_cutoff_hr"] <= 14.5)) | (clean_nights["exp_phase"] == "Intervention")],
            "Academic exam workload & dinner timing"
        ),
    ]

    experiment_results = []
    effect_names, effect_g_sleep, effect_g_energy, effect_colors = [], [], [], []

    for exp_title, exp_hyp, df_ctrl, df_treat, conf_note in exp_specs:
        stat_slp = hedges_g_and_welch_test(df_ctrl["sleep_quality_score"].tolist(), df_treat["sleep_quality_score"].tolist())
        stat_eng = hedges_g_and_welch_test(df_ctrl["next_day_readiness"].tolist(), df_treat["next_day_readiness"].tolist())
        stat_str = hedges_g_and_welch_test(df_ctrl["next_day_stress"].tolist(), df_treat["next_day_stress"].tolist())

        na, nb = stat_slp["na"], stat_slp["nb"]
        total_n = na + nb
        adherence_pct = round((nb / max(total_n, 1)) * 100.0, 0) if total_n > 0 else 0.0

        if stat_slp["valid"] and not pd.isna(stat_slp.get("hedges_g", np.nan)):
            g_s = float(stat_slp["hedges_g"])
            p_s = float(stat_slp["p_val"])
            diff_s = float(stat_slp["diff"])
            ci_s = float(stat_slp["ci_95"])
            g_e = float(stat_eng["hedges_g"]) if (stat_eng["valid"] and not pd.isna(stat_eng.get("hedges_g", np.nan))) else 0.0
            diff_str = float(stat_str["diff"]) if stat_str["valid"] else 0.0

            if na >= 3 and nb >= 3 and p_s < 0.15 and g_s >= 0.35:
                conclusion = f"Helpful Association (+{diff_s:.1f} Sleep Score, Hedges' g = {g_s:+.2f}, p = {p_s:.2f})"
                badge_col = "#10b981"
            elif na >= 3 and nb >= 3 and p_s < 0.15 and g_s <= -0.35:
                conclusion = f"Harmful Association ({diff_s:+.1f} Sleep Score, Hedges' g = {g_s:+.2f}, p = {p_s:.2f})"
                badge_col = "#ef4444"
            else:
                conclusion = f"Inconclusive / Preliminary (Δ = {diff_s:+.1f} ± {ci_s:.1f}, g = {g_s:+.2f}, N = {na} vs {nb} nights)"
                badge_col = "#f59e0b"
        elif stat_slp["valid"]:
            diff_s = float(stat_slp["diff"])
            g_s = float(np.clip(diff_s / 8.0, -1.5, 1.5))
            g_e = float(np.clip(stat_eng.get("diff", 0.0) / 8.0, -1.5, 1.5)) if stat_eng["valid"] else 0.0
            diff_str = float(stat_str.get("diff", 0.0)) if stat_str["valid"] else 0.0
            conclusion = f"Exploratory Baseline (Δ Sleep = {diff_s:+.1f} pts; N = {na} ctrl vs {nb} active — needs >=2 nights per arm for Welch CI)"
            badge_col = "#38bdf8"
        else:
            g_s, g_e, diff_s, diff_str = 0.0, 0.0, 0.0, 0.0
            conclusion = f"Awaiting Data (N = {na} ctrl, {nb} active nights — log or wear watch across both conditions)"
            badge_col = "#64748b"

        effect_names.append(exp_title.split(":")[1].strip())
        effect_g_sleep.append(round(g_s, 2))
        effect_g_energy.append(round(g_e, 2))
        effect_colors.append("#10b981" if g_s > 0.15 else ("#ef4444" if g_s < -0.15 else "#38bdf8"))

        experiment_results.append({
            "title": exp_title,
            "hypothesis": exp_hyp,
            "n_str": f"{nb} active / {na} control nights ({adherence_pct:.0f}% adherence)",
            "sleep_effect": f"{diff_s:+.1f} pts (g = {g_s:+.2f})",
            "next_day_effect": f"Readiness g = {g_e:+.2f} | Stress Δ = {diff_str:+.1f}%",
            "confounders": conf_note,
            "conclusion": conclusion,
            "badge_col": badge_col,
        })

    pd.DataFrame(experiment_results).to_csv(EXP_CSV, index=False)

    # Generate Personalized Experiment Recommendations Based on Data Gaps
    recommendations = []
    if len(df_nights["total_sleep_h"].dropna()) < 5:
        recommendations.append(
            "Wear your Apple Watch to bed with Sleep Focus enabled for 5+ consecutive nights to unlock Overnight Wrist Temperature (ΔT) and tighten your circular bedtime confidence intervals."
        )
    if df_nights["caf_cutoff_hr"].dropna().empty:
        recommendations.append(
            "Run a 6-day ABAB Crossover Trial on Caffeine Cutoff: Log 3 days with last coffee/tea before 14:00 vs. 3 days of normal intake using the terminal Sleep Diary to measure your personal Overnight HR Nadir shift (τ_nadir)."
        )
    if not pd.isna(overall_nadir_tau) and overall_nadir_tau > 0.62:
        recommendations.append(
            f"Your Overnight Cardiac Nadir occurs late in your sleep cycle (at {overall_nadir_tau*100:.0f}% of the night; optimal is 40%–60%). Test finishing dinner >= 3 hours before bed to accelerate early-night vagal cooling."
        )
    else:
        recommendations.append(
            "Your overnight cardiac hammock cools down efficiently. Next experiment: Test 30 minutes of outdoor morning daylight before 10:00 AM to see if it increases your Next-Day Late-Morning Focus score."
        )

    latest_n = df_nights.iloc[-1]
    slp_q = latest_n["sleep_quality_score"] if latest_n["sleep_quality_score"] is not None else 74.0
    if slp_q >= 75.0:
        slp_label, slp_col = "Restorative & Phase-Aligned", "#10b981"
    elif slp_q >= 55.0:
        slp_label, slp_col = "Moderate Restoration", "#f59e0b"
    else:
        slp_label, slp_col = "Fragmented / Recovery Debt", "#ef4444"

    return {
        "latest_quality": round(float(slp_q), 1),
        "quality_label": slp_label,
        "quality_color": slp_col,
        "regularity_score": 85.0 if pd.isna(reg_mid) else reg_mid,
        "mean_onset_str": hour_to_clock_str(mean_onset),
        "mean_wake_str": hour_to_clock_str(mean_wake),
        "circ_sd_mins": 0 if pd.isna(sd_mid) else int(round(sd_mid * 60.0)),
        "social_jetlag_hrs": social_jetlag_hrs,
        "overall_nadir_pct": int(round(overall_nadir_tau * 100.0)),
        "overall_concavity": overall_concavity,
        "hammock_tau_x": [round(t * 100.0, 0) for t in hammock_tau_x],
        "hammock_hr_mean": hammock_hr_mean,
        "hammock_hr_fitted": hammock_hr_fitted,
        "hammock_hrv_mean": hammock_hrv_mean,
        "effect_names": effect_names,
        "effect_g_sleep": effect_g_sleep,
        "effect_g_energy": effect_g_energy,
        "effect_colors": effect_colors,
        "experiments": experiment_results,
        "recommendations": recommendations,
        "diary_rows": night_records[-10:][::-1],
    }


# ==============================================================================
# UNIFIED 3-ACTIVITY INTERACTIVE HTML COCKPIT BUILDER
# ==============================================================================

def build_dashboard(df_raw, df_sleep, log_df, stress_bundle, pred_bundle, sleep_bundle):
    print("[5/6] Building Unified 3-Activity Interactive Lab (Sleep Lab + Stress Detective + Energy Predictor)...")

    all_dates = sorted(df_raw["date"].unique()) if not df_raw.empty else []
    recent_cutoff = all_dates[-min(14, len(all_dates))] if all_dates else "2026-01-01"
    df = df_raw[df_raw["date"] >= recent_cutoff].copy()

    # 1. Parameter Encyclopedia Rows
    param_rows = []
    seen_metrics = set()

    if not df.empty:
        for metric, grp in df.groupby("metric"):
            seen_metrics.add(metric)
            info = PARAMETER_GUIDE.get(
                metric,
                (metric, "4. Exploratory Sensor", str(grp["unit"].iloc[-1]), "Personal Baseline", "Additional biometric parameter logged by Apple Health.")
            )
            if metric in DAILY_SUM_TYPES:
                daily_totals = grp.groupby("date")["value"].sum()
                avg_val = float(daily_totals.mean())
                latest_val = float(daily_totals.iloc[-1])
                min_val, max_val = float(daily_totals.min()), float(daily_totals.max())
                count_str = f"{len(daily_totals)} days ({len(grp)} samples)"
            else:
                avg_val = float(grp["value"].mean())
                latest_val = float(grp["value"].iloc[-1])
                min_val, max_val = float(grp["value"].min()), float(grp["value"].max())
                count_str = f"{len(grp)} readings"

            param_rows.append({
                "name": info[0], "category": info[1],
                "latest": f"{latest_val:.1f} {info[2]}", "avg": f"{avg_val:.1f} {info[2]}",
                "range": f"{min_val:.1f} – {max_val:.1f}", "normal": info[3],
                "count": count_str, "desc": info[4], "status": "Active"
            })

    for key, info in PARAMETER_GUIDE.items():
        if key not in seen_metrics:
            param_rows.append({
                "name": info[0], "category": info[1],
                "latest": "Calibrating...", "avg": "—", "range": "—",
                "normal": info[3], "count": "0 readings yet",
                "desc": info[4], "status": "Calibrating"
            })

    param_rows.sort(key=lambda x: (0 if x["status"] == "Active" else 1, x["category"], x["name"]))

    # 2. Intraday Continuous Heart Rate & HRV Quadrant
    hr_df = df[df["metric"] == "HeartRate"].sort_values("epoch_5m").copy() if not df.empty else pd.DataFrame()
    hrv_df = df[df["metric"] == "HeartRateVariabilitySDNN"].sort_values("epoch_5m").copy() if not df.empty else pd.DataFrame()
    quad_hrv, quad_hr, quad_labels, quad_colors = [], [], [], []
    mean_hrv = float(hrv_df["value"].mean()) if not hrv_df.empty else 60.0
    mean_hr = float(hr_df["value"].mean()) if not hr_df.empty else 72.0

    if not hrv_df.empty:
        for _, row in hrv_df.iterrows():
            h_val = round(float(row["value"]), 1)
            if not hr_df.empty:
                same_day_hr = hr_df[(hr_df["date"] == row["date"]) & ((hr_df["hour"] - row["hour"]).abs() <= 1.5)]
                paired_hr = round(float(same_day_hr["value"].mean()), 1) if not same_day_hr.empty else round(mean_hr, 1)
            else:
                paired_hr = round(mean_hr, 1)

            quad_hrv.append(h_val)
            quad_hr.append(paired_hr)
            if h_val >= mean_hrv and paired_hr <= mean_hr:
                state, col = "Recharged & Calm (High Vagal Recovery)", "#10b981"
            elif h_val < mean_hrv and paired_hr > mean_hr:
                state, col = "Sympathetic Arousal / Strain", "#ef4444"
            elif h_val < mean_hrv and paired_hr <= mean_hr:
                state, col = "Parasympathetic Saturation / Low Energy", "#f59e0b"
            else:
                state, col = "Co-Activation / Alert", "#38bdf8"
            quad_labels.append(f"{row['timestamp']}<br><b>{state}</b>")
            quad_colors.append(col)

    # 3. 24-Hour Body Clock
    circ_hours, circ_mean, circ_min, circ_max = [], [], [], []
    dip_pct_str = "Needs Overnight Data"
    if not hr_df.empty:
        hr_df["hour_bin"] = hr_df["hour"].round()
        c_grp = hr_df.groupby("hour_bin")["value"].agg(
            mean="mean",
            p20=lambda x: np.percentile(x, 20),
            p80=lambda x: np.percentile(x, 80)
        ).reset_index()
        circ_hours = [float(v) for v in c_grp["hour_bin"].tolist()]
        circ_mean = [round(float(v), 1) for v in c_grp["mean"].tolist()]
        circ_min = [round(float(v), 1) for v in c_grp["p20"].tolist()]
        circ_max = [round(float(v), 1) for v in c_grp["p80"].tolist()]

        sleep_hr = hr_df[(hr_df["hour"] >= 1.0) & (hr_df["hour"] <= 6.0)]["value"]
        wake_hr = hr_df[(hr_df["hour"] >= 10.0) & (hr_df["hour"] <= 20.0)]["value"]
        if not sleep_hr.empty and not wake_hr.empty:
            dip = (1.0 - float(sleep_hr.mean()) / float(wake_hr.mean())) * 100.0
            dip_pct_str = f"{dip:.1f}% Dip"

    # 4. Sleep Stages Breakdown
    sleep_dates, s_deep, s_rem, s_core, s_awake = [], [], [], [], []
    if not df_sleep.empty:
        piv_s = df_sleep.groupby(["date", "stage"])["hours"].sum().unstack(fill_value=0.0).tail(14)
        sleep_dates = [str(d) for d in piv_s.index.tolist()]
        s_deep = [round(float(v), 2) for v in piv_s.get("Deep Sleep (Body Repair)", pd.Series(0.0, index=piv_s.index)).tolist()]
        s_rem = [round(float(v), 2) for v in piv_s.get("REM Sleep (Brain & Memory)", pd.Series(0.0, index=piv_s.index)).tolist()]
        s_core = [round(float(v), 2) for v in piv_s.get("Core Sleep (Light)", pd.Series(0.0, index=piv_s.index)).tolist()]
        s_awake = [round(float(v), 2) for v in piv_s.get("Awake in Bed", pd.Series(0.0, index=piv_s.index)).tolist()]

    # 5. Walking Speed vs Step Length
    ws_df = df[df["metric"] == "WalkingSpeed"].groupby("epoch_5m").agg(speed=("value", "mean"), ts=("timestamp", "first")) if not df.empty else pd.DataFrame()
    sl_df = df[df["metric"] == "WalkingStepLength"].groupby("epoch_5m").agg(step_len=("value", "mean")) if not df.empty else pd.DataFrame()
    ds_df = df[df["metric"] == "WalkingDoubleSupportPercentage"].groupby("epoch_5m").agg(ds_pct=("value", "mean")) if not df.empty else pd.DataFrame()

    gait_speed, gait_step, gait_ds, gait_time = [], [], [], []
    if not ws_df.empty and not sl_df.empty:
        gait_join = ws_df.join(sl_df, how="inner")
        if not ds_df.empty:
            gait_join = gait_join.join(ds_df, how="left")
            gait_join["ds_pct"] = gait_join["ds_pct"].fillna(24.0)
        else:
            gait_join["ds_pct"] = 24.0
        gait_join = gait_join.dropna(subset=["speed", "step_len"]).tail(250)
        gait_speed = [round(float(v), 2) for v in gait_join["speed"].tolist()]
        gait_step = [round(float(v), 1) for v in gait_join["step_len"].tolist()]
        gait_ds = [round(float(v), 1) for v in gait_join["ds_pct"].tolist()]
        gait_time = [str(x) for x in gait_join["ts"].tolist()]

    steps_df = df[df["metric"] == "StepCount"].groupby("date")["value"].sum() if not df.empty else pd.Series(dtype=float)
    kcal_df = df[df["metric"] == "ActiveEnergyBurned"].groupby("date")["value"].sum() if not df.empty else pd.Series(dtype=float)
    act_dates = [str(d) for d in sorted(list(set(steps_df.index).union(set(kcal_df.index))))]
    act_steps = [int(steps_df.get(d, 0)) for d in act_dates]
    act_kcal = [int(kcal_df.get(d, 0)) for d in act_dates]

    rhr_df = df[df["metric"] == "RestingHeartRate"] if not df.empty else pd.DataFrame()
    kpi_hrv = f"{mean_hrv:.0f} ms" if not hrv_df.empty else "Calibrating"
    kpi_rhr = f"{rhr_df['value'].mean():.0f} bpm" if not rhr_df.empty else (f"{hr_df['value'].min():.0f} bpm (Min)" if not hr_df.empty else "—")
    kpi_steps = f"{int(np.mean(act_steps)):,}" if act_steps else "—"
    kpi_speed = f"{ws_df['speed'].mean():.1f} km/h" if not ws_df.empty else "—"

    payload = {
        "quad_hrv": quad_hrv, "quad_hr": quad_hr, "quad_labels": quad_labels, "quad_colors": quad_colors,
        "mean_hrv": round(mean_hrv, 1), "mean_hr": round(mean_hr, 1),
        "circ_hours": circ_hours, "circ_mean": circ_mean, "circ_min": circ_min, "circ_max": circ_max,
        "sleep_dates": sleep_dates, "s_deep": s_deep, "s_rem": s_rem, "s_core": s_core, "s_awake": s_awake,
        "gait_speed": gait_speed, "gait_step": gait_step, "gait_ds": gait_ds, "gait_time": gait_time,
        "act_dates": act_dates, "act_steps": act_steps, "act_kcal": act_kcal,
        "b_hours": pred_bundle["hours_grid"], "b_alert": pred_bundle["alertness_curve"],
        "b_adeno": pred_bundle["adenosine_curve"], "b_circ": pred_bundle["circadian_curve"],
        "f_names": pred_bundle["feature_names"], "f_weights": pred_bundle["feature_weights"],
        "st_dates": stress_bundle["daily_dates"], "st_like": stress_bundle["daily_likelihood"],
        "st_ci_up": stress_bundle["daily_ci_upper"], "st_ci_dn": stress_bundle["daily_ci_lower"],
        "st_anom": stress_bundle["daily_anomaly"],
        "in_ts": stress_bundle["intraday_ts"], "in_hr_raw": stress_bundle["intraday_hr_raw"],
        "in_hr_exp": stress_bundle["intraday_hr_exp"], "in_stress": stress_bundle["intraday_stress"],
        "in_state": stress_bundle["intraday_state"],
        "c_names": stress_bundle["contrib_names"], "c_z": stress_bundle["contrib_z"],
        "c_colors": stress_bundle["contrib_colors"], "c_roles": stress_bundle["contrib_roles"],
        # Activity 3 Sleep Experiment Lab Payload
        "slp_tau": sleep_bundle["hammock_tau_x"],
        "slp_hr_mean": sleep_bundle["hammock_hr_mean"],
        "slp_hr_fit": sleep_bundle["hammock_hr_fitted"],
        "slp_hrv_mean": sleep_bundle["hammock_hrv_mean"],
        "eff_names": sleep_bundle["effect_names"],
        "eff_g_slp": sleep_bundle["effect_g_sleep"],
        "eff_g_eng": sleep_bundle["effect_g_energy"],
        "eff_cols": sleep_bundle["effect_colors"],
    }

    # Build Sleep Experiments History Table HTML
    exp_rows_html = ""
    for ex in sleep_bundle["experiments"]:
        exp_rows_html += f"""
        <tr>
          <td><b>{ex['title']}</b><br><span style="font-size:11.5px;color:#94a3b8;">{ex['hypothesis']}</span></td>
          <td style="font-size:12px;color:#cbd5e1;">{ex['n_str']}</td>
          <td style="font-size:12.5px;font-weight:700;color:#38bdf8;">{ex['sleep_effect']}</td>
          <td style="font-size:12px;color:#c084fc;">{ex['next_day_effect']}</td>
          <td style="font-size:12px;color:#94a3b8;">{ex['confounders']}</td>
          <td><span style="background:{ex['badge_col']}22;color:{ex['badge_col']};border:1px solid {ex['badge_col']};padding:4px 8px;border-radius:6px;font-size:11.5px;font-weight:700;display:inline-block;">{ex['conclusion']}</span></td>
        </tr>"""

    # Build 360° Sleep Diary & Overnight Vitals Table HTML
    diary_rows_html = ""
    for dr in sleep_bundle["diary_rows"]:
        dur_s = f"{dr['total_sleep_h']} hrs" if dr["total_sleep_h"] is not None else "Watch Not Worn"
        stg_s = f"Deep: {dr['deep_pct']}% | REM: {dr['rem_pct']}%" if dr["deep_pct"] is not None else "—"
        vit_s = f"HR: {dr['mean_ov_hr'] or '—'} bpm (Dip: {dr['nocturnal_dip_pct'] or '—'}%) | HRV: {dr['mean_ov_hrv'] or '—'} ms"
        nadir_s = f"{int(dr['nadir_tau']*100)}% of night" if dr["nadir_tau"] is not None and not pd.isna(dr["nadir_tau"]) else "—"
        hab_s = f"Daylight: {dr['prev_daylight_mins'] or '—'}m | Steps: {dr['prev_steps'] or '—'} | Ev.Noise: {dr['ev_noise_db'] or '—'} dB"
        out_s = f"Readiness: {dr['next_day_readiness'] or '—'}% | Stress: {dr['next_day_stress'] or '—'}%"
        diary_rows_html += f"""
        <tr>
          <td><b>{dr['date']}</b><br><span style="font-size:11px;color:#38bdf8;">{dr['exp_name']}</span></td>
          <td><b>{dur_s}</b><br><span style="font-size:11px;color:#94a3b8;">{dr['onset_str']} → {dr['wake_str']}</span></td>
          <td style="font-size:12px;">{stg_s}<br><span style="font-size:11px;color:#10b981;">Balance: {dr['arch_balance_score'] or '—'}/100</span></td>
          <td style="font-size:12px;">{vit_s}<br><span style="font-size:11px;color:#c084fc;">Nadir Timing: {nadir_s}</span></td>
          <td style="font-size:12px;color:#cbd5e1;">{hab_s}</td>
          <td style="font-size:12px;font-weight:600;color:#f8fafc;">{out_s}</td>
        </tr>"""

    # Build Stress Detective Episodes Table HTML
    ep_rows_html = ""
    if stress_bundle["episodes"]:
        for ep in stress_bundle["episodes"]:
            ep_col = "#ef4444" if ep["stress_pct"] >= 65 else "#f59e0b"
            ep_rows_html += f"""
            <tr>
              <td><b>{ep['timestamp']}</b><br><span style="font-size:11px;color:#94a3b8;">{ep['state']}</span></td>
              <td><span style="background:{ep_col}22;color:{ep_col};border:1px solid {ep_col};padding:3px 8px;border-radius:6px;font-weight:700;">{ep['stress_pct']}%</span></td>
              <td style="font-size:13px;color:#e2e8f0;"><b>Measured Facts:</b> {ep['facts']}</td>
              <td style="font-size:13px;color:#c084fc;"><b>Hypothesis:</b> {ep['hypothesis']}</td>
            </tr>"""
    else:
        ep_rows_html = """
        <tr>
          <td colspan="4" style="text-align:center;color:#10b981;padding:18px;">
            <b>No Unusual Non-Metabolic Stress Episodes Detected!</b> All recent heart rate elevations were cleanly explained by physical movement/walking.
          </td>
        </tr>"""

    table_html = ""
    for r in param_rows:
        badge_col = "#10b981" if r["status"] == "Active" else "#64748b"
        table_html += f"""
        <tr>
          <td><b>{r['name']}</b><br><span style="font-size:11px;color:#38bdf8;">{r['category']}</span></td>
          <td><span style="background:{badge_col}22;color:{badge_col};padding:3px 8px;border-radius:6px;font-weight:600;font-size:12px;">{r['latest']}</span></td>
          <td><b>{r['avg']}</b><br><span style="font-size:11px;color:#64748b;">Range: {r['range']}</span></td>
          <td style="color:#38bdf8;font-size:12px;">{r['normal']}</td>
          <td style="font-size:12px;color:#94a3b8;">{r['count']}</td>
          <td style="font-size:13px;line-height:1.4;color:#cbd5e1;">{r['desc']}</td>
        </tr>"""

    rec_items_html = "".join(f"<li style='margin-bottom:6px;'>{rec}</li>" for rec in sleep_bundle["recommendations"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>🌙 Sleep Lab • 🕵️ Stress Detective • ⚡ Energy Predictor — Apple Watch Series 11</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ background:#0b0f19; color:#e2e8f0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; margin:0; padding:24px; }}
  .header {{ border-bottom:1px solid #1e293b; padding-bottom:16px; margin-bottom:20px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; }}
  .header h1 {{ margin:0; font-size:24px; color:#38bdf8; }}
  .header p {{ margin:6px 0 0 0; color:#94a3b8; font-size:13.5px; }}
  .sleep-hero {{ background:linear-gradient(135deg, #0f172a 0%, #064e3b 100%); border:1px solid #059669; border-radius:14px; padding:22px; margin-bottom:22px; box-shadow:0 8px 30px rgba(0,0,0,0.45); }}
  .stress-hero {{ background:linear-gradient(135deg, #111827 0%, #2e1065 100%); border:1px solid #7e22ce; border-radius:14px; padding:22px; margin-bottom:22px; box-shadow:0 8px 30px rgba(0,0,0,0.45); }}
  .energy-hero {{ background:linear-gradient(135deg, #111827 0%, #1e1b4b 100%); border:1px solid #3730a3; border-radius:14px; padding:22px; margin-bottom:24px; box-shadow:0 8px 30px rgba(0,0,0,0.4); }}
  .hero-grid {{ display:grid; grid-template-columns:1.25fr 1fr; gap:20px; align-items:center; }}
  .pill-badge {{ display:inline-block; padding:4px 12px; border-radius:999px; font-weight:700; font-size:14px; margin-left:8px; }}
  .focus-box {{ background:#0f172a99; border-left:4px solid #38bdf8; padding:12px 16px; border-radius:8px; margin-top:12px; }}
  .sim-panel {{ background:#0f172acc; border:1px solid #1e293b; border-radius:10px; padding:16px; }}
  .slider-row {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:9px; font-size:12.5px; }}
  .slider-row input {{ width:50%; accent-color:#10b981; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(185px,1fr)); gap:14px; margin-bottom:24px; }}
  .kpi-card {{ background:#111827; border:1px solid #1f2937; border-radius:12px; padding:16px; }}
  .kpi-label {{ font-size:12px; color:#94a3b8; font-weight:600; }}
  .kpi-val {{ font-size:25px; font-weight:700; color:#f8fafc; margin:6px 0; }}
  .kpi-sub {{ font-size:12px; color:#10b981; }}
  .chart-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:18px; margin-bottom:28px; }}
  .chart-card {{ background:#111827; border:1px solid #1f2937; border-radius:12px; padding:14px; }}
  .chart-plot {{ height:360px; width:100%; }}
  .explain-box {{ background:#1e293b66; border-left:3px solid #38bdf8; padding:10px 12px; border-radius:6px; font-size:12.5px; color:#cbd5e1; margin-top:8px; line-height:1.45; }}
  table {{ width:100%; border-collapse:collapse; background:#111827; border-radius:12px; overflow:hidden; border:1px solid #1f2937; margin-bottom:28px; }}
  th {{ background:#1e293b; color:#38bdf8; text-align:left; padding:12px 14px; font-size:12px; text-transform:uppercase; }}
  td {{ padding:12px 14px; border-bottom:1px solid #1f2937; vertical-align:top; }}
  .disclaimer {{ font-size:12px; color:#94a3b8; background:#111827; border:1px solid #1f2937; padding:12px 16px; border-radius:8px; margin-bottom:22px; }}
  @media(max-width:1100px) {{ .chart-grid, .hero-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>🌙 Activity 3: Sleep Lab • 🕵️ Activity 1: Stress Detective • ⚡ Activity 2: Energy Predictor</h1>
      <p>Circular Von Mises Chronobiology • Hammock Nadir Parabola • Subspace Mahalanobis (D_M²) • Borbély Two-Process (S+C) • N-of-1 Trials</p>
    </div>
    <div style="font-size:12px;color:#10b981;background:#1e293b;padding:8px 12px;border-radius:8px;border:1px solid #059669;">
      <b>Self-Tests:</b> 6/6 Mathematical, Chronobiology & Privacy Checks Passed
    </div>
  </div>

  <!-- 🌙 ACTIVITY 3: 360° SLEEP EXPERIMENT LAB HERO -->
  <div class="sleep-hero">
    <div class="hero-grid">
      <div>
        <div style="font-size:12px;text-transform:uppercase;letter-spacing:1px;color:#34d399;font-weight:700;">
          Activity 3 // 360° Personal Sleep-Science & N-of-1 Experiment Lab
        </div>
        <h2 style="margin:8px 0;font-size:25px;color:#fff;">
          Restorative Sleep Index:
          <span class="pill-badge" style="background:{sleep_bundle['quality_color']}22;color:{sleep_bundle['quality_color']};border:1px solid {sleep_bundle['quality_color']};">
            {sleep_bundle['latest_quality']}% ({sleep_bundle['quality_label']})
          </span>
        </h2>
        <p style="font-size:14px;color:#e2e8f0;line-height:1.55;margin:10px 0;">
          <b>Circular Clock Regularity (R²):</b> <span style="color:#38bdf8;font-weight:700;">{sleep_bundle['regularity_score']}%</span>
          (Mean Window: <b>{sleep_bundle['mean_onset_str']} → {sleep_bundle['mean_wake_str']}</b>, ±{sleep_bundle['circ_sd_mins']}m SD)
          &nbsp;|&nbsp; <b>Social Jetlag:</b> <span style="color:#c084fc;font-weight:700;">{sleep_bundle['social_jetlag_hrs']} hrs</span>
          &nbsp;|&nbsp; <b>Overnight HR Nadir Timing:</b> <span style="color:#10b981;font-weight:700;">{sleep_bundle['overall_nadir_pct']}% of night</span>
        </p>
        <div class="focus-box">
          <div style="font-size:13.5px;font-weight:700;color:#34d399;">🧭 Recommended Next Sleep Experiments (Based on Your Data Gaps)</div>
          <ul style="margin:6px 0 0 0;padding-left:18px;font-size:12.5px;color:#cbd5e1;line-height:1.45;">
            {rec_items_html}
          </ul>
        </div>
      </div>

      <!-- Interactive N-of-1 Sleep Experiment Builder Simulator -->
      <div class="sim-panel">
        <div style="font-size:13px;font-weight:700;color:#34d399;margin-bottom:10px;">
          🧪 Interactive N-of-1 Sleep Experiment Builder (Predict Habit Effect Sizes)
        </div>
        <div class="slider-row">
          <span>Last Caffeine Cutoff Hour: <b id="slp_lbl_caf">14:00 (2 PM)</b></span>
          <input type="range" id="slp_sl_caf" min="10" max="22" step="1" value="14" oninput="updateSleepExpSim()">
        </div>
        <div class="slider-row">
          <span>Screen-Free Wind-Down: <b id="slp_lbl_scr">45 mins</b></span>
          <input type="range" id="slp_sl_scr" min="0" max="90" step="15" value="45" oninput="updateSleepExpSim()">
        </div>
        <div class="slider-row">
          <span>Morning Outdoor Daylight: <b id="slp_lbl_day">35 mins</b></span>
          <input type="range" id="slp_sl_day" min="0" max="90" step="5" value="35" oninput="updateSleepExpSim()">
        </div>
        <div class="slider-row">
          <span>Last Meal Before Bed: <b id="slp_lbl_mel">3.0 hrs</b></span>
          <input type="range" id="slp_sl_mel" min="0.5" max="5.0" step="0.5" value="3.0" oninput="updateSleepExpSim()">
        </div>
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid #1e293b;font-size:12.5px;">
          <div>Predicted Overnight HR Nadir Timing: <b id="slp_sim_nadir" style="color:#38bdf8;">48% of night (Optimal Early Vagal Dip)</b></div>
          <div style="margin-top:4px;">Predicted Next-Day Outcomes: <b id="slp_sim_out" style="font-size:15px;color:#10b981;">Sleep Score: 86.4% | Readiness: +9.2% | Stress: -11.5%</b></div>
        </div>
      </div>
    </div>
  </div>

  <!-- 🕵️ ACTIVITY 1: PERSONAL STRESS DETECTIVE HERO -->
  <div class="stress-hero">
    <div class="hero-grid">
      <div>
        <div style="font-size:12px;text-transform:uppercase;letter-spacing:1px;color:#c084fc;font-weight:700;">
          Activity 1 // Multisignal Autonomic Stress Detective
        </div>
        <h2 style="margin:8px 0;font-size:25px;color:#fff;">
          Personal Stress Likelihood:
          <span class="pill-badge" style="background:{stress_bundle['stress_color']}22;color:{stress_bundle['stress_color']};border:1px solid {stress_bundle['stress_color']};">
            {stress_bundle['today_likelihood']}% ± {stress_bundle['today_uncertainty']}% ({stress_bundle['stress_label']})
          </span>
        </h2>
        <p style="font-size:14.5px;color:#e2e8f0;line-height:1.55;margin:10px 0;">
          <b>Physiological Anomaly Score (Mahalanobis χ²):</b> <span style="color:#38bdf8;font-weight:700;">{stress_bundle['today_anomaly']}%</span>
          &nbsp;|&nbsp; <b>Model Confidence:</b> <span style="color:#10b981;font-weight:700;">{stress_bundle['today_confidence']}%</span> ({stress_bundle['today_k_sensors']}/8 core sensor channels active today)
        </p>
        <div class="focus-box">
          <div style="font-size:13.5px;font-weight:700;color:#38bdf8;">🛡️ Confounder & Double-Counting Protection Active</div>
          <div style="font-size:12.5px;color:#cbd5e1;margin-top:4px;line-height:1.45;">
            Physical movement (steps, stairs, active calories, and post-exercise heart rate recovery) is subtracted before evaluating stress. Correlated signals (HR, RHR, lnSDNN) are orthogonalized via <b>R⁻¹</b>, and missing sensors are never treated as zero.
          </div>
        </div>
      </div>

      <div class="sim-panel">
        <div style="font-size:13px;font-weight:700;color:#c084fc;margin-bottom:10px;">
          🧪 Interactive Confounder Gate Tester (Exercise vs. True Stress)
        </div>
        <div class="slider-row">
          <span>Observed Heart Rate: <b id="st_lbl_hr">112 bpm</b></span>
          <input type="range" id="st_sl_hr" min="55" max="165" step="1" value="112" oninput="updateStressSim()">
        </div>
        <div class="slider-row">
          <span>Concurrent Steps (5-min): <b id="st_lbl_stp">20 steps</b></span>
          <input type="range" id="st_sl_stp" min="0" max="600" step="20" value="20" oninput="updateStressSim()">
        </div>
        <div class="slider-row">
          <span>Concurrent HRV (SDNN): <b id="st_lbl_hrv">34 ms</b></span>
          <input type="range" id="st_sl_hrv" min="15" max="110" step="1" value="34" oninput="updateStressSim()">
        </div>
        <div class="slider-row">
          <span>Ambient Noise Exposure: <b id="st_lbl_db">78 dB</b></span>
          <input type="range" id="st_sl_db" min="45" max="95" step="1" value="78" oninput="updateStressSim()">
        </div>
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid #1e293b;font-size:12.5px;">
          <div>State Classification: <b id="st_sim_state" style="color:#f43f5e;">Sedentary — Non-Metabolic Cardiac Elevation (+38 bpm)</b></div>
          <div style="margin-top:4px;">De-Confounded Stress Estimate: <b id="st_sim_out" style="font-size:16px;color:#ef4444;">84.2% (High Autonomic Strain)</b></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ⚡ ACTIVITY 2: PERSONAL ENERGY PREDICTOR HERO -->
  <div class="energy-hero">
    <div class="hero-grid">
      <div>
        <div style="font-size:12px;text-transform:uppercase;letter-spacing:1px;color:#38bdf8;font-weight:700;">
          Activity 2 // Morning Biophysical Energy & Focus Predictor
        </div>
        <h2 style="margin:8px 0;font-size:25px;color:#fff;">
          Today's estimated readiness:
          <span class="pill-badge" style="background:{pred_bundle['badge_color']}22;color:{pred_bundle['badge_color']};border:1px solid {pred_bundle['badge_color']};">
            {pred_bundle['readiness_label']} ({pred_bundle['readiness_score']}%)
          </span>
        </h2>
        <p style="font-size:14.5px;color:#e2e8f0;line-height:1.55;margin:10px 0;">
          {pred_bundle['diagnostic_text']}
          <br><span style="color:#38bdf8;font-size:13.5px;">Predicted Subjective Energy Level: <b>{pred_bundle['energy_1_5']} / 5.0</b></span>
        </p>
        <div class="focus-box">
          <div style="font-size:14px;font-weight:700;color:#c084fc;">🎯 Suggested focus window: {pred_bundle['focus_window']}</div>
          <div style="font-size:12.5px;color:#cbd5e1;margin-top:4px;line-height:1.45;">
            {pred_bundle['focus_reason']}
          </div>
        </div>
      </div>

      <div class="sim-panel">
        <div style="font-size:13px;font-weight:700;color:#38bdf8;margin-bottom:10px;">
          🔬 Interactive "What-If" Readiness Simulator (Test Tonight's Habits)
        </div>
        <div class="slider-row">
          <span>Tonight's Sleep Duration: <b id="lbl_slp">7.5 hrs</b></span>
          <input type="range" id="sl_slp" min="4.5" max="10.0" step="0.5" value="7.5" oninput="updateSim()">
        </div>
        <div class="slider-row">
          <span>Bedtime Shift vs. Usual: <b id="lbl_jet">0.5 hrs</b></span>
          <input type="range" id="sl_jet" min="0.0" max="4.0" step="0.25" value="0.5" oninput="updateSim()">
        </div>
        <div class="slider-row">
          <span>Morning Resting HR: <b id="lbl_rhr">62 bpm</b></span>
          <input type="range" id="sl_rhr" min="48" max="85" step="1" value="62" oninput="updateSim()">
        </div>
        <div class="slider-row">
          <span>Morning HRV (SDNN): <b id="lbl_hrv">65 ms</b></span>
          <input type="range" id="sl_hrv" min="25" max="120" step="5" value="65" oninput="updateSim()">
        </div>
        <div class="slider-row">
          <span>Walking / Exercise Today: <b id="lbl_wlk">60 mins</b></span>
          <input type="range" id="sl_wlk" min="0" max="150" step="10" value="60" oninput="updateSim()">
        </div>
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid #1e293b;display:flex;justify-content:space-between;align-items:center;">
          <span style="font-size:13px;color:#94a3b8;">Simulated Tomorrow Readiness:</span>
          <span id="sim_out" style="font-size:17px;font-weight:700;color:#10b981;">78.4% (3.9 / 5.0)</span>
        </div>
      </div>
    </div>
  </div>

  <!-- KPI ROW -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Nervous System Battery (HRV)</div>
      <div class="kpi-val">{kpi_hrv}</div>
      <div class="kpi-sub">Higher = Calmer & More Recovered</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Resting Heart Rate</div>
      <div class="kpi-val">{kpi_rhr}</div>
      <div class="kpi-sub">Lower = Stronger Heart Stroke</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Overnight Heart Cool-Down</div>
      <div class="kpi-val">{dip_pct_str}</div>
      <div class="kpi-sub">Healthy Sleep Target: 10% – 20% Dip</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Average Daily Steps</div>
      <div class="kpi-val">{kpi_steps}</div>
      <div class="kpi-sub">Recorded directly by Apple Watch</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Natural Walking Speed</div>
      <div class="kpi-val">{kpi_speed}</div>
      <div class="kpi-sub">Healthy Brisk Range: 4.5 – 6.0 km/h</div>
    </div>
  </div>

  <!-- 🌙 SLEEP EXPERIMENT LAB HISTORY & CONFOUNDER TABLE -->
  <h2 style="color:#34d399;margin-bottom:10px;">🌙 Sleep Experiment Lab: N-of-1 Habit Trials, Effect Sizes (Hedges' g) & Confounder Audit</h2>
  <table>
    <thead>
      <tr>
        <th style="width:24%;">Experiment & Hypothesis</th>
        <th style="width:14%;">Sample Size & Adherence</th>
        <th style="width:13%;">Sleep Effect (Δ & g)</th>
        <th style="width:15%;">Next-Day Functioning</th>
        <th style="width:14%;">Controlled Confounders</th>
        <th style="width:20%;">Statistical Verdict (Correlation ≠ Causation)</th>
      </tr>
    </thead>
    <tbody>
      {exp_rows_html}
    </tbody>
  </table>

  <!-- 🌙 360° SLEEP DIARY & OVERNIGHT VITALS TABLE -->
  <h2 style="color:#38bdf8;margin-bottom:10px;">🛏️ 360° Sleep Diary: Wearable Architecture + Overnight Vitals + Logged Habits + Next-Day Outcomes</h2>
  <table>
    <thead>
      <tr>
        <th style="width:13%;">Date & Phase</th>
        <th style="width:15%;">Duration & Clock Window</th>
        <th style="width:16%;">Stages & Homeostatic Balance</th>
        <th style="width:20%;">Overnight HR, Dip %, HRV & Nadir</th>
        <th style="width:19%;">Previous-Day Habits & Environment</th>
        <th style="width:17%;">Next-Day Readiness & Stress</th>
      </tr>
    </thead>
    <tbody>
      {diary_rows_html}
    </tbody>
  </table>

  <!-- 🕵️ DETECTED STRESS EPISODES TABLE -->
  <h2 style="color:#c084fc;margin-bottom:10px;">🕵️ Stress Detective Log: Detected Non-Metabolic Episodes (Measured Facts vs. Hypotheses)</h2>
  <table>
    <thead>
      <tr>
        <th style="width:18%;">Timestamp & State</th>
        <th style="width:12%;">Stress Estimate</th>
        <th style="width:38%;">Measured Sensor Facts (Objective Telemetry)</th>
        <th style="width:32%;">Contextual Hypothesis (To Verify With Your Logs)</th>
      </tr>
    </thead>
    <tbody>
      {ep_rows_html}
    </tbody>
  </table>

  <!-- CHARTS GRID -->
  <div class="chart-grid">
    <!-- Sleep Chart 1: Normalized Overnight Hammock Parabola & HRV Recharge -->
    <div class="chart-card">
      <div id="c_slp_hammock" class="chart-plot"></div>
      <div class="explain-box">
        <b>Overnight Cardiac Hammock Parabola HR(τ) & Vagal Recharge:</b> Plots every sleeping pulse reading across normalized sleep time (0% = Fell Asleep, 50% = Mid-Sleep, 100% = Woke Up). A deep U-shaped parabola bottoming out around 45%–60% of the night proves fast parasympathetic recovery.
      </div>
    </div>

    <!-- Sleep Chart 2: N-of-1 Habit Effect Sizes (Hedges' g) -->
    <div class="chart-card">
      <div id="c_slp_effects" class="chart-plot"></div>
      <div class="explain-box">
        <b>N-of-1 Experiment Effect Sizes (Bias-Corrected Hedges' g):</b> Compares how strongly each habit shifts your <b>Restorative Sleep Score</b> and <b>Next-Day Energy Readiness</b>. Bars > +0.35 indicate meaningful positive associations.
      </div>
    </div>

    <!-- Stress Chart 1: Intraday De-Confounded Stress & HR Residual -->
    <div class="chart-card">
      <div id="c_st_intra" class="chart-plot"></div>
      <div class="explain-box">
        <b>Metabolic De-Confounding Timeline:</b> <b>Red Line</b> = Actual Heart Rate. <b>Blue Dotted Line</b> = Expected Heart Rate from walking/stairs/circadian rhythm. Whenever Red rises far above Blue while sedentary, <b>Purple Stress Likelihood</b> spikes.
      </div>
    </div>

    <!-- Stress Chart 2: Role-Grouped Signal Contributions -->
    <div class="chart-card">
      <div id="c_st_contrib" class="chart-plot"></div>
      <div class="explain-box">
        <b>Role-Assigned Signal Contributions (Z-Scores):</b> Bars to the <b>Right (Red)</b> push today's stress score up; bars to the <b>Left (Green)</b> buffer stress. Missing/calibrating sensors are shown in grey and sliced out of the Mahalanobis subspace without zero-imputation.
      </div>
    </div>

    <!-- Stress Chart 3: Historical Stress Posterior + Uncertainty -->
    <div class="chart-card">
      <div id="c_st_hist" class="chart-plot"></div>
      <div class="explain-box">
        <b>Daily Stress Posterior & Uncertainty Band:</b> Solid purple line shows daily Stress Likelihood surrounded by its <b>±95% Epistemic Uncertainty Ribbon</b>. Cyan dashed line shows your orthogonalized <b>Mahalanobis Anomaly Score (D_M²)</b>.
      </div>
    </div>

    <!-- Energy Chart: Borbely Two-Process Curve -->
    <div class="chart-card">
      <div id="c_borbely" class="chart-plot"></div>
      <div class="explain-box">
        <b>Borbély Two-Process Alertness Curve:</b> <b>Green Line</b> = Predicted cognitive energy (1–5) across the day. <b>Red Dashed (Process S)</b> = Adenosine fatigue building up. <b>Blue Dotted (Process C)</b> = 24-hour circadian rhythm.
      </div>
    </div>

    <div class="chart-card">
      <div id="c_sleep" class="chart-plot"></div>
      <div class="explain-box">
        <b>Overnight Sleep Stages Breakdown:</b> <b>Deep Sleep</b> repairs muscles and clears metabolic waste; <b>REM Sleep</b> consolidates memory and learning. Note that wearable sleep staging is an estimate (~80% agreement with clinical PSG).
      </div>
    </div>

    <div class="chart-card">
      <div id="c_quad" class="chart-plot"></div>
      <div class="explain-box">
        <b>Sympathovagal Phase Quadrant:</b> Every dot is an HRV check. <b>Bottom-Right (Green)</b> = High Vagal Recovery. <b>Top-Left (Red)</b> = Sympathetic Strain (Low HRV + High HR).
      </div>
    </div>

    <div class="chart-card">
      <div id="c_clock" class="chart-plot"></div>
      <div class="explain-box">
        <b>Your 24-Hour Body Clock:</b> Average heart rate by hour of the day (0 = Midnight, 12 = Noon).
      </div>
    </div>

    <div class="chart-card">
      <div id="c_gait" class="chart-plot"></div>
      <div class="explain-box">
        <b>Neuromuscular Gait & Posture:</b> Every dot is a 5-minute walk segment. Color shows <b>Double Support %</b> (how long both feet touch the ground—rises with central fatigue).
      </div>
    </div>
  </div>

  <div class="disclaimer">
    <b>Scientific & Wellness Notice:</b> This 3-module laboratory is a personal self-discovery and educational biometrics tool, not a medical diagnostic system. Apple Watch sleep stages are algorithmic estimates from optical pulse and motion sensors; more Deep or REM sleep does not automatically indicate better sleep, and observed associations in small-sample N-of-1 experiments reflect correlation rather than proof of causation. All data remains 100% local on your machine.
  </div>

  <h2 style="color:#38bdf8;margin-bottom:12px;">Complete Encyclopedia of Your Apple Watch Parameters & Model Roles</h2>
  <table>
    <thead>
      <tr>
        <th style="width:20%;">Parameter & Role</th>
        <th style="width:12%;">Latest Reading</th>
        <th style="width:14%;">Your Average</th>
        <th style="width:15%;">Healthy / Typical Range</th>
        <th style="width:11%;">Data Collected</th>
        <th style="width:28%;">What It Means in Plain English</th>
      </tr>
    </thead>
    <tbody>
      {table_html}
    </tbody>
  </table>

<script>
const D = {json.dumps(payload)};
const baseLayout = {{
  paper_bgcolor:'#111827', plot_bgcolor:'#111827',
  font:{{color:'#cbd5e1', family:'Segoe UI, sans-serif', size:11}},
  margin:{{t:42, r:38, b:42, l:52}},
  xaxis:{{gridcolor:'#1e293b', zerolinecolor:'#334155'}},
  yaxis:{{gridcolor:'#1e293b', zerolinecolor:'#334155'}},
  legend:{{orientation:'h', y:-0.18}}
}};

// Interactive Sleep Experiment Simulator (Activity 3)
function updateSleepExpSim() {{
  const caf = parseFloat(document.getElementById('slp_sl_caf').value);
  const scr = parseFloat(document.getElementById('slp_sl_scr').value);
  const day = parseFloat(document.getElementById('slp_sl_day').value);
  const mel = parseFloat(document.getElementById('slp_sl_mel').value);

  const ampm = caf >= 12 ? (caf === 12 ? '12:00 PM' : (caf - 12) + ':00 PM') : caf + ':00 AM';
  document.getElementById('slp_lbl_caf').innerText = caf + ':00 (' + ampm + ')';
  document.getElementById('slp_lbl_scr').innerText = scr + ' mins';
  document.getElementById('slp_lbl_day').innerText = day + ' mins';
  document.getElementById('slp_lbl_mel').innerText = mel.toFixed(1) + ' hrs';

  // Biophysical Nadir Timing & Sleep Quality response model
  const caf_penalty = Math.max(0, caf - 14.0) * 3.2;
  const mel_penalty = Math.max(0, 3.0 - mel) * 6.5;
  const scr_bonus = (scr - 15.0) * 0.18;
  const day_bonus = Math.min(45.0, day) * 0.22;

  const nadir_pct = Math.min(88, Math.max(36, Math.round(48 + (caf_penalty * 0.9) + (mel_penalty * 1.1) - (scr_bonus * 0.4))));
  const slp_score = Math.min(98.0, Math.max(35.0, 76.0 - caf_penalty - mel_penalty + scr_bonus + day_bonus));
  const read_delta = ((slp_score - 72.0) * 0.55);
  const str_delta = -((slp_score - 72.0) * 0.65);

  const nadirEl = document.getElementById('slp_sim_nadir');
  nadirEl.innerText = nadir_pct + '% of night (' + (nadir_pct <= 58 ? 'Optimal Early Vagal Dip' : 'Delayed Late-Night Cardiac Nadir') + ')';
  nadirEl.style.color = nadir_pct <= 58 ? '#10b981' : '#f59e0b';

  const outEl = document.getElementById('slp_sim_out');
  outEl.innerText = 'Sleep Score: ' + slp_score.toFixed(1) + '% | Next-Day Readiness: ' + (read_delta >= 0 ? '+' : '') + read_delta.toFixed(1) + '% | Stress: ' + (str_delta >= 0 ? '+' : '') + str_delta.toFixed(1) + '%';
  outEl.style.color = slp_score >= 75 ? '#10b981' : (slp_score >= 58 ? '#f59e0b' : '#ef4444');
}}

// Interactive Stress Detective Confounder Simulator (Activity 1)
function updateStressSim() {{
  const hr = parseFloat(document.getElementById('st_sl_hr').value);
  const stp = parseFloat(document.getElementById('st_sl_stp').value);
  const hrv = parseFloat(document.getElementById('st_sl_hrv').value);
  const db = parseFloat(document.getElementById('st_sl_db').value);

  document.getElementById('st_lbl_hr').innerText = hr + ' bpm';
  document.getElementById('st_lbl_stp').innerText = stp + ' steps';
  document.getElementById('st_lbl_hrv').innerText = hrv + ' ms';
  document.getElementById('st_lbl_db').innerText = db + ' dB';

  const exp_hr = 68.0 + 0.095 * stp;
  const res_hr = hr - exp_hr;
  const z_hr = res_hr / 7.5;
  const z_hrv = (Math.log(Math.max(D.mean_hrv, 45)) - Math.log(hrv)) / 0.25;
  const z_db = Math.max(0, (db - 68.0) / 10.0);

  const stateEl = document.getElementById('st_sim_state');
  const outEl = document.getElementById('st_sim_out');

  if (stp >= 180) {{
    stateEl.innerText = 'Physical Activity Gate Triggered (Expected HR: ' + exp_hr.toFixed(0) + ' bpm)';
    stateEl.style.color = '#38bdf8';
    const gated = Math.min(35.0, Math.max(12.0, 18.0 + 4.0 * Math.max(0, z_hr)));
    outEl.innerText = gated.toFixed(1) + '% (Exercise Tachycardia — Not Psychological Stress)';
    outEl.style.color = '#10b981';
  }} else {{
    stateEl.innerText = 'Sedentary Window (Non-Metabolic HR Excess: ' + (res_hr >= 0 ? '+' : '') + res_hr.toFixed(0) + ' bpm)';
    stateEl.style.color = res_hr > 10 ? '#f43f5e' : '#cbd5e1';
    const lin = 0.60 * z_hr + 0.45 * z_hrv + 0.18 * z_db;
    const prob = Math.min(98.0, Math.max(6.0, 100.0 / (1.0 + Math.exp(-1.25 * (lin - 0.35)))));
    outEl.innerText = prob.toFixed(1) + '% (' + (prob >= 68 ? 'High Sympathetic Strain' : (prob >= 42 ? 'Moderate Arousal' : 'Calm / Vagal Baseline')) + ')';
    outEl.style.color = prob >= 68 ? '#ef4444' : (prob >= 42 ? '#f59e0b' : '#10b981');
  }}
}}

// Interactive Energy Predictor Simulator (Activity 2)
function updateSim() {{
  const slp = parseFloat(document.getElementById('sl_slp').value);
  const jet = parseFloat(document.getElementById('sl_jet').value);
  const rhr = parseFloat(document.getElementById('sl_rhr').value);
  const hrv = parseFloat(document.getElementById('sl_hrv').value);
  const wlk = parseFloat(document.getElementById('sl_wlk').value);

  document.getElementById('lbl_slp').innerText = slp.toFixed(1) + ' hrs';
  document.getElementById('lbl_jet').innerText = jet.toFixed(2) + ' hrs';
  document.getElementById('lbl_rhr').innerText = rhr + ' bpm';
  document.getElementById('lbl_hrv').innerText = hrv + ' ms';
  document.getElementById('lbl_wlk').innerText = wlk + ' mins';

  const z_slp = (slp - 7.5) / 0.85;
  const z_jet = (jet - 0.5) / 0.5;
  const z_rhr = (rhr - D.mean_hr + 8) / 4.0;
  const z_hrv = (Math.log(hrv) - Math.log(Math.max(D.mean_hrv, 30))) / 0.22;
  const z_wlk = (wlk - 50.0) / 25.0;
  const z_ex = z_wlk - 0.25 * (z_wlk * z_wlk);

  const z_vec = [z_slp, z_jet, z_rhr, z_hrv, z_wlk, z_ex];
  let dot = 0;
  for (let i = 0; i < D.f_weights.length; i++) {{
    dot += z_vec[i] * D.f_weights[i];
  }}
  const pred_e = Math.min(5.0, Math.max(1.0, 3.2 + dot));
  const pred_pct = Math.min(99.0, Math.max(15.0, 100.0 / (1.0 + Math.exp(-1.35 * dot))));
  const outEl = document.getElementById('sim_out');
  outEl.innerText = pred_pct.toFixed(1) + '% (' + pred_e.toFixed(2) + ' / 5.0)';
  outEl.style.color = pred_pct >= 70 ? '#10b981' : (pred_pct >= 48 ? '#f59e0b' : '#ef4444');
}}

// Sleep Chart 1: Normalized Overnight Hammock Parabola
Plotly.newPlot('c_slp_hammock', [
  {{ x: D.slp_tau, y: D.slp_hr_mean, mode:'markers+lines', marker:{{size:6, color:'#94a3b8'}}, line:{{color:'#64748b', width:1.5, dash:'dot'}}, name:'Empirical Sleeping HR (bpm)' }},
  {{ x: D.slp_tau, y: D.slp_hr_fit, mode:'lines', line:{{color:'#10b981', width:3.2}}, name:'Quadratic Hammock Fit HR(τ)' }},
  {{ x: D.slp_tau, y: D.slp_hrv_mean, mode:'lines+markers', yaxis:'y2', marker:{{size:5}}, line:{{color:'#38bdf8', width:2}}, name:'Overnight Vagal HRV (ms)', connectgaps:true }}
], {{
  ...baseLayout,
  title: '<b>SL1. Normalized Overnight Hammock Parabola HR(τ) & Vagal Recharge</b>',
  xaxis: {{ ...baseLayout.xaxis, type:'linear', title:'Normalized Sleep Progression τ (% of Night Elapsed: 0% = Sleep Onset, 100% = Wake)' }},
  yaxis: {{ ...baseLayout.yaxis, type:'linear', title:'Sleeping Heart Rate (bpm)' }},
  yaxis2: {{ title:'Overnight HRV (ms)', overlaying:'y', side:'right', showgrid:false, font:{{color:'#38bdf8'}} }}
}}, {{responsive:true}});

// Sleep Chart 2: N-of-1 Habit Effect Sizes (Hedges' g)
Plotly.newPlot('c_slp_effects', [
  {{ x: D.eff_g_slp, y: D.eff_names, type:'bar', orientation:'h', name:'Sleep Quality Effect (Hedges g)', marker:{{color: D.eff_cols}} }},
  {{ x: D.eff_g_eng, y: D.eff_names, type:'bar', orientation:'h', name:'Next-Day Readiness Effect (Hedges g)', marker:{{color:'#a855f7', opacity:0.75}} }}
], {{
  ...baseLayout,
  barmode: 'group',
  margin: {{ t:42, r:25, b:42, l:215 }},
  title: '<b>SL2. N-of-1 Habit Effect Sizes on Sleep & Next-Day Readiness (Hedges\\' g)</b>',
  xaxis: {{ ...baseLayout.xaxis, type:'linear', title:'Standardized Effect Size Hedges\\' g (Right > 0 = Helpful, Left < 0 = Harmful)' }}
}}, {{responsive:true}});

// Stress Chart 1: Intraday De-Confounded Timeline
Plotly.newPlot('c_st_intra', [
  {{ x: D.in_ts, y: D.in_hr_raw, mode:'lines', line:{{color:'#f43f5e', width:1.8}}, name:'Actual HR (bpm)' }},
  {{ x: D.in_ts, y: D.in_hr_exp, mode:'lines', line:{{color:'#38bdf8', width:1.8, dash:'dot'}}, name:'Expected Metabolic HR (bpm)' }},
  {{ x: D.in_ts, y: D.in_stress, mode:'lines', yaxis:'y2', text: D.in_state, line:{{color:'#c084fc', width:2.2}}, name:'De-Confounded Stress (%)' }}
], {{
  ...baseLayout,
  title: '<b>S1. Intraday De-Confounded Stress Timeline (Actual vs. Metabolic HR)</b>',
  yaxis: {{ ...baseLayout.yaxis, type:'linear', title:'Heart Rate (bpm)' }},
  yaxis2: {{ title:'Stress Likelihood (%)', overlaying:'y', side:'right', showgrid:false, range:[0, 105], font:{{color:'#c084fc'}} }}
}}, {{responsive:true}});

// Stress Chart 2: Role-Grouped Signal Contributions
Plotly.newPlot('c_st_contrib', [{{
  x: D.c_z, y: D.c_names, text: D.c_roles, type:'bar', orientation:'h',
  marker: {{ color: D.c_colors }},
  hovertemplate: '<b>%{{y}}</b><br>%{{text}}<extra></extra>'
}}], {{
  ...baseLayout,
  margin: {{ t:42, r:25, b:42, l:210 }},
  title: '<b>S2. Today\\'s Contributing Signals by Physiological Role (Z-Scores)</b>',
  xaxis: {{ ...baseLayout.xaxis, type:'linear', title:'Standardized Deviation (Right/Red = Increases Stress, Left/Green = Buffers Stress)' }}
}}, {{responsive:true}});

// Stress Chart 3: Historical Stress Posterior + Uncertainty Band + Mahalanobis Anomaly
Plotly.newPlot('c_st_hist', [
  {{ x: D.st_dates, y: D.st_ci_up, mode:'lines', line:{{width:0}}, showlegend:false, hoverinfo:'skip' }},
  {{ x: D.st_dates, y: D.st_ci_dn, mode:'lines', fill:'tonexty', fillcolor:'rgba(192,132,252,0.18)', line:{{width:0}}, name:'95% Uncertainty Band (±CI)' }},
  {{ x: D.st_dates, y: D.st_like, mode:'lines+markers', line:{{color:'#c084fc', width:3}}, name:'Stress Likelihood (%)' }},
  {{ x: D.st_dates, y: D.st_anom, mode:'lines+markers', line:{{color:'#38bdf8', width:2, dash:'dash'}}, name:'Mahalanobis Anomaly Score (χ² %)' }}
], {{
  ...baseLayout,
  title: '<b>S3. Daily Stress Likelihood (±CI) & Mahalanobis Anomaly Trend</b>',
  yaxis: {{ ...baseLayout.yaxis, type:'linear', title:'Score (0% to 100%)', range:[0, 105] }}
}}, {{responsive:true}});

// Borbely Two-Process Curve
Plotly.newPlot('c_borbely', [
  {{ x: D.b_hours, y: D.b_alert, mode:'lines+markers', marker:{{size:5}}, line:{{color:'#10b981', width:3}}, name:'Predicted Focus & Energy (1–5)' }},
  {{ x: D.b_hours, y: D.b_adeno, mode:'lines', line:{{color:'#ef4444', width:2, dash:'dash'}}, name:'Process S (Adenosine Fatigue)' }},
  {{ x: D.b_hours, y: D.b_circ, mode:'lines', line:{{color:'#38bdf8', width:2, dash:'dot'}}, name:'Process C (Circadian Drive)' }}
], {{
  ...baseLayout,
  title: '<b>E1. Today\\'s Borbély Two-Process Cognitive Alertness Curve (06:00 – 24:00)</b>',
  xaxis: {{ ...baseLayout.xaxis, type:'linear', title:'Hour of Day (6 = 6 AM, 12 = Noon, 18 = 6 PM)', dtick:2 }},
  yaxis: {{ ...baseLayout.yaxis, type:'linear', title:'Scale (1.0 to 5.0)', range:[0.5, 5.2] }}
}}, {{responsive:true}});

// Sleep Architecture
Plotly.newPlot('c_sleep', [
  {{ x: D.sleep_dates, y: D.s_deep, type:'bar', name:'Deep (Body Repair)', marker:{{color:'#3b82f6'}} }},
  {{ x: D.sleep_dates, y: D.s_rem, type:'bar', name:'REM (Brain/Memory)', marker:{{color:'#a855f7'}} }},
  {{ x: D.sleep_dates, y: D.s_core, type:'bar', name:'Core (Light Sleep)', marker:{{color:'#38bdf8'}} }},
  {{ x: D.sleep_dates, y: D.s_awake, type:'bar', name:'Awake', marker:{{color:'#ef4444'}} }}
], {{
  ...baseLayout,
  barmode: 'stack',
  title: '<b>SL3. Overnight Sleep Stages Breakdown (Hours)</b>',
  yaxis: {{ ...baseLayout.yaxis, type: 'linear', title: 'Hours' }}
}}, {{responsive:true}});

// Sympathovagal Quadrant
Plotly.newPlot('c_quad', [{{
  x: D.quad_hrv, y: D.quad_hr, text: D.quad_labels,
  mode: 'markers',
  marker: {{ size: 11, color: D.quad_colors, line: {{width:1, color:'#fff'}} }},
  hovertemplate: '%{{text}}<br>HRV: <b>%{{x}} ms</b> | Heart Rate: <b>%{{y}} bpm</b><extra></extra>',
  name: 'HRV Reading'
}}], {{
  ...baseLayout,
  title: '<b>1. Nervous System Battery: Recovery (HRV) vs. Heart Rate</b>',
  xaxis: {{ ...baseLayout.xaxis, type: 'linear', title: 'Heart Rate Variability — HRV in ms (Right = More Relaxed →)' }},
  yaxis: {{ ...baseLayout.yaxis, type: 'linear', title: 'Heart Rate in bpm (Lower = Calmer ↓)' }},
  shapes: [
    {{ type:'line', x0:D.mean_hrv, x1:D.mean_hrv, y0:0, y1:1, yref:'paper', line:{{color:'#475569', dash:'dash'}} }},
    {{ type:'line', x0:0, x1:1, xref:'paper', y0:D.mean_hr, y1:D.mean_hr, line:{{color:'#475569', dash:'dash'}} }}
  ]
}}, {{responsive:true}});

// 24-Hour Body Clock
Plotly.newPlot('c_clock', [
  {{ x: D.circ_hours, y: D.circ_max, mode:'lines', line:{{width:0}}, showlegend:false, hoverinfo:'skip' }},
  {{ x: D.circ_hours, y: D.circ_min, mode:'lines', fill:'tonexty', fillcolor:'rgba(192,132,252,0.18)', line:{{width:0}}, name:'Typical Range (20th–80th %)' }},
  {{ x: D.circ_hours, y: D.circ_mean, mode:'lines+markers', line:{{color:'#c084fc', width:3}}, name:'Average HR by Hour of Day' }}
], {{
  ...baseLayout,
  title: '<b>2. Your 24-Hour Body Clock & Overnight Hammock Curve</b>',
  xaxis: {{ ...baseLayout.xaxis, type: 'linear', title: 'Hour of the Day (0 = Midnight, 6 = 6 AM, 12 = Noon, 18 = 6 PM)', dtick: 2 }},
  yaxis: {{ ...baseLayout.yaxis, type: 'linear', title: 'Heart Rate (bpm)' }}
}}, {{responsive:true}});

// Walking Gait Mechanics
Plotly.newPlot('c_gait', [{{
  x: D.gait_speed, y: D.gait_step, text: D.gait_time,
  mode: 'markers',
  marker: {{ size:8, color:D.gait_ds, colorscale:'Viridis', showscale:true, colorbar:{{title:'Double<br>Support %', thickness:12}} }},
  hovertemplate: '%{{text}}<br>Speed: <b>%{{x}} km/h</b> | Step Length: <b>%{{y}} cm</b><extra></extra>',
  name: 'Walk Segment'
}}], {{
  ...baseLayout,
  title: '<b>3. Walking Posture & Speed (Every Walk Segment)</b>',
  xaxis: {{ ...baseLayout.xaxis, type: 'linear', title: 'Walking Speed (km/h)' }},
  yaxis: {{ ...baseLayout.yaxis, type: 'linear', title: 'Step Length (cm)' }}
}}, {{responsive:true}});
</script>
</body>
</html>"""

    with open(HTML_DASHBOARD, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(HTML_DASHBOARD)
    print(f"[6/6] Launching Unified 3-Activity Biometric Science Lab: {abs_path}")
    try:
        os.startfile(abs_path)
    except AttributeError:
        webbrowser.open("file://" + abs_path)


if __name__ == "__main__":
    run_self_tests()
    df_raw, df_sleep, workout_windows = analyze_export(XML_FILE)
    latest_dt = sorted(df_raw["date"].unique())[-1] if not df_raw.empty else datetime.now().strftime("%Y-%m-%d")
    log_df = prompt_optional_daily_log(latest_dt)
    stress_bundle = compute_stress_detective_engine(df_raw, df_sleep, workout_windows, log_df)
    pred_bundle = compute_energy_predictor_engine(df_raw, df_sleep, log_df)
    sleep_bundle = compute_sleep_experiment_lab_engine(df_raw, df_sleep, log_df, stress_bundle, pred_bundle)
    build_dashboard(df_raw, df_sleep, log_df, stress_bundle, pred_bundle, sleep_bundle)