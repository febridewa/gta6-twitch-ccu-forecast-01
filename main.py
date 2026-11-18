"""GTA VI launch-day Twitch CCU forecaster (gta6-twitch-ccu-forecast-01).

Reads each /input/<id>.in case (schema makefun/twitch-ccu-case/v1) and writes
/output/<id>.out: for hours 2..24 of the release day, five quantiles of the
Twitch category's concurrent viewers, as {"hours": [{"hour": h, "q": [...]}]}.

Method (fitted at run time from the case's own comparables, stdlib only):
  Past big launches' release-day curves are the reference class. Each
  comparable launch contributes its 24 hourly viewer counts. Three
  estimators are fitted per hour and blended by geometric mean:
    A. DST-aligned log-log OLS on rolling-midnight launches: launches are
       aligned by the hour their US-East audience unlocks (04:00Z in daylight
       time, 05:00Z in standard time, i.e. first US hour 5 or 6), because the
       morning step lands on a different UTC hour depending on the date.
       Training set: launches whose hour-1 print is at least 8% of their
       day's peak, whose peak lands in the evening (hour >= 16), and which
       had no early access.
    B. Plain per-hour log-log OLS on ALL usable comparables: log viewers(h)
       regressed on log viewers(1). This keeps the very biggest launches
       (Elden Ring, Cyberpunk 2077: simultaneous global unlocks, hour-1
       prints near 900k) as anchors for the scale the final case will sit at.
       Its training set is windowed to launches whose hour-1 print is at
       least 1/20 of the target's: the simultaneous-unlock minnows (hour-1
       prints in the hundreds, growing 100-1000x by evening) live in a
       different regime and would otherwise drag this estimator's slope
       down so far that it predicts a 1M-print launch's evening below
       half its hour-1 print, which no big launch ever does.
    C. Huber (robust) per-hour log-log OLS on the same rolling set as A,
       which down-weights launches whose curve is an outlier for that hour.
  Each estimator's spread is the empirical quantiles of its own residuals,
  shrunk by 0.7 (in-sample residuals understate predictive spread, and the
  shrinkage was chosen on leave-one-launch-out tests, not on the practice).
  The target's log hour-1 print is winsorised to the training range, so a
  GTA-VI-sized print (beyond any comparable) does not extrapolate the
  log-linear slope into nonsense: the biggest launches' evening ratios sit
  near 0.9-1.3x their hour-1 print, and the winsorised forecast says so.
  Weights are 1 + 0.08 * (release year - 2018): recent launches count more.
  Blend weights: A 0.5, B 0.25, C 0.25 (geometric mean of quantiles).

Fallbacks: any hour an estimator cannot fit (< 5 training points) is dropped
from the blend for that hour; if no estimator fits, the house persistence
shape is used for that hour. With no usable comparables at all the whole
forecast is the persistence shape, so the program always writes a valid file.

Deterministic: pure arithmetic, no randomness, no network, no time or
environment dependence beyond the case file itself. Two runs on the same
case write byte-identical output.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. This entry is a single file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

SPREAD_MULT = 0.6          # shrinkage applied to empirical residual quantiles
BLEND_W = (0.6, 0.3, 0.1)  # (aligned rolling OLS, plain all-launch OLS, Huber rolling OLS)
MIN_POINTS = 5             # minimum training launches to fit one hour
RECENCY_PER_YEAR = 0.08    # weight grows 8% per year after 2018
SCALE_WINDOW = 20.0        # plain OLS trains only on launches within 20x of the target's print
SHRINK_B = 1.0             # slope shrinkage: fitted slope b -> 1 + SHRINK_B*(b-1); 1 keeps OLS
EXCLUDE_WEAK = False       # drop comparables the source tiers as weak analogues (F2P, slow-burn)
EXCLUDE_WEAK_A = True     # drop weak-tier launches from the aligned component only
EXCLUDE_UNVER_A = False    # drop unverified-unlock launches from the aligned component only
RATIO_W = 0.0              # weight of a 4th component: median log-ratio per hour (rolling set)
KERNEL_W = 0.15            # weight of a kernel (nearest-neighbour) component (rolling set)
KERNEL_BW = 1.5            # Gaussian bandwidth of the kernel, in log-viewer units
KLL_W = 0.0                # weight of a local-linear kernel OLS component (rolling set)
CONSOLE_BONUS = 1.0        # extra weight for console-only launches (no PC in platforms)


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def step_hour(day_start):
    """First UTC hour holding the US-East midnight unlock for a release day.

    US Eastern midnight is 04:00Z under daylight time (roughly March to
    early November) and 05:00Z under standard time, so the first US hour is
    hour 5 in daylight time and hour 6 in standard time. GTA VI releases
    2026-11-19 (standard time): first US hour 6, US West follows at 08:00Z.
    """
    dt = parse_ts(day_start)
    month, day = dt.month, dt.day
    if month in (4, 5, 6, 7, 8, 9, 10):
        return 5
    if month == 3:
        return 5 if day >= 15 else 6
    if month == 11:
        return 5 if day <= 2 else 6
    return 6


def is_early_access(unlock_text):
    text = (unlock_text or "").lower()
    return "early access" in text or "early-access" in text


def day_series(comparable):
    """Hourly viewers for hours 1..24 of the comparable's release day."""
    day_start = parse_ts(comparable["day_start"])
    by_hour = {}
    for point in comparable.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        hour = int((parse_ts(point["from"]) - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            by_hour[hour] = viewers
    return [by_hour.get(h) for h in range(1, 25)]


def is_weak(comparable):
    """The source tiers some launches as weak analogues (free-to-play,
    slow-burn, live-service): their curves follow a different regime."""
    return "tier: weak" in (comparable.get("unlock", "") or "").lower()


def build_curves(case):
    """Usable comparable launches, each with its release-day curve and weight."""
    own_name = (case.get("game") or {}).get("name", "")
    curves = []
    for comp in case.get("comparables") or []:
        name = comp.get("name", "")
        if name.startswith("Grand Theft Auto VI") or name == own_name:
            continue  # the GTA VI category's own pre-release history is no launch
        if EXCLUDE_WEAK and is_weak(comp):
            continue
        series = day_series(comp)
        if not series[0] or sum(1 for v in series if v) < 20:
            continue
        peak = max(v or 0 for v in series)
        if peak <= 0:
            continue
        peak_hour = series.index(peak) + 1
        year = parse_ts(comp["day_start"]).year
        weight = 1.0 + RECENCY_PER_YEAR * (year - 2018)
        if CONSOLE_BONUS != 1.0 and "PC" not in (comp.get("platforms") or ""):
            weight *= CONSOLE_BONUS
        curves.append({
            "name": name,
            "series": series,
            "weight": weight,
            "day_start": comp["day_start"],
            "p1_over_peak": series[0] / peak,
            "peak_hour": peak_hour,
            "early": is_early_access(comp.get("unlock", "")),
            "tier": comp.get("tier", ""),
            "weak": is_weak(comp),
            "unlock": comp.get("unlock", ""),
        })
    return curves


def is_rolling(curve):
    """A rolling-midnight launch: hour 1 already holds a real audience, the
    day's peak lands in the evening, and there was no early access."""
    return (curve["p1_over_peak"] >= 0.08 and curve["peak_hour"] >= 16
            and not curve["early"])


def quantile(sorted_values, tau):
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def wls(points):
    """Weighted least squares of y on x. points: (x, y, weight). Returns (a, b)."""
    total_w = sum(w for _, _, w in points)
    mx = sum(w * x for x, _, w in points) / total_w
    my = sum(w * y for _, y, w in points) / total_w
    sxx = sum(w * (x - mx) ** 2 for x, _, w in points)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in points)
    slope = sxy / sxx if sxx > 0 else 1.0
    return my - slope * mx, slope


def fit_hour(points, log_p1):
    """OLS fit for one hour. Returns (median log viewers, sorted residuals)."""
    intercept, slope = wls(points)
    slope = 1.0 + SHRINK_B * (slope - 1.0)
    intercept = (sum(w * y for _, y, w in points)
                 - slope * sum(w * x for x, _, w in points)) / sum(w for _, _, w in points)
    residuals = sorted(y - (intercept + slope * x) for x, y, _ in points)
    return intercept + slope * log_p1, residuals


def quantiles_from_residuals(median_log, residuals):
    return [math.exp(median_log + quantile(residuals, tau) * SPREAD_MULT)
            for tau in LEVELS]


def component_aligned(curves, log_p1, target_step):
    """Per-offset OLS on rolling launches, aligned by the US-East unlock hour.
    When EXCLUDE_WEAK_A / EXCLUDE_UNVER_A are set, weak-tier or
    unverified-unlock launches are left out of this component only."""
    rolling = [c for c in curves if is_rolling(c)]
    if EXCLUDE_WEAK_A:
        rolling = [c for c in rolling if not c.get("weak")]
    if EXCLUDE_UNVER_A:
        rolling = [c for c in rolling if "unverified" not in (c.get("unlock") or "").lower()]
    out = {}
    for offset in range(-4, 20):
        points = []
        for curve in rolling:
            hour = step_hour(curve["day_start"]) + offset
            if 1 <= hour <= 24 and curve["series"][hour - 1] and curve["series"][0]:
                points.append((math.log(max(1.0, curve["series"][0])),
                               math.log(max(1.0, curve["series"][hour - 1])),
                               curve["weight"]))
        if len(points) < MIN_POINTS:
            continue
        median_log, residuals = fit_hour(points, log_p1)
        out[offset] = quantiles_from_residuals(median_log, residuals)
    result = {}
    for hour in range(2, 25):
        if hour - target_step in out:
            result[hour] = out[hour - target_step]
    return result


def component_plain(curves, log_p1):
    """Per-hour OLS on every usable comparable launch within 20x of the
    target's hour-1 print (see the module docstring for why the window)."""
    out = {}
    for hour in range(2, 25):
        points = [(math.log(max(1.0, c["series"][0])),
                   math.log(max(1.0, c["series"][hour - 1])),
                   c["weight"])
                  for c in curves
                  if c["series"][hour - 1] and c["series"][0]
                  and math.log(max(1.0, c["series"][0])) >= log_p1 - math.log(SCALE_WINDOW)]
        if len(points) < MIN_POINTS:
            continue
        median_log, residuals = fit_hour(points, log_p1)
        out[hour] = quantiles_from_residuals(median_log, residuals)
    return out


def wquantile(pairs, tau):
    """Weighted quantile of (value, weight) pairs, linear interpolation."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return float("nan")
    target = tau * total
    acc = 0.0
    for i, (value, w) in enumerate(ordered):
        prev = acc
        acc += w
        if acc >= target:
            if i == 0 or acc == prev:
                return value
            frac = (target - prev) / (acc - prev)
            return ordered[i - 1][0] * (1.0 - frac) + value * frac
    return ordered[-1][0]


def component_kernel(curves, log_p1):
    """Nearest-neighbour curve forecast: per hour, a Gaussian kernel over
    log hour-1 prints weights the rolling launches, and the forecast
    quantiles are the kernel-weighted quantiles of their log viewers.
    Local in scale, so a huge target borrows only from the biggest launches
    and the shape can saturate instead of following one global line."""
    rolling = [c for c in curves if is_rolling(c)]
    out = {}
    for hour in range(2, 25):
        pairs = []
        for c in rolling:
            if c["series"][hour - 1] and c["series"][0]:
                x = math.log(max(1.0, c["series"][0]))
                kw = math.exp(-0.5 * ((x - log_p1) / KERNEL_BW) ** 2)
                pairs.append((math.log(max(1.0, c["series"][hour - 1])),
                              kw * c["weight"]))
        if len(pairs) < MIN_POINTS or sum(w for _, w in pairs) <= 0:
            continue
        centre = wquantile(pairs, 0.5)
        out[hour] = [math.exp(centre + (wquantile(pairs, tau) - centre) * SPREAD_MULT)
                     for tau in LEVELS]
    return out


def component_kernel_ll(curves, log_p1):
    """Local-linear kernel OLS: per hour, a Gaussian kernel over log hour-1
    prints weights the rolling launches and an OLS line is fitted through
    them. Local like the kernel, parametric like the OLS components."""
    rolling = [c for c in curves if is_rolling(c)]
    out = {}
    for hour in range(2, 25):
        points = []
        for c in rolling:
            if c["series"][hour - 1] and c["series"][0]:
                x = math.log(max(1.0, c["series"][0]))
                kw = math.exp(-0.5 * ((x - log_p1) / KERNEL_BW) ** 2)
                points.append((x, math.log(max(1.0, c["series"][hour - 1])),
                               kw * c["weight"]))
        if len(points) < MIN_POINTS or sum(w for _, _, w in points) <= 0:
            continue
        median_log, residuals = fit_hour(points, log_p1)
        out[hour] = quantiles_from_residuals(median_log, residuals)
    return out


def component_ratio(curves, log_p1):
    """Median log-ratio per hour on rolling launches: the evening/morning
    growth ratio's own distribution, no regression on the hour-1 print."""
    rolling = [c for c in curves if is_rolling(c)]
    out = {}
    for hour in range(2, 25):
        ratios = sorted(math.log(max(1.0, c["series"][hour - 1]))
                        - math.log(max(1.0, c["series"][0]))
                        for c in rolling
                        if c["series"][hour - 1] and c["series"][0])
        if len(ratios) < MIN_POINTS:
            continue
        out[hour] = [math.exp(log_p1 + quantile(ratios, tau) * SPREAD_MULT)
                     for tau in LEVELS]
    return out


def component_huber(curves, log_p1):
    """Per-hour Huber-robust OLS on rolling launches (down-weights outliers)."""
    rolling = [c for c in curves if is_rolling(c)]
    out = {}
    for hour in range(2, 25):
        points = [(math.log(max(1.0, c["series"][0])),
                   math.log(max(1.0, c["series"][hour - 1])),
                   c["weight"])
                  for c in rolling if c["series"][hour - 1] and c["series"][0]]
        if len(points) < MIN_POINTS:
            continue
        weights = [w for _, _, w in points]
        intercept = slope = None
        for _ in range(4):  # one plain fit, then three reweightings
            weighted = [(x, y, w) for (x, y, _), w in zip(points, weights)]
            intercept, slope = wls(weighted)
            residuals = [y - (intercept + slope * x) for x, y, _ in points]
            centre = statistics.median(residuals)
            scale = 1.4826 * statistics.median([abs(r - centre) for r in residuals]) or 0.3
            k = 1.345
            weights = [w * (1.0 if abs(r) <= k * scale else k * scale / abs(r))
                       for (_, _, w), r in zip(points, residuals)]
        slope = 1.0 + SHRINK_B * (slope - 1.0)
        intercept = (sum(w * y for (_, y, _), w in zip(points, weights))
                     - slope * sum(w * x for (x, _, _), w in zip(points, weights))) / sum(weights)
        residuals = sorted(y - (intercept + slope * x) for x, y, _ in points)
        median_log = intercept + slope * log_p1
        out[hour] = quantiles_from_residuals(median_log, residuals)
    return out


def round2(value):
    """Round to 2 decimals, half up, as the house baseline rounds it."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100
    return int(rounded) if float(rounded).is_integer() else rounded


def baseline_quantiles(p1, hour):
    """House persistence shape, used only where no estimator fits an hour."""
    return [round2(max(1.0, p1 * math.exp(z * 0.15 * math.sqrt(hour - 1)))) for z in Z]


def forecast_case(case):
    known = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if known is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(known["viewers"]))
    curves = build_curves(case)

    if not curves:
        return {"hours": [{"hour": h, "q": baseline_quantiles(p1, h)}
                          for h in range(2, 25)]}

    # Winsorise the target's log print to the comparables' observed range:
    # the slope is only supported inside it, and GTA VI's print may sit above it.
    train_logs = [math.log(max(1.0, c["series"][0])) for c in curves]
    log_p1 = math.log(p1)
    log_p1 = min(max(log_p1, min(train_logs)), max(train_logs))

    target_step = step_hour(case["day_start"])
    components = [
        component_aligned(curves, log_p1, target_step),
        component_plain(curves, log_p1),
        component_huber(curves, log_p1),
    ]
    weights = list(BLEND_W)
    if RATIO_W > 0:
        components.append(component_ratio(curves, log_p1))
        weights.append(RATIO_W)
    if KERNEL_W > 0:
        components.append(component_kernel(curves, log_p1))
        weights.append(KERNEL_W)
    if KLL_W > 0:
        components.append(component_kernel_ll(curves, log_p1))
        weights.append(KLL_W)

    hours = []
    for hour in range(2, 25):
        available = [(w, comp[hour]) for w, comp in zip(weights, components)
                     if hour in comp]
        if not available:
            hours.append({"hour": hour, "q": baseline_quantiles(p1, hour)})
            continue
        total_w = sum(w for w, _ in available)
        blended = []
        for i in range(5):
            log_q = sum(w * math.log(max(1.0, q[i])) for w, q in available) / total_w
            blended.append(math.exp(log_q))
        # Enforce the output contract: finite, > 0, <= 1e9, non-decreasing.
        cleaned = []
        previous = 0.0
        for value in blended:
            value = min(max(value, 1.0), 1e9)
            value = max(value, previous)
            previous = value
            cleaned.append(round2(value))
        # Rounding can only keep order (round2 is monotone), but re-enforce it.
        for i in range(1, 5):
            if cleaned[i] < cleaned[i - 1]:
                cleaned[i] = cleaned[i - 1]
        hours.append({"hour": hour, "q": cleaned})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
