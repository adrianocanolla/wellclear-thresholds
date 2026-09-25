"""Well Clear threshold evaluation for LLTEM terminal encounters.

CPA metrics, Loss-of-Well-Clear evaluation, per-geometry and population-
reweighted detection performance, predicted-CPA accuracy, and bootstrap
confidence intervals. Reading and preparing the trajectories is the job of
`lltem_preprocessing`; this module picks up from the aligned encounters and
the encounter-metadata DataFrame that module produces.

The threshold under test is the RTCA DO-365 en-route default
(DMOD=4000 ft, ZTHR=450 ft, TAUMOD=35 s), compared against the terminal-area
range recommended by Vincent et al. (2018).

Two scoring scales appear throughout and are deliberately kept apart:

  NMAC severity    500 ft / 100 ft, fixed, independent of the configuration
                   being tested -- the positive class for the headline
                   detection metrics
  Well Clear       the configuration's own DMOD / ZTHR volume -- used for
                   region classification and for the complementary
                   WCV-positive evaluation
"""

import numpy as np
import pandas as pd

DEFAULT_DMOD = 4000.0    # ft, RTCA DO-365 en-route default (baseline under test)
DEFAULT_ZTHR = 450.0     # ft
DEFAULT_TAUMOD = 35.0    # s

SEVERITY_TIERS = [
    ('Hazardous', 500.0, 100.0),
    ('Unsafe', DEFAULT_DMOD, DEFAULT_ZTHR),
]
# Descriptive severity taxonomy -- three nested
# tiers, both boundaries fixed and externally referenced (not chosen by this
# paper, and independent of whichever DMOD/ZTHR configuration is under test
# in a given sweep row):
#   'Hazardous' = the standard NMAC (Near Mid-Air Collision) definition used
#                 throughout the DAA/DO-365 literature (< 500ft horizontal,
#                 < 100ft vertical).
#   'Unsafe'    = a Well Clear Violation (WCV): inside the RTCA DO-365
#                 en-route default's own Well Clear volume (< DMOD/ZTHR,
#                 i.e. < 4000ft/450ft) but not NMAC-severity.
#   'Safe'      = outside the en-route default's Well Clear volume entirely.
# Only 'Hazardous' is the positive class for the detection metrics (see
# confusion_counts) -- 'Unsafe' is a descriptive middle category (used in
# population/sample characterization) that does not, by itself, change any
# detection-performance metric in this paper. An earlier revision used four
# nested tiers (Hazardous/Severe/Moderate/Safe) with an invented intermediate
# boundary at 2000ft/300ft that had no independent standards basis; a later
# revision collapsed to a strict Hazardous/Safe binary to remove ground-truth
# circularity with the tested DMOD/ZTHR entirely. This three-tier version
# restores a middle category without reintroducing that circularity, since
# both 'Unsafe' tier's boundaries (NMAC and the en-route default) are fixed
# reference points, not the swept threshold under test.

# The originally swept ranges (2000-6000/300-600/25-45) kept pinning at
# their lower bound in every sweep run. An initial extension down to DMOD=500ft/
# TAUMOD=0s confirmed the boundary-pinning persisted, but those floors are not
# operationally realistic: TAUMOD=0 gives no predictive lead time to alert before
# Vincent et al.'s final terminal-area recommendation (their Section V Conclusion):
# dh unchanged at 450ft, HMD in [1000, 2000]ft, tau_mod in [15, 25]s -- used in
# ## 9 to check whether this LLTEM-driven analysis lands inside, above, or below
# their range.
VINCENT_2018_RECOMMENDATION = {
    'DMOD': (1000.0, 2000.0),
    'ZTHR': (450.0, 450.0),
    'TAUMOD': (15.0, 25.0),
}

# Named (DMOD, ZTHR, TAUMOD) configurations to evaluate jointly and compare
# directly, rather than only reading off the one-parameter-at-a-time sweep.
NAMED_CONFIGS = {
    'En-route baseline (DO-365)': (DEFAULT_DMOD, DEFAULT_ZTHR, DEFAULT_TAUMOD),
    "Vincent et al. minimum (HMD=1000ft, tau=15s)": (1000.0, 450.0, 15.0),
    "Vincent et al. maximum (HMD=2000ft, tau=25s)": (2000.0, 450.0, 25.0),
    "Below-Vincent comparison (HMD=1000ft, tau=10s)": (1000.0, 450.0, 10.0),
}

# RTCA DO-365B's terminal DAA Well Clear volume, adopted by SC-228 for Phase 2
# approach and departure operations: distance-only, with no time-based branch.
# Kept out of NAMED_CONFIGS deliberately -- it is reported as one reference
# point alongside the four configurations under test, not folded into the
# per-geometry, reweighted, sensitivity and bootstrap analyses, which are scoped
# to the en-route default and Vincent et al.'s range.
DO365B_TERMINAL = (1500.0, 450.0, 0.0)


def _label_tiers(r_h, r_v, tiers):
    """Nested-tier labeling shared by label_severity and label_from_cpa.

    Outermost (widest) tier first, inner (tighter) tiers overwrite, so a
    later, tighter match wins over an earlier, looser one.
    """
    r_h = np.atleast_1d(np.asarray(r_h, dtype=float))
    r_v = np.atleast_1d(np.asarray(r_v, dtype=float))
    label = np.full(r_h.shape, 'Safe', dtype=object)
    for name, h_thr, v_thr in reversed(tiers):
        label[(r_h < h_thr) & (r_v < v_thr)] = name
    return label


def label_severity(r_h, r_v):
    """Hazardous (NMAC) / Unsafe (WCV) / Safe -- see SEVERITY_TIERS."""
    return _label_tiers(r_h, r_v, SEVERITY_TIERS)


def label_from_cpa(r_h, r_v, dmod=DEFAULT_DMOD, zthr=DEFAULT_ZTHR):
    """label_severity generalized to an arbitrary (DMOD, ZTHR) Well Clear
    volume, for comparing a predicted vs. true CPA under the SAME threshold
    configuration under test; see predicted_cpa_accuracy.

    Hazardous keeps its fixed NMAC boundary (500ft/100ft) regardless of
    configuration -- it is the externally-defined positive class and is
    never a function of the thresholds under test (see SEVERITY_TIERS).
    Only the middle 'Unsafe' (WCV) tier follows (dmod, zthr), since "inside
    the Well Clear volume" is by definition relative to whichever volume is
    being evaluated. With the default arguments this reproduces
    label_severity exactly, since SEVERITY_TIERS' 'Unsafe' tier IS the
    en-route default's own (DMOD, ZTHR).
    """
    return _label_tiers(r_h, r_v, [('Hazardous', 500.0, 100.0), ('Unsafe', float(dmod), float(zthr))])


# ---------------------------------------------------------------------------
# Step 2 -- CPA metrics
# ---------------------------------------------------------------------------

def classify_geometry_vectorized(own_trk, int_trk):
    """Numpy-vectorized form of the notebook's classify_geometry (degrees).

    Head-on: |Δtrack| > 135 deg | Overtaking: < 45 deg | Crossing: otherwise.
    """
    own_trk = np.asarray(own_trk, dtype=float)
    int_trk = np.asarray(int_trk, dtype=float)
    rel = np.abs((int_trk - own_trk + 180) % 360 - 180)
    return np.where(rel > 135, 'head-on', np.where(rel < 45, 'overtaking', 'crossing'))


def scalar_cpa_metrics(info_df):
    """Population-scale (1M-row) CPA metrics from precomputed metadata only.

    hmd_ft/vmd_ft ARE the encounter's true r_h,min / r_v,CPA (LLTEM computes
    them directly from the simulated trajectory), so no time series is
    needed for severity labeling at this scale. tau_mod, however, needs a
    real closure-rate history and can only be approximated here from the
    CPA-snapshot speeds/tracks (law-of-cosines closing speed) -- this is for
    descriptive population statistics only, NOT used for LoWC time-series
    evaluation (see evaluate_lowc_batch / timeseries_cpa_metrics, which use
    the real per-second trajectories instead).
    """
    own_trk = info_df['own_trk_angle'].values
    int_trk = info_df['int_trk_angle'].values

    df = pd.DataFrame({
        'id': info_df['id'].values,
        'geometry': classify_geometry_vectorized(own_trk, int_trk),
        'int_intent': info_df['int_intent'].values,
        'r_h_min': info_df['hmd_ft'].abs().values,
        'r_v_at_cpa': info_df['vmd_ft'].abs().values,
    })

    delta_track = np.radians((int_trk - own_trk + 180) % 360 - 180)
    own_speed = info_df['own_speed'].values
    int_speed = info_df['int_speed'].values
    closing_speed = np.sqrt(
        own_speed ** 2 + int_speed ** 2 - 2 * own_speed * int_speed * np.cos(delta_track)
    )
    df['closing_speed_approx'] = closing_speed
    with np.errstate(divide='ignore', invalid='ignore'):
        df['tau_mod_approx'] = np.where(
            closing_speed > 1e-6,
            (df['r_h_min'] - DEFAULT_DMOD) / closing_speed,
            np.nan,
        )
    df['severity'] = label_severity(df['r_h_min'].values, df['r_v_at_cpa'].values)
    return df


def compute_tau_mod(r_h, d_range_h, dmod=DEFAULT_DMOD):
    """Modified tau at an arbitrary DMOD, in the RTCA DO-365 / DAIDALUS form:

        tau_mod(t) = (DMOD^2 - ||s||^2) / (s . v)   for closing encounters
                   = +inf                            otherwise

    where s is the horizontal relative position vector and v the horizontal
    relative velocity vector. Since d(r_h)/dt = (s . v) / ||s||, the inner
    product is recovered here as s.v = r_h * d_range_h, so this needs no
    inputs beyond the range and its signed rate.

    DMOD here is the SAME parameter as the distance gate's DMOD in
    eq. `eq:lowc` -- it is not a separate constant. evaluate_lowc_batch
    recomputes tau_mod at whichever dmod it is called with; reusing a
    tau_mod array computed at a different DMOD (as an earlier revision of
    this module did, pinned at DEFAULT_DMOD regardless of the dmod under
    test) silently evaluates the predictive gate against the wrong
    threshold.

    Non-closing encounters (s . v >= 0, i.e. the range is stable or opening)
    have no predicted incursion to extrapolate, so tau_mod is +inf and the
    predictive gate can never fire. This is the standard's explicit
    treatment; NaN rates fall into the same branch and are likewise +inf.

    Note on the previous implementation: this module previously computed
    (r_h - DMOD) / |d_range_h|, which differs from the standard by a factor
    of (1 + DMOD/r_h) -- always smaller, so the predictive gate fired more
    readily than DO-365 specifies -- and, by taking the absolute value of
    the range rate, allowed a diverging encounter to produce a small
    positive tau_mod and trigger an alert.
    """
    r_h = np.asarray(r_h, dtype=float)
    r_dot = np.asarray(d_range_h, dtype=float)   # signed range rate
    s_dot_v = r_h * r_dot                         # s . v
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = (float(dmod) ** 2 - r_h ** 2) / s_dot_v
    return np.where(s_dot_v < -1e-6, tau, np.inf)


def timeseries_cpa_metrics(aligned):
    """Per-timestep TCPA(t)/HMD(t)/ZCPA(t)/tau_mod(t) for one aligned encounter
    reusing its smoothed position channels
    and a smoothed relative velocity (built from the per-aircraft smoothed
    velocity channels, since rel_vx/vy/vz themselves aren't smoothed upstream).

    Locates the trajectory's own argmin(r_h(t)) as the true CPA, rather than
    assuming the alignment window's center (t=0) is exactly CPA -- the window
    is only centered on the metadata's reported tcpa, which can differ
    slightly from the smoothed trajectory's actual minimum-range point.
    """
    t = aligned['time']

    rx, ry, rz = aligned['rel_x_smooth'], aligned['rel_y_smooth'], aligned['rel_z_smooth']
    vx = aligned['int_vx_smooth'] - aligned['own_vx_smooth']
    vy = aligned['int_vy_smooth'] - aligned['own_vy_smooth']
    vz = aligned['int_vz_smooth'] - aligned['own_vz_smooth']

    # range_h_smooth can overshoot slightly below zero (Savitzky-Golay cubic
    # fit ringing near a sharp, near-zero minimum) -- physically impossible
    # for a horizontal range, so clip.
    r_h = np.maximum(aligned['range_h_smooth'], 0.0)
    r_v = np.abs(rz)
    d_range_h = aligned['d_range_h']

    # DO-365 defines the horizontal miss distance at the time that minimises
    # the range *in the horizontal plane*, so TCPA is taken on the horizontal
    # components alone. ZCPA is then the vertical separation at that same
    # instant.
    vh_sq = vx ** 2 + vy ** 2
    v_sq = vh_sq + vz ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        tcpa = np.where(vh_sq > 1e-6, -(rx * vx + ry * vy) / vh_sq, 0.0)
        hmd = np.sqrt((rx + vx * tcpa) ** 2 + (ry + vy * tcpa) ** 2)
        zcpa = np.abs(rz + vz * tcpa)

    # 'tau_mod_default_dmod' is evaluated at DEFAULT_DMOD ONLY -- it is for
    # standalone per-metric use (a "tau_mod at CPA" summary
    # comparison, and the alert-trigger breakdown, both of which describe
    # the en-route baseline specifically). It is NOT used by
    # evaluate_lowc_batch, which recomputes tau_mod fresh (compute_tau_mod)
    # at whichever dmod is actually under test -- see that function's
    # docstring for why reusing this array there would be a bug.
    tau_mod_default_dmod = compute_tau_mod(r_h, d_range_h, DEFAULT_DMOD)

    idx_cpa = int(np.argmin(r_h))
    closure_at_cpa = float(np.sqrt(v_sq[idx_cpa]))

    return {
        'encounter_id': aligned['encounter_id'],
        'geometry': aligned['geometry'],
        'time': t,
        'r_h': r_h, 'r_v': r_v, 'TCPA': tcpa, 'HMD': hmd, 'ZCPA': zcpa,
        'd_range_h': d_range_h, 'tau_mod_default_dmod': tau_mod_default_dmod,
        'idx_cpa': idx_cpa, 't_cpa': float(t[idx_cpa]),
        'r_h_min': float(r_h[idx_cpa]), 'r_v_at_cpa': float(r_v[idx_cpa]),
        'closure_at_cpa': closure_at_cpa,
    }


def stack_metrics(metrics_list):
    """Combine a list of timeseries_cpa_metrics() dicts into 2D arrays
    (n_encounters x n_timesteps) for fast vectorized threshold evaluation --
    the paper's bootstrap/CV/sweep steps need thousands of LoWC evaluations,
    far too slow done one encounter at a time in a Python loop.
    """
    return {
        'encounter_id': np.array([m['encounter_id'] for m in metrics_list]),
        'geometry': np.array([m['geometry'] for m in metrics_list], dtype=object),
        'time': metrics_list[0]['time'],
        'r_h': np.stack([m['r_h'] for m in metrics_list]),
        'r_v': np.stack([m['r_v'] for m in metrics_list]),
        'HMD': np.stack([m['HMD'] for m in metrics_list]),
        'ZCPA': np.stack([m['ZCPA'] for m in metrics_list]),
        'd_range_h': np.stack([m['d_range_h'] for m in metrics_list]),
        'tau_mod_default_dmod': np.stack([m['tau_mod_default_dmod'] for m in metrics_list]),
        'idx_cpa': np.array([m['idx_cpa'] for m in metrics_list]),
        't_cpa': np.array([m['t_cpa'] for m in metrics_list]),
        'r_h_min': np.array([m['r_h_min'] for m in metrics_list]),
        'r_v_at_cpa': np.array([m['r_v_at_cpa'] for m in metrics_list]),
    }


def build_sample_metrics(aligned_list):
    """Stack per-encounter CPA metrics over a list of aligned encounters.

    Returns (stacked, severity, geometry), the three inputs every evaluation
    function in this module takes.
    """
    metrics_list = [timeseries_cpa_metrics(a) for a in aligned_list]
    stacked = stack_metrics(metrics_list)
    severity = label_severity(stacked['r_h_min'], stacked['r_v_at_cpa'])
    return stacked, severity, stacked['geometry']


# ---------------------------------------------------------------------------
# Step 3 -- Well Clear threshold evaluation
# ---------------------------------------------------------------------------

def evaluate_lowc_batch(stacked, dmod=DEFAULT_DMOD, zthr=DEFAULT_ZTHR, taumod=DEFAULT_TAUMOD,
                         hmd_threshold=None):
    """Vectorized Loss-of-Well-Clear evaluation over every encounter in a
    stack_metrics() result, for one threshold configuration:

        LoWC(t) = (r_h(t) < DMOD  OR  (0 <= tau_mod(t) < TAUMOD
                                       AND HMD(t) < HMD_threshold))
                  AND (r_v(t) < ZTHR)

    The lower bound is DAIDALUS's and is stated for conformance, not for
    effect: tau_mod(t) < 0 requires DMOD^2 - ||s(t)||^2 > 0, i.e. r_h(t) < DMOD,
    which already satisfies the first branch of the disjunction. The guard
    therefore cannot change any decision, and the assertion below checks that
    on the data rather than leaving it as an argument.

    This corrects an operator-precedence ambiguity in the paper draft's
    As literally typeset in the paper (AND binds tighter than OR),
    "r_h < DMOD OR (...) AND r_v < ZTHR" would let r_h < DMOD alone trigger
    LoWC without ever checking vertical separation. The structure above
    (horizontal-OR-tau, AND vertical) matches the paper's evident intent and
    standard DAA Well Clear logic.

    `dmod` sets BOTH the distance gate's radius and DMOD in tau_mod's
    numerator (eq. `eq:taumod`, via compute_tau_mod) -- they are the same
    parameter in the standard and must move together; tau_mod is therefore
    recomputed fresh here at THIS call's dmod, never read from a
    precomputed array. `hmd_threshold` decouples ONLY the HMD gate's radius
    from `dmod` and defaults to it; it never affects tau_mod.
    """
    if hmd_threshold is None:
        hmd_threshold = dmod

    tau_mod = compute_tau_mod(stacked['r_h'], stacked['d_range_h'], dmod)
    inside = stacked['r_h'] < dmod
    predictive = (tau_mod >= 0) & (tau_mod < taumod) & (stacked['HMD'] < hmd_threshold)
    assert not np.any((tau_mod < 0) & ~inside), (
        "tau_mod < 0 outside DMOD: the lower bound in eq. `eq:lowc` would then "
        "change a decision, contradicting the docstring's conformance argument")
    horizontal = inside | predictive
    lowc = horizontal & (stacked['r_v'] < zthr)

    any_lowc = lowc.any(axis=1)
    first_idx = np.where(any_lowc, lowc.argmax(axis=1), -1)
    t = stacked['time']
    t_alert = np.where(any_lowc, t[first_idx], np.nan)
    lead_time = np.where(any_lowc, stacked['t_cpa'] - t_alert, np.nan)

    return {'lowc': lowc, 'any_lowc': any_lowc, 'first_idx': first_idx,
            't_alert': t_alert, 'lead_time': lead_time}


def confusion_counts(severity_labels, lowc_flags, positive_labels=('Hazardous',)):
    """TP/FP/TN/FN and TPR/FPR/Precision/F1; positive class = Hazardous
    (NMAC ground truth). 'Unsafe' (WCV, non-NMAC)
    and 'Safe' labels are both treated as negative here -- the three-tier
    severity taxonomy is descriptive; only the Hazardous/NMAC boundary
    defines the positive class for every detection-performance metric.

    positive_labels widens that ground-truth positive class. The default
    ('Hazardous',) is the NMAC-severity question the paper's headline table
    answers. positive_labels=('Hazardous', 'Unsafe') instead asks the
    Well-Clear-violation question -- i.e. did the alert correctly identify
    an encounter that entered the Well Clear volume, regardless of whether
    it reached NMAC severity. Reporting both separates "this alert was not
    an NMAC" from "this alert was not a violation of anything", which are
    different operational claims (added in response to review)."""
    severity_labels = np.asarray(severity_labels, dtype=object)
    lowc_flags = np.asarray(lowc_flags, dtype=bool)

    positive = np.isin(severity_labels, np.asarray(positive_labels, dtype=object))
    tp = int(np.sum(positive & lowc_flags))
    fn = int(np.sum(positive & ~lowc_flags))
    fp = int(np.sum(~positive & lowc_flags))
    tn = int(np.sum(~positive & ~lowc_flags))

    tpr = tp / (tp + fn) if (tp + fn) else float('nan')
    fpr = fp / (fp + tn) if (fp + tn) else float('nan')
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    if np.isnan(precision) or np.isnan(tpr) or (precision + tpr) == 0:
        f1 = float('nan')
    else:
        f1 = 2 * precision * tpr / (precision + tpr)

    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'TPR': tpr, 'FPR': fpr,
            'Precision': precision, 'F1': f1}


# ---------------------------------------------------------------------------
# Step 5 -- bootstrap validation and statistical testing
# ---------------------------------------------------------------------------

# Exactly the per-encounter keys evaluate_lowc_batch reads (it recomputes
# tau_mod itself from 'r_h'/'d_range_h', so 'd_range_h' -- not a precomputed
# tau_mod -- must be here); _resample_stacked must carry all of them, plus
# the shared 'time' grid.
LOWC_STACK_KEYS = ('r_h', 'r_v', 'HMD', 'd_range_h', 't_cpa')


def _resample_stacked(stacked, idx):
    """Index just the per-encounter arrays evaluate_lowc_batch actually reads;
    'time' is the shared uniform grid and isn't per-encounter."""
    resampled = {'time': stacked['time']}
    resampled.update({key: stacked[key][idx] for key in LOWC_STACK_KEYS})
    return resampled


def _subset_stacked(stacked, idx):
    """Index *every* per-encounter array, not just the LoWC subset.

    _resample_stacked above is deliberately minimal because the bootstrap calls
    it hundreds of thousands of times and only needs the LoWC inputs. The
    predicted-CPA path needs more (ZCPA, idx_cpa, r_h_min, r_v_at_cpa,
    tau_mod_default_dmod), so stratified reporting uses this fuller version."""
    n = len(stacked['r_h'])
    out = {}
    for key, val in stacked.items():
        if key == 'time':
            out[key] = val
        elif isinstance(val, np.ndarray) and len(val) == n:
            out[key] = val[idx]
        else:
            out[key] = val
    return out


# ---------------------------------------------------------------------------
# Step 6 -- threshold recommendations
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 7 -- predicted-CPA vs. true-CPA labeling accuracy
# ---------------------------------------------------------------------------

SEVERITY_RANK = {'Safe': 0, 'Unsafe': 1, 'Hazardous': 2}

# The three configurations compared by the predicted-CPA accuracy analysis,
# selected by name from NAMED_CONFIGS so the (DMOD, ZTHR, TAUMOD) tuples have
# a single definition site. The below-Vincent comparison point is excluded:
# it is a sensitivity probe below Vincent et al.'s stated floor (Section
# 5.3), not one of the three operationally-motivated configurations compared
# here.
PREDICTED_CPA_CONFIGS = {name: NAMED_CONFIGS[name] for name in (
    'En-route baseline (DO-365)',
    "Vincent et al. maximum (HMD=2000ft, tau=25s)",
    "Vincent et al. minimum (HMD=1000ft, tau=15s)",
    "Below-Vincent comparison (HMD=1000ft, tau=10s)",
)}


def predict_cpa_from_gate(stacked, dmod=DEFAULT_DMOD, taumod=DEFAULT_TAUMOD):
    """Per-encounter predicted CPA taken from the predictive gate's most
    alarming engaged timestep, for one (DMOD, TAUMOD) configuration.

    The predictive gate is 'engaged' at every t where tau_mod(t; dmod) <
    taumod, with tau_mod recomputed at THIS dmod (compute_tau_mod) -- not
    also requiring HMD(t) < dmod, so an 'estimated = Safe' outcome stays
    reachable in predicted_cpa_accuracy() below (the gate can be engaged in
    time while still predicting a mild outcome). Among engaged timesteps the
    encounter's estimate is taken at argmin HMD(t) -- the single most
    alarming forward-extrapolated CPA the gate ever asserts -- and reported
    as the (r_h, r_v) pair (HMD(t*), ZCPA(t*)). Encounters whose gate never
    engages have no prediction (has_prediction False, est_r_h/est_r_v NaN).

    No restriction to pre-CPA timesteps: once r_h(t) < dmod, tau_mod(t) is
    negative and so trivially satisfies tau_mod < taumod for every t
    afterward too, including well past the true CPA, where HMD(t) is no
    longer a genuine forward prediction but a near-exact reconstruction of
    what already happened. This is a known, accepted property of the
    analysis as configured, not a bug.
    """
    tau_mod = compute_tau_mod(stacked['r_h'], stacked['d_range_h'], dmod)
    engaged = (tau_mod < taumod) & np.isfinite(stacked['HMD']) & np.isfinite(stacked['ZCPA'])

    has_prediction = engaged.any(axis=1)
    idx_pred = np.where(engaged, stacked['HMD'], np.inf).argmin(axis=1)
    rows = np.arange(len(idx_pred))
    est_r_h = np.where(has_prediction, stacked['HMD'][rows, idx_pred], np.nan)
    est_r_v = np.where(has_prediction, stacked['ZCPA'][rows, idx_pred], np.nan)
    t_pred = np.where(has_prediction, stacked['time'][idx_pred], np.nan)
    return {
        'has_prediction': has_prediction,
        'idx_pred': np.where(has_prediction, idx_pred, -1),
        't_pred': t_pred, 'est_r_h': est_r_h, 'est_r_v': est_r_v,
        'lead_time': np.where(has_prediction, stacked['t_cpa'] - t_pred, np.nan),
    }


def predicted_cpa_accuracy(stacked, dmod=DEFAULT_DMOD, zthr=DEFAULT_ZTHR, taumod=DEFAULT_TAUMOD):
    """3x3 estimated-vs-true severity crosstab for one configuration (paper
    ): does the predictive gate's extrapolated CPA receive the
    same label as the encounter's true CPA, under the SAME (dmod, zthr)?

    estimated_label = label_from_cpa(est_r_h, est_r_v, dmod, zthr) from
    predict_cpa_from_gate; true_label = label_from_cpa(r_h_min, r_v_at_cpa,
    dmod, zthr), the encounter's retrospective argmin-r_h CPA already in
    `stacked`. Encounters with no prediction are labeled 'Safe' (the gate
    never asserted a predicted violation for that configuration);
    n_no_prediction is reported separately so that contribution stays
    auditable rather than folding invisibly into the Safe/Safe cell.

    Off-diagonal cells are split by severity rank (SEVERITY_RANK): estimated
    rank > true rank is an over-alarm (the gate predicts a worse outcome
    than occurred); estimated rank < true rank is an under-alarm (the gate
    predicts a milder outcome than occurred -- the safety-relevant
    direction, since it means a genuinely severe encounter's predictive
    gate never flagged it, though the distance gate can
    still catch it independently once truly close).
    """
    pred = predict_cpa_from_gate(stacked, dmod=dmod, taumod=taumod)
    estimated = label_from_cpa(pred['est_r_h'], pred['est_r_v'], dmod, zthr)
    estimated = np.where(pred['has_prediction'], estimated, 'Safe')
    true = label_from_cpa(stacked['r_h_min'], stacked['r_v_at_cpa'], dmod, zthr)

    order = ['Safe', 'Unsafe', 'Hazardous']
    ct = pd.crosstab(pd.Series(estimated, name='estimated'), pd.Series(true, name='true'))
    ct = ct.reindex(index=order, columns=order).fillna(0).astype(int)

    est_rank = np.array([SEVERITY_RANK[s] for s in estimated])
    true_rank = np.array([SEVERITY_RANK[s] for s in true])
    n = len(estimated)
    n_correct = int(np.sum(est_rank == true_rank))
    n_over = int(np.sum(est_rank > true_rank))
    n_under = int(np.sum(est_rank < true_rank))

    finite = np.isfinite(pred['est_r_h'])
    r_h_diff = pred['est_r_h'][finite] - stacked['r_h_min'][finite]
    r_v_diff = pred['est_r_v'][finite] - stacked['r_v_at_cpa'][finite]

    return {
        'confusion': ct,
        'n': n, 'n_correct': n_correct, 'accuracy': n_correct / n if n else float('nan'),
        'n_no_prediction': int((~pred['has_prediction']).sum()),
        'n_over_alarm': n_over, 'n_under_alarm': n_under,
        'frac_over_alarm': n_over / n if n else float('nan'),
        'frac_under_alarm': n_under / n if n else float('nan'),
        'n_hazardous_missed': int(np.sum((true == 'Hazardous') & (estimated == 'Safe'))),
        'r_h_bias_ft': float(np.mean(r_h_diff)) if len(r_h_diff) else float('nan'),
        'r_h_mae_ft': float(np.mean(np.abs(r_h_diff))) if len(r_h_diff) else float('nan'),
        'r_v_bias_ft': float(np.mean(r_v_diff)) if len(r_v_diff) else float('nan'),
        'r_v_mae_ft': float(np.mean(np.abs(r_v_diff))) if len(r_v_diff) else float('nan'),
        'lead_time_mean_s': float(np.nanmean(pred['lead_time'])) if pred['has_prediction'].any() else float('nan'),
    }


def evaluate_predicted_cpa_configs(stacked, configs=None, group_labels=None):
    """predicted_cpa_accuracy across several named configurations (paper
    ), one row per configuration. Returns
    (summary_df, confusion_by_config): summary_df has one row per config;
    confusion_by_config maps each config name to its 3x3 DataFrame.

    group_labels (added in response to review) additionally reports each
    configuration within each stratum -- geometry class, closure-rate band,
    turn-rate band -- so the predictive gate's error can be attributed to
    encounter conditions instead of being reported only as a pooled average.
    summary_df then carries a 'group' column ('__all__' for the pooled row)."""
    if configs is None:
        configs = PREDICTED_CPA_CONFIGS

    rows = []
    confusion_by_config = {}
    for name, (dmod, zthr, taumod) in configs.items():
        result = predicted_cpa_accuracy(stacked, dmod=dmod, zthr=zthr, taumod=taumod)
        confusion_by_config[name] = result['confusion']
        rows.append({'config': name, 'DMOD': dmod, 'ZTHR': zthr, 'TAUMOD': taumod,
                     'group': '__all__',
                     **{k: v for k, v in result.items() if k != 'confusion'}})

        if group_labels is not None:
            groups = np.asarray(group_labels, dtype=object)
            for gval in pd.unique(groups):
                sub = _subset_stacked(stacked, np.where(groups == gval)[0])
                gres = predicted_cpa_accuracy(sub, dmod=dmod, zthr=zthr, taumod=taumod)
                rows.append({'config': name, 'DMOD': dmod, 'ZTHR': zthr, 'TAUMOD': taumod,
                             'group': gval,
                             **{k: v for k, v in gres.items() if k != 'confusion'}})

    return pd.DataFrame(rows), confusion_by_config


def stratify_by_quantile(values, n_bands=3, labels=None):
    """Split a continuous per-encounter quantity (closure rate, turn rate) into
    equal-count bands for the stratified predicted-CPA breakdown requested in
    review. Returns (band_labels, edges) so the band definitions can be reported
    alongside the results rather than left implicit."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    edges = np.nanquantile(values[finite], np.linspace(0, 1, n_bands + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    if labels is None:
        labels = [f'Q{i+1}' for i in range(n_bands)]
    out = np.full(len(values), 'undefined', dtype=object)
    for i in range(n_bands):
        out[finite & (values >= edges[i]) & (values < edges[i + 1])] = labels[i]
    return out, edges


# ---------------------------------------------------------------------------
# Robustness checks for the joint (DMOD, ZTHR, TAUMOD) configuration
# comparison: bootstrap confidence intervals, and reweighting the pooled rate
# to the population's true geometry mix.
# ---------------------------------------------------------------------------

# Population-level geometry shares: head-on
# 644,309 (64.4%), overtaking 202,987 (20.3%), crossing 152,704 (15.3%) of the
# full 1,000,000-encounter population. The 60,000-encounter sample instead
# draws an equal 20,000-each quota per geometry,
# so a raw sample-wide FPR implicitly weights each geometry equally (~1/3
# each) rather than by its true population share.
POPULATION_GEOMETRY_SHARE = {
    'head-on': 644_309 / 1_000_000,
    'overtaking': 202_987 / 1_000_000,
    'crossing': 152_704 / 1_000_000,
}


# ---------------------------------------------------------------------------
# Step 9 (post-review addition): region-based (NMAC/WCV/Safe) confusion
# evaluation. Every function above this point scores a config's own (wide)
# DMOD/ZTHR alert boundary as a binary "predicted positive" against NMAC-only
# ground truth -- an alert correctly identifying "this violates the 1000ft
# Well Clear volume" gets counted as a false positive against "this is a
# sub-500ft NMAC," even though the alert never claimed NMAC severity. The
# functions below instead evaluate the SAME LoWC criterion at two nested
# boundaries (NMAC scale, and the config's own WCV scale) and classify each
# encounter's *predicted* severity the same nested-tier way label_from_cpa
# already classifies its *true* severity -- an apples-to-apples comparison.
# Nothing above this point is modified; everything below is additive.
# ---------------------------------------------------------------------------

def evaluate_lowc_region(stacked, dmod=DEFAULT_DMOD, zthr=DEFAULT_ZTHR, taumod=DEFAULT_TAUMOD,
                          nmac_dmod=500.0, nmac_zthr=100.0):
    """Predicted region ('Hazardous'/'Unsafe'/'Safe') per encounter, for one
    (dmod, zthr, taumod) configuration, evaluating the unchanged LoWC
    criterion (evaluate_lowc_batch) at two nested boundaries sharing this
    configuration's own taumod: the fixed NMAC scale (nmac_dmod, nmac_zthr)
    and the configuration's own WCV scale (dmod, zthr).

    Well-nested by construction: LoWC(t) is monotonically non-decreasing in
    (DMOD, ZTHR) -- both r_h(t)<DMOD and (tau_mod(t;dmod)<TAUMOD AND
    HMD(t)<DMOD) get easier to satisfy as DMOD grows (tau_mod decreases in
    DMOD, per compute_tau_mod), and r_v(t)<ZTHR gets easier as ZTHR grows --
    so NMAC-scale firing implies WCV-scale firing whenever dmod>=nmac_dmod
    and zthr>=nmac_zthr, which is asserted below rather than assumed.
    """
    if dmod < nmac_dmod or zthr < nmac_zthr:
        raise ValueError(
            f"evaluate_lowc_region requires dmod>=nmac_dmod and zthr>=nmac_zthr "
            f"for the nesting guarantee to hold (got dmod={dmod}, nmac_dmod={nmac_dmod}, "
            f"zthr={zthr}, nmac_zthr={nmac_zthr})")

    wcv = evaluate_lowc_batch(stacked, dmod=dmod, zthr=zthr, taumod=taumod)
    nmac = evaluate_lowc_batch(stacked, dmod=nmac_dmod, zthr=nmac_zthr, taumod=taumod)

    assert not np.any(nmac['any_lowc'] & ~wcv['any_lowc']), (
        "nesting invariant violated: an encounter fired at NMAC scale but not "
        "at WCV scale -- should be structurally impossible given the dmod/zthr guard above")

    predicted_region = np.full(len(wcv['any_lowc']), 'Safe', dtype=object)
    predicted_region[wcv['any_lowc']] = 'Unsafe'
    predicted_region[nmac['any_lowc']] = 'Hazardous'
    return {'wcv': wcv, 'nmac': nmac, 'predicted_region': predicted_region}


def region_confusion_matrix(true_labels, predicted_region, region_order=('Hazardous', 'Unsafe', 'Safe')):
    """3x3 crosstab, index=predicted tier, columns=true tier, both ordered
    most-severe-first."""
    ct = pd.crosstab(pd.Series(predicted_region, name='predicted'),
                      pd.Series(np.asarray(true_labels, dtype=object), name='true'))
    return ct.reindex(index=region_order, columns=region_order, fill_value=0)


def region_confusion_counts(true_labels, predicted_region, positive_tiers=('Hazardous',),
                             positive_labels=('Hazardous',)):
    """Collapses a 3-way predicted region to a binary "predicted positive"
    flag (predicted_region in positive_tiers) and reuses confusion_counts'
    unmodified TP/FP/TN/FN/TPR/FPR/Precision/F1 arithmetic against the
    existing (Hazardous-only-positive) severity ground truth.

    positive_tiers=('Hazardous',) is the corrected, apples-to-apples metric:
    "predicted positive" now means the alert reached NMAC severity, not just
    WCV severity, matching what the NMAC-only ground truth actually asks.
    positive_tiers=('Hazardous','Unsafe') exactly reconstructs the OLD
    any_lowc-vs-Hazardous-only metric (predicted_region != 'Safe' iff the old
    WCV-scale any_lowc flag fired) -- used as this fix's own regression check.
    """
    predicted_positive = np.isin(np.asarray(predicted_region, dtype=object), positive_tiers)
    return confusion_counts(true_labels, predicted_positive, positive_labels=positive_labels)


def evaluate_named_configs_region(stacked, severity_labels, configs=None, group_labels=None,
                                   positive_tiers=('Hazardous',)):
    """Region-based joint configuration comparison: for each
    named config, computes the predicted region (evaluate_lowc_region) and
    both the collapsed-binary metrics (region_confusion_counts) and the full
    3x3 crosstab (region_confusion_matrix), per group if group_labels given.
    Returns (metrics_df, crosstabs): metrics_df has the same shape/columns as
    columns (config/DMOD/ZTHR/TAUMOD/group/TP/FP/TN/FN/
    TPR/FPR/Precision/F1); crosstabs maps (config_name, group_or_'__all__') to
    its 3x3 DataFrame.
    """
    if configs is None:
        configs = NAMED_CONFIGS
    severity_labels = np.asarray(severity_labels, dtype=object)
    groups = np.asarray(group_labels, dtype=object) if group_labels is not None else None
    group_values = ['__all__'] if groups is None else list(pd.unique(groups))

    rows = []
    crosstabs = {}
    for name, (dmod, zthr, taumod) in configs.items():
        predicted_region = evaluate_lowc_region(stacked, dmod=dmod, zthr=zthr, taumod=taumod)['predicted_region']
        # The Hazardous tier is fixed (500ft/100ft) regardless of (dmod, zthr), so the
        # passed-in severity_labels (e.g. from label_severity) already gives the correct
        # Hazardous/non-Hazardous split for every config -- used as-is below for the
        # metrics_df's Hazardous-tier TPR/FPR/Precision/F1. The Unsafe/Safe split DOES
        # depend on (dmod, zthr) (per label_from_cpa), so the 3x3 crosstab recomputes
        # the true region per-config from stacked['r_h_min']/['r_v_at_cpa'] rather than
        # reusing severity_labels, which would otherwise silently compare every
        # non-baseline config's predicted region against the BASELINE's Unsafe/Safe
        # boundary instead of its own.
        true_region_this_config = label_from_cpa(stacked['r_h_min'], stacked['r_v_at_cpa'], dmod, zthr)
        for gval in group_values:
            mask = np.ones(len(severity_labels), dtype=bool) if groups is None else (groups == gval)
            metrics = region_confusion_counts(severity_labels[mask], predicted_region[mask],
                                               positive_tiers=positive_tiers)
            rows.append({'config': name, 'DMOD': dmod, 'ZTHR': zthr, 'TAUMOD': taumod,
                         'group': gval, **metrics})
            crosstabs[(name, gval)] = region_confusion_matrix(true_region_this_config[mask], predicted_region[mask])

    return pd.DataFrame(rows), crosstabs


def bootstrap_named_configs_region(stacked, severity_labels, configs=None, n_boot=1000,
                                    random_state=42, positive_tiers=('Hazardous',),
                                    group_labels=None):
    """Region-based structural parallel to bootstrap_named_configs: percentile
    bootstrap 95% CI for TPR/FPR/F1/TN at each NAMED_CONFIGS entry, using the
    corrected region-based positive-prediction definition instead of the old
    any_lowc-vs-Hazardous-only one.
    """
    if configs is None:
        configs = NAMED_CONFIGS
    rng = np.random.default_rng(random_state)
    severity_labels = np.asarray(severity_labels, dtype=object)
    n = len(severity_labels)

    # group_labels (added in response to review) switches the resampling scheme
    # from simple i.i.d. resampling over all encounters to a *stratified*
    # bootstrap that resamples within each geometry stratum and holds the
    # stratum sizes fixed. Because the time-series sample is drawn on an equal
    # geometry quota, simple resampling lets the stratum mix drift between
    # replicates, mixing between-stratum composition variance into an interval
    # meant to express within-stratum sampling variance.
    strata_idx = None
    if group_labels is not None:
        groups = np.asarray(group_labels, dtype=object)
        strata_idx = [np.where(groups == g)[0] for g in pd.unique(groups)]

    metrics = ('TPR', 'FPR', 'F1', 'TN')
    records = {name: {m: [] for m in metrics} for name in configs}
    for _ in range(n_boot):
        if strata_idx is None:
            idx = rng.integers(0, n, size=n)
        else:
            idx = np.concatenate([g[rng.integers(0, len(g), size=len(g))]
                                  for g in strata_idx])
        resampled = _resample_stacked(stacked, idx)
        labels_i = severity_labels[idx]
        for name, (dmod, zthr, taumod) in configs.items():
            predicted_region = evaluate_lowc_region(resampled, dmod=dmod, zthr=zthr, taumod=taumod)['predicted_region']
            cc = region_confusion_counts(labels_i, predicted_region, positive_tiers=positive_tiers)
            for m in metrics:
                records[name][m].append(cc[m])

    rows = []
    for name, (dmod, zthr, taumod) in configs.items():
        row = {'config': name, 'DMOD': dmod, 'ZTHR': zthr, 'TAUMOD': taumod}
        for m in metrics:
            arr = np.array(records[name][m], dtype=float)
            arr = arr[np.isfinite(arr)]
            row[f'{m}_ci_low'] = float(np.percentile(arr, 2.5)) if len(arr) else float('nan')
            row[f'{m}_ci_high'] = float(np.percentile(arr, 97.5)) if len(arr) else float('nan')
        rows.append(row)
    return pd.DataFrame(rows)


def population_reweighted_fpr_region(stacked, severity_labels, geometry_labels, configs=None,
                                      weights=None, positive_tiers=('Hazardous',)):
    """Region-based structural parallel to population_reweighted_fpr: FPR
    reweighted by true population geometry share, using the corrected
    region-based positive-prediction definition.
    """
    if configs is None:
        configs = NAMED_CONFIGS
    if weights is None:
        weights = POPULATION_GEOMETRY_SHARE
    geometry_labels = np.asarray(geometry_labels, dtype=object)
    severity_labels = np.asarray(severity_labels, dtype=object)

    rows = []
    for name, (dmod, zthr, taumod) in configs.items():
        predicted_region = evaluate_lowc_region(stacked, dmod=dmod, zthr=zthr, taumod=taumod)['predicted_region']
        raw_cc = region_confusion_counts(severity_labels, predicted_region, positive_tiers=positive_tiers)
        weighted_fpr = 0.0
        per_geom = {}
        for g, w in weights.items():
            mask = geometry_labels == g
            cc_g = region_confusion_counts(severity_labels[mask], predicted_region[mask], positive_tiers=positive_tiers)
            per_geom[g] = cc_g['FPR']
            weighted_fpr += w * cc_g['FPR']
        rows.append({'config': name, 'DMOD': dmod, 'ZTHR': zthr, 'TAUMOD': taumod,
                      'fpr_sample_raw': raw_cc['FPR'],
                      'fpr_population_reweighted': weighted_fpr,
                      **{f'fpr_{g}': v for g, v in per_geom.items()}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting helpers: the quantities the manuscript tabulates directly
# ---------------------------------------------------------------------------

def lead_time_distribution(lowc_result):
    """Full alert lead-time distribution among alerted encounters.

    Lead time runs from the first alert to the encounter's own closest point
    of approach. Both can fall anywhere inside the +/-60 s analysis window,
    so the quantity is bounded by +/-120 s rather than +/-60 s.

    Negative lead times are real rather than artefacts: the LoWC criterion is
    a conjunction, so an encounter can pass its horizontal minimum while still
    vertically separated and satisfy the full criterion only later, once
    vertical separation is lost. They are reported, not discarded.
    """
    alerted = lowc_result['any_lowc']
    lead = lowc_result['lead_time'][alerted]
    if lead.size == 0:
        return {'n_alerted': 0}
    return {
        'n_alerted': int(alerted.sum()),
        'n_total': int(alerted.size),
        'alert_rate': float(alerted.mean()),
        'lead_mean': float(np.mean(lead)),
        'lead_p10': float(np.percentile(lead, 10)),
        'lead_p50': float(np.percentile(lead, 50)),
        'lead_p90': float(np.percentile(lead, 90)),
        'lead_min': float(np.min(lead)),
        'lead_max': float(np.max(lead)),
        'frac_negative_lead': float(np.mean(lead < 0)),
    }


def peak_closing_speed(stacked):
    """Largest horizontal closing speed before CPA, per encounter, in ft/s.

    `d_range_h` is the signed range rate, so closing is negative; this returns
    its peak magnitude over the approach only, ignoring what happens after the
    aircraft have passed.
    """
    n = len(stacked['r_h_min'])
    out = np.empty(n, dtype=float)
    for k in range(n):
        upto = int(stacked['idx_cpa'][k]) + 1
        out[k] = np.nanmax(-stacked['d_range_h'][k, :upto])
    return out


def false_positive_composition(stacked, configs=None):
    """What each configuration's NMAC-severity false positives actually were.

    An alert that fails the NMAC test has not necessarily failed at its job:
    it may have flagged a genuine Well Clear violation. This decomposes the
    false positives by the encounter's true severity, judged against **that
    configuration's own** Well Clear volume -- the volume it is actually
    alerting to, not a fixed external one.

    Well Clear violation is deliberately not reported as a positive class.
    Every configuration flags every violation of its own volume, for the same
    reason the NMAC true-positive rate is 1.000: if the true CPA lies inside
    the volume then at the closest-approach timestep r_h < DMOD and
    r_v < ZTHR, so both gates are satisfied. `wcv_detection_own_volume` is
    returned anyway, so that degeneracy is visible in the output and can be
    asserted rather than taken on trust.
    """
    configs = configs or NAMED_CONFIGS
    n = len(stacked['r_h_min'])
    rows = []
    for name, (dmod, zthr, taumod) in configs.items():
        region = evaluate_lowc_region(stacked, dmod=dmod, zthr=zthr,
                                      taumod=taumod)['predicted_region']
        true_region = label_from_cpa(stacked['r_h_min'], stacked['r_v_at_cpa'],
                                     dmod, zthr)

        predicted_nmac = region == 'Hazardous'
        tp = int(np.sum(predicted_nmac & (true_region == 'Hazardous')))
        fp_wcv = int(np.sum(predicted_nmac & (true_region == 'Unsafe')))
        fp_safe = int(np.sum(predicted_nmac & (true_region == 'Safe')))
        fp_total = fp_wcv + fp_safe

        positive = ('Hazardous', 'Unsafe')
        true_wcv = np.isin(true_region, positive)
        predicted_wcv = np.isin(region, positive)

        rows.append({
            'config': name, 'DMOD': dmod, 'ZTHR': zthr, 'TAUMOD': taumod,
            'n_alerts_nmac': tp + fp_total,
            'tp_nmac': tp,
            'fp_total': fp_total,
            'fp_genuine_wcv': fp_wcv,
            'fp_truly_safe': fp_safe,
            'frac_fp_genuine_wcv': fp_wcv / fp_total if fp_total else float('nan'),
            'safe_alert_rate': fp_safe / n if n else float('nan'),
            'wcv_detection_own_volume':
                float(np.sum(true_wcv & predicted_wcv) / np.sum(true_wcv))
                if true_wcv.any() else float('nan'),
        })
    return pd.DataFrame(rows)


def predicted_cpa_accuracy_engaged(stacked, configs=None):
    """Predicted-CPA labelling accuracy, pooled and restricted to engagement.

    The pooled figure counts encounters the predictive gate never engaged on
    as correctly labelled Safe. Engagement varies sharply with configuration,
    so the pooled column is not a like-for-like comparison across rows -- it
    fills with easy cases as the volume tightens. `acc_engaged` restricts to
    encounters where the tau_mod branch actually engaged and is the honest
    basis for comparing configurations.

    `frac_nopred_true_safe` records whether the no-prediction default is ever
    wrong; it is 1.0 throughout this dataset, so the default introduces no
    error of its own.
    """
    configs = configs or NAMED_CONFIGS
    rows = []
    for name, (dmod, zthr, taumod) in configs.items():
        pred = predict_cpa_from_gate(stacked, dmod=dmod, taumod=taumod)
        engaged = pred['has_prediction']
        estimated = label_from_cpa(pred['est_r_h'], pred['est_r_v'], dmod, zthr)
        true = label_from_cpa(stacked['r_h_min'], stacked['r_v_at_cpa'], dmod, zthr)

        est_rank = np.array([SEVERITY_RANK[x] for x in estimated])
        true_rank = np.array([SEVERITY_RANK[x] for x in true])
        pooled_rank = np.array([SEVERITY_RANK[x] for x in
                                np.where(engaged, estimated, 'Safe')])

        finite = engaged & np.isfinite(pred['est_r_h'])
        err = pred['est_r_h'][finite] - stacked['r_h_min'][finite]
        no_pred = ~engaged

        rows.append({
            'config': name, 'DMOD': dmod, 'ZTHR': zthr, 'TAUMOD': taumod,
            'n_engaged': int(engaged.sum()),
            'frac_engaged': float(engaged.mean()),
            'acc_pooled': float(np.mean(pooled_rank == true_rank)),
            'acc_engaged': float(np.mean(est_rank[engaged] == true_rank[engaged]))
            if engaged.any() else float('nan'),
            'over_alarm_engaged': float(np.mean(est_rank[engaged] > true_rank[engaged]))
            if engaged.any() else float('nan'),
            'under_alarm_engaged': float(np.mean(est_rank[engaged] < true_rank[engaged]))
            if engaged.any() else float('nan'),
            'bias_engaged_ft': float(np.mean(err)) if finite.any() else float('nan'),
            'mae_engaged_ft': float(np.mean(np.abs(err))) if finite.any() else float('nan'),
            'frac_nopred_true_safe': float(np.mean(true[no_pred] == 'Safe'))
            if no_pred.any() else float('nan'),
        })
    return pd.DataFrame(rows)


def fpr_vs_taumod(stacked, severity_labels, taumod_grid=None, dmod_probe=None,
                  zthr=DEFAULT_ZTHR):
    """NMAC-severity false-positive rate against TAUMOD, at several DMODs.

    Scoring happens at the fixed NMAC scale, which the configuration's own
    DMOD and ZTHR never enter, so the false-positive rate should depend on
    TAUMOD alone. Evaluating across a range of DMOD values tests that rather
    than asserting it: `max_fpr_spread_across_dmod` in the returned summary is
    the largest disagreement found, and is expected to be zero.
    """
    taumod_grid = np.arange(0.0, 46.0, 5.0) if taumod_grid is None else np.asarray(taumod_grid)
    dmod_probe = (500.0, 1000.0, 4000.0, 6000.0) if dmod_probe is None else tuple(dmod_probe)

    rows = []
    for taumod in taumod_grid:
        row = {'TAUMOD': float(taumod)}
        for dmod in dmod_probe:
            region = evaluate_lowc_region(stacked, dmod=dmod, zthr=zthr,
                                          taumod=float(taumod))['predicted_region']
            counts = region_confusion_counts(severity_labels, region)
            row[f'FPR_dmod{int(dmod)}'] = counts['FPR']
            row[f'TPR_dmod{int(dmod)}'] = counts['TPR']
        rows.append(row)

    curve = pd.DataFrame(rows)
    fpr_cols = [c for c in curve.columns if c.startswith('FPR_')]
    tpr_cols = [c for c in curve.columns if c.startswith('TPR_')]
    summary = {
        'max_fpr_spread_across_dmod':
            float((curve[fpr_cols].max(axis=1) - curve[fpr_cols].min(axis=1)).max()),
        'fpr_min': float(curve[fpr_cols].min().min()),
        'fpr_max': float(curve[fpr_cols].max().max()),
        'tpr_always_one': bool((curve[tpr_cols] == 1.0).all().all()),
    }
    return curve, summary


def smoothing_sensitivity(raw_aligned, settings=((7, 3), (5, 2), (11, 3), (9, 4)),
                          configs=None):
    """Repeat the evaluation under alternative Savitzky-Golay settings.

    `raw_aligned` must be aligned but NOT yet differentiated, since each
    setting re-runs `estimate_derivatives` on its own copy. The smoothing
    parameters are analyst choices, and this reports how far the headline
    results move when they change.
    """
    import copy

    from lltem_preprocessing import estimate_derivatives

    configs = configs or NAMED_CONFIGS
    rows = []
    for window, order in settings:
        smoothed = [estimate_derivatives(copy.deepcopy(a), window=window, polyorder=order)
                    for a in raw_aligned]
        stacked, severity, _ = build_sample_metrics(smoothed)
        for name, (dmod, zthr, taumod) in configs.items():
            region = evaluate_lowc_region(stacked, dmod=dmod, zthr=zthr,
                                          taumod=taumod)['predicted_region']
            counts = region_confusion_counts(severity, region)
            rows.append({'sg_window': window, 'sg_order': order, 'config': name,
                         'n': int(len(severity)),
                         **{k: counts[k] for k in
                            ('TP', 'FP', 'TN', 'FN', 'FPR', 'Precision')}})
    return pd.DataFrame(rows)
