# -*- coding: utf-8 -*-
"""
Logit Analysis — Logit Distributions, Threshold Sweep, Calibration
══════════════════════════════════════════════════════════════════════

Implements Objective 3:
  - Logit/probability distribution analysis per label (gold=0 vs gold=1)
  - Threshold sweep for expression modes (Pareto: F1 vs violation rate)
  - Calibration analysis (reliability diagrams, Expected Calibration Error)
"""

import numpy as np
import pandas as pd

from . import config


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _proba_to_logit(p, eps=1e-7):
    """Convert probability to logit (log-odds), clamping to avoid ±inf."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _compute_f1_arrays(gold, pred):
    """F1 from numpy arrays."""
    tp = ((gold == 1) & (pred == 1)).sum()
    fp = ((gold == 0) & (pred == 1)).sum()
    fn = ((gold == 1) & (pred == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  LOGIT DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def logit_distribution_analysis(df):
    """
    For each label, analyse the logit/probability distribution conditioned
    on gold=0 vs gold=1.

    Metrics computed per label:
        - logit_separation: mean(logit|gold=1) - mean(logit|gold=0)
        - proba_overlap: overlap coefficient between the two proba distributions
        - mean/std of probabilities for positive and negative gold classes

    Returns
    -------
    pd.DataFrame
        One row per label with distribution statistics.
    """
    all_labels = config.EMOTION_12 + config.MODES_4 + config.META_LABELS + config.TYPES_2
    results = []

    for label in all_labels:
        proba_col = f"proba_{label}"
        if proba_col not in df.columns or label not in df.columns:
            continue

        probas = df[proba_col].values.astype(float)
        golds = df[label].values.astype(int)

        mask_pos = golds == 1
        mask_neg = golds == 0

        n_pos = mask_pos.sum()
        n_neg = mask_neg.sum()

        if n_pos == 0 or n_neg == 0:
            results.append({
                "label": label,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "logit_separation": np.nan,
                "proba_mean_pos": np.nan,
                "proba_mean_neg": np.nan,
                "proba_std_pos": np.nan,
                "proba_std_neg": np.nan,
                "proba_overlap": np.nan,
            })
            continue

        # Logit analysis
        logits = _proba_to_logit(probas)
        logit_mean_pos = logits[mask_pos].mean()
        logit_mean_neg = logits[mask_neg].mean()
        logit_sep = logit_mean_pos - logit_mean_neg

        # Probability distribution statistics
        proba_mean_pos = probas[mask_pos].mean()
        proba_mean_neg = probas[mask_neg].mean()
        proba_std_pos = probas[mask_pos].std()
        proba_std_neg = probas[mask_neg].std()

        # Overlap coefficient (histogram-based)
        bins = np.linspace(0, 1, 51)
        hist_pos, _ = np.histogram(probas[mask_pos], bins=bins, density=True)
        hist_neg, _ = np.histogram(probas[mask_neg], bins=bins, density=True)
        # Normalize to probability distribution
        bin_width = bins[1] - bins[0]
        hist_pos_prob = hist_pos * bin_width
        hist_neg_prob = hist_neg * bin_width
        overlap = np.minimum(hist_pos_prob, hist_neg_prob).sum()

        results.append({
            "label": label,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "logit_separation": round(logit_sep, 4),
            "logit_mean_pos": round(logit_mean_pos, 4),
            "logit_mean_neg": round(logit_mean_neg, 4),
            "proba_mean_pos": round(proba_mean_pos, 4),
            "proba_mean_neg": round(proba_mean_neg, 4),
            "proba_std_pos": round(proba_std_pos, 4),
            "proba_std_neg": round(proba_std_neg, 4),
            "proba_overlap": round(overlap, 4),
        })

    logit_df = pd.DataFrame(results)

    if len(logit_df) > 0:
        print(f"\n  ═══ Logit Distribution Analysis ({len(logit_df)} labels) ═══")
        print(f"  {'Label':<18s} {'n+':<5s} {'n-':<5s} {'sep':>6s} "
              f"{'p̄(+)':>7s} {'p̄(-)':>7s} {'overlap':>8s}")
        for _, r in logit_df.iterrows():
            sep = f"{r['logit_separation']:+.2f}" if not np.isnan(r['logit_separation']) else "  N/A"
            print(f"  {r['label']:<18s} {r['n_pos']:<5d} {r['n_neg']:<5d} {sep:>6s} "
                  f"{r['proba_mean_pos']:>7.3f} {r['proba_mean_neg']:>7.3f} "
                  f"{r['proba_overlap']:>8.3f}")

    return logit_df


# ═══════════════════════════════════════════════════════════════════════════
#  THRESHOLD SWEEP FOR MODES
# ═══════════════════════════════════════════════════════════════════════════

def threshold_sweep_modes(df, thresholds=None):
    """
    For each expression mode, sweep the decision threshold and compute:
      - F1 at each threshold
      - Annotation scheme violation rate (emotion_no_mode)
      - Precision and recall

    This finds the Pareto-optimal threshold that maximizes F1 while
    minimizing annotation scheme violations.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain gold mode columns, proba_* columns, and
        gold/pred emotion columns.
    thresholds : array-like, optional
        Threshold values to sweep. Default: 0.05 to 0.95 step 0.01.

    Returns
    -------
    dict with keys:
        "sweep_results" : pd.DataFrame
            Full sweep results (mode, threshold, f1, precision, recall,
            violation_rate)
        "optimal_thresholds" : dict
            {mode: optimal_threshold} — maximizes F1 among thresholds
            where violation_rate < current_violation_rate
        "pareto_front" : dict
            {mode: [(threshold, f1, violation_rate), ...]}
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    modes = [m for m in config.MODES_4
             if m in df.columns and f"proba_{m}" in df.columns]

    if not modes:
        print("  ⚠ Colonnes mode/proba manquantes. Threshold sweep sauté.")
        return {}

    # Check which emotions are active per sample (gold)
    emotions = [e for e in config.EMOTION_12 if e in df.columns]
    if emotions:
        any_emotion_gold = df[emotions].sum(axis=1) > 0
    else:
        any_emotion_gold = pd.Series(False, index=df.index)

    sweep_rows = []
    optimal_thresholds = {}
    pareto_fronts = {}

    for mode in modes:
        gold = df[mode].values.astype(int)
        probas = df[f"proba_{mode}"].values.astype(float)

        # Current violation rate at threshold 0.5
        pred_05 = (probas >= 0.5).astype(int)
        any_mode_05 = pred_05  # single mode contributes to any_mode
        current_viol = _compute_violation_rate(df, mode, 0.5, emotions.copy())

        best_f1 = -1
        best_thr = 0.5

        for thr in thresholds:
            pred = (probas >= thr).astype(int)

            f1 = _compute_f1_arrays(gold, pred)
            tp = ((gold == 1) & (pred == 1)).sum()
            fp = ((gold == 0) & (pred == 1)).sum()
            fn = ((gold == 1) & (pred == 0)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # Violation rate: samples with emotion but no mode active
            viol_rate = _compute_violation_rate(df, mode, thr, emotions.copy())

            sweep_rows.append({
                "mode": mode,
                "threshold": round(thr, 3),
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "violation_rate": round(viol_rate, 4),
            })

            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        optimal_thresholds[mode] = round(best_thr, 3)

        # Pareto front: non-dominated (f1, -violation_rate) solutions
        mode_sweep = [r for r in sweep_rows if r["mode"] == mode]
        pareto = _extract_pareto(mode_sweep)
        pareto_fronts[mode] = pareto

    sweep_df = pd.DataFrame(sweep_rows)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n  ═══ Threshold Sweep for Modes ═══")
    for mode in modes:
        opt_thr = optimal_thresholds.get(mode, 0.5)
        mode_data = sweep_df[sweep_df["mode"] == mode]
        at_05 = mode_data[mode_data["threshold"] == 0.5]
        at_opt = mode_data[
            (mode_data["threshold"] - opt_thr).abs() < 0.005
        ]

        f1_05 = at_05["f1"].values[0] if len(at_05) > 0 else np.nan
        viol_05 = at_05["violation_rate"].values[0] if len(at_05) > 0 else np.nan
        f1_opt = at_opt["f1"].values[0] if len(at_opt) > 0 else np.nan
        viol_opt = at_opt["violation_rate"].values[0] if len(at_opt) > 0 else np.nan

        print(f"\n  {mode}:")
        print(f"    @0.5:    F1={f1_05:.3f}  violations={viol_05:.3f}")
        print(f"    @opt({opt_thr:.2f}): F1={f1_opt:.3f}  "
              f"violations={viol_opt:.3f}")

    return {
        "sweep_results": sweep_df,
        "optimal_thresholds": optimal_thresholds,
        "pareto_fronts": pareto_fronts,
    }


def _compute_violation_rate(df, mode, threshold, emotions):
    """Compute the 'emotion without any mode' violation rate
    when the given mode uses a custom threshold."""
    # Predict modes with current thresholds, overriding this mode
    all_modes = config.MODES_4
    any_mode_active = np.zeros(len(df), dtype=bool)

    for m in all_modes:
        proba_col = f"proba_{m}"
        if proba_col not in df.columns:
            continue
        if m == mode:
            pred = df[proba_col].values >= threshold
        else:
            pred = df[proba_col].values >= config.DEFAULT_THRESHOLD
        any_mode_active |= pred

    # Any emotion active in gold
    any_emotion = np.zeros(len(df), dtype=bool)
    for e in emotions:
        if e in df.columns:
            any_emotion |= (df[e].values.astype(int) == 1)

    # Violation: emotion present but no mode predicted
    violations = any_emotion & ~any_mode_active
    return violations.sum() / len(df) if len(df) > 0 else 0.0


def _extract_pareto(sweep_points):
    """Extract Pareto-optimal points (maximize F1, minimize violation_rate)."""
    # Sort by F1 descending
    sorted_pts = sorted(sweep_points, key=lambda x: -x["f1"])
    pareto = []
    best_viol = float("inf")
    for pt in sorted_pts:
        if pt["violation_rate"] < best_viol:
            pareto.append((pt["threshold"], pt["f1"], pt["violation_rate"]))
            best_viol = pt["violation_rate"]
    return pareto


# ═══════════════════════════════════════════════════════════════════════════
#  CALIBRATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def calibration_analysis(df, label_list=None, n_bins=10):
    """
    Compute calibration data for reliability diagrams.

    For each label, bins predicted probabilities and computes the observed
    frequency of positives within each bin.

    Parameters
    ----------
    df : pd.DataFrame
    label_list : list[str], optional
        Labels to analyze. Default: EMOTION_12 + MODES_4
    n_bins : int

    Returns
    -------
    dict
        {label: {"bin_midpoints": [...], "observed_freq": [...],
                 "mean_pred": [...], "bin_count": [...], "ece": float}}
    """
    if label_list is None:
        label_list = config.EMOTION_12 + config.MODES_4

    calibration_data = {}

    for label in label_list:
        proba_col = f"proba_{label}"
        if proba_col not in df.columns or label not in df.columns:
            continue

        probas = df[proba_col].values.astype(float)
        golds = df[label].values.astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probas, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_midpoints = []
        observed_freq = []
        mean_pred = []
        bin_count = []
        ece = 0.0

        for k in range(n_bins):
            mask = bin_indices == k
            n_k = mask.sum()
            mid = (bin_edges[k] + bin_edges[k + 1]) / 2
            bin_midpoints.append(mid)
            bin_count.append(n_k)

            if n_k == 0:
                observed_freq.append(np.nan)
                mean_pred.append(np.nan)
                continue

            obs = golds[mask].mean()
            pred_mean = probas[mask].mean()
            observed_freq.append(obs)
            mean_pred.append(pred_mean)
            ece += n_k * abs(pred_mean - obs)

        ece /= len(probas)

        calibration_data[label] = {
            "bin_midpoints": bin_midpoints,
            "observed_freq": observed_freq,
            "mean_pred": mean_pred,
            "bin_count": bin_count,
            "ece": round(ece, 6),
        }

    # Summary
    print(f"\n  ═══ Calibration Summary ═══")
    emo_eces = []
    mode_eces = []
    for label, data in calibration_data.items():
        group = "mode" if label in config.MODES_4 else "emotion"
        print(f"  {label:<18s}  ECE={data['ece']:.4f}  [{group}]")
        if group == "mode":
            mode_eces.append(data["ece"])
        else:
            emo_eces.append(data["ece"])

    if emo_eces:
        print(f"\n  Mean ECE (emotions): {np.mean(emo_eces):.4f}")
    if mode_eces:
        print(f"  Mean ECE (modes):    {np.mean(mode_eces):.4f}")

    return calibration_data
