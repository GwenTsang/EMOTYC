# -*- coding: utf-8 -*-
"""
Explainability — RF + SHAP, Univariate, Bivariate, Association Rules
═════════════════════════════════════════════════════════════════════

Preserved from the original error_analysis.py with plotting code
moved to visualization.py. Functions return data; they do not plot.
"""

import numpy as np
import pandas as pd
from scipy import stats

from . import config


# ═══════════════════════════════════════════════════════════════════════════
#  UNIVARIATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def univariate_analysis(df, metric="hamming_12"):
    """
    For each explanatory feature, compares the distribution of the
    error metric between its levels. Statistical test + effect size.

    Parameters
    ----------
    df : pd.DataFrame
    metric : str
        Error metric column.

    Returns
    -------
    list[dict]
        Sorted by p-value (ascending). Each dict has: feature, test,
        statistic, p_value, eta_squared, n_levels, levels.
    """
    if metric not in df.columns:
        return []

    results = []
    all_features = config.BINARY_FEATURES + config.QUALITATIVE_FEATURES + ["domain"]

    for feat in all_features:
        if feat not in df.columns:
            continue

        col = df[feat].copy()
        if feat in config.BINARY_FEATURES:
            col = col.astype(str)
        else:
            col = col.astype(str).replace("nan", "MISSING")

        # Group by level
        groups = {}
        for level in col.unique():
            mask = col == level
            vals = df.loc[mask, metric].dropna()
            if len(vals) >= 3:
                groups[level] = vals

        if len(groups) < 2:
            continue

        # Statistical test
        group_arrays = list(groups.values())
        if len(groups) == 2:
            stat_val, p_val = stats.mannwhitneyu(
                *group_arrays, alternative="two-sided"
            )
            test_name = "Mann-Whitney U"
        else:
            stat_val, p_val = stats.kruskal(*group_arrays)
            test_name = "Kruskal-Wallis"

        # Effect size: eta² for Kruskal-Wallis
        n_total_kw = sum(len(g) for g in group_arrays)
        eta_sq = (
            (stat_val - len(groups) + 1) / (n_total_kw - len(groups))
            if n_total_kw > len(groups) else 0
        )

        # Stats per level
        level_stats = []
        for level, vals in sorted(groups.items(), key=lambda x: -x[1].mean()):
            level_stats.append({
                "level": level,
                "n": len(vals),
                "mean_error": round(vals.mean(), 4),
                "median_error": round(vals.median(), 4),
                "std_error": round(vals.std(), 4),
            })

        results.append({
            "feature": feat,
            "test": test_name,
            "statistic": round(stat_val, 4),
            "p_value": p_val,
            "eta_squared": round(max(eta_sq, 0), 4),
            "n_levels": len(groups),
            "levels": level_stats,
        })

    results.sort(key=lambda r: r["p_value"])

    print(f"\n  ═══ Analyse Univariée (top 10) ═══")
    for r in results[:10]:
        sig = (
            "***" if r["p_value"] < 0.001 else
            "**" if r["p_value"] < 0.01 else
            "*" if r["p_value"] < 0.05 else "ns"
        )
        print(f"  {r['feature']:<25s}  {r['test']:<18s}  "
              f"p={r['p_value']:.2e} {sig}  η²={r['eta_squared']:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  BIVARIATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def bivariate_analysis(df, metric="hamming_12", top_n_pairs=10):
    """
    Analyses interaction between pairs of features on the error metric.

    Returns
    -------
    list[dict]
        Sorted by error_range (descending). Each dict has: f1, f2,
        interaction_var, error_range, max_error, min_error, n_cells.
    """
    if metric not in df.columns:
        return []

    cat_features = []
    for feat in config.QUALITATIVE_FEATURES + ["domain"]:
        if feat in df.columns:
            col = (
                df[feat].astype(str).replace("nan", "MISSING")
                if feat != "domain" else df[feat].astype(str)
            )
            n_levels = col.nunique()
            if 2 <= n_levels <= 8:
                cat_features.append(feat)

    # Add binary features with variance
    for feat in config.BINARY_FEATURES:
        if feat in df.columns and df[feat].nunique() >= 2:
            cat_features.append(feat)

    interaction_scores = []

    for i, f1 in enumerate(cat_features):
        for f2 in cat_features[i + 1:]:
            col1 = (
                df[f1].astype(str).replace("nan", "MISSING")
                if f1 not in config.BINARY_FEATURES
                else df[f1].astype(str)
            )
            col2 = (
                df[f2].astype(str).replace("nan", "MISSING")
                if f2 not in config.BINARY_FEATURES
                else df[f2].astype(str)
            )

            combined = pd.DataFrame({
                "f1": col1, "f2": col2, "error": df[metric]
            }).dropna(subset=["error"])
            ct = combined.groupby(["f1", "f2"])["error"].agg(["mean", "count"])
            ct = ct[ct["count"] >= 5]

            if len(ct) < 4:
                continue

            cell_means = ct["mean"]
            interaction_var = cell_means.var()
            max_error = cell_means.max()
            min_error = cell_means.min()
            error_range = max_error - min_error

            interaction_scores.append({
                "f1": f1, "f2": f2,
                "interaction_var": round(interaction_var, 6),
                "error_range": round(error_range, 4),
                "max_error": round(max_error, 4),
                "min_error": round(min_error, 4),
                "n_cells": len(ct),
            })

    interaction_scores.sort(key=lambda x: -x["error_range"])

    print(f"\n  ═══ Analyse Bivariée — Top {min(top_n_pairs, len(interaction_scores))} paires ═══")
    for r in interaction_scores[:top_n_pairs]:
        print(f"  {r['f1']:<20s} × {r['f2']:<20s}  "
              f"range={r['error_range']:.3f}  max={r['max_error']:.3f}  "
              f"min={r['min_error']:.3f}")

    return interaction_scores


# ═══════════════════════════════════════════════════════════════════════════
#  RANDOM FOREST + SHAP
# ═══════════════════════════════════════════════════════════════════════════

def rf_shap_analysis(df, X_df, feature_names, metric="hamming_12"):
    """
    Trains a Random Forest Regressor to predict the error metric from
    explanatory features, then uses SHAP for interpretation.

    Parameters
    ----------
    df : pd.DataFrame
    X_df : pd.DataFrame
        Feature matrix (from data_loader.build_analysis_features).
    feature_names : list[str]
    metric : str

    Returns
    -------
    rf_model : RandomForestRegressor
    shap_values : np.ndarray or None
    feature_names : list[str]
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeRegressor, export_text

    if metric not in df.columns:
        print(f"  ⚠ Métrique '{metric}' non trouvée.")
        return None, None, feature_names

    y = df[metric].values
    X = X_df.values

    # ── Random Forest ─────────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=8, min_samples_leaf=10,
        random_state=42, n_jobs=-1, oob_score=True,
    )
    rf.fit(X, y)

    # Cross-validation
    cv_r2 = cross_val_score(rf, X, y, cv=5, scoring="r2")
    cv_mae = cross_val_score(rf, X, y, cv=5, scoring="neg_mean_absolute_error")

    print(f"\n  ═══ Random Forest Regressor ═══")
    print(f"  OOB R²         : {rf.oob_score_:.4f}")
    print(f"  CV R² (5-fold) : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  CV MAE (5-fold): {-cv_mae.mean():.4f} ± {cv_mae.std():.4f}")

    # Feature importance (MDI)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n  Top 20 features (MDI importance) :")
    for rank, idx in enumerate(sorted_idx[:20]):
        print(f"    {rank+1:2d}. {feature_names[idx]:<35s}  {importances[idx]:.4f}")

    # Store feature names on the model for the report
    rf.feature_names_in_ = np.array(feature_names)

    # ── Decision Tree (interpretable) ─────────────────────────────────
    dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=15, random_state=42)
    dt.fit(X, y)
    dt.tree_text = export_text(dt, feature_names=feature_names, max_depth=4)

    # ── SHAP ──────────────────────────────────────────────────────────
    shap_values = None
    try:
        import shap

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_df)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_rank = np.argsort(mean_abs_shap)[::-1]
        print(f"\n  Top 15 features (mean |SHAP|) :")
        for rank, idx in enumerate(shap_rank[:15]):
            print(f"    {rank+1:2d}. {feature_names[idx]:<35s}  "
                  f"{mean_abs_shap[idx]:.4f}")

    except ImportError:
        print("  ⚠ Bibliothèque 'shap' non installée. "
              "Installez-la : pip install shap")
        print("    L'analyse SHAP est sautée.")

    return rf, shap_values, feature_names, dt


# ═══════════════════════════════════════════════════════════════════════════
#  ASSOCIATION RULE MINING
# ═══════════════════════════════════════════════════════════════════════════

def association_rule_analysis(df, metric="hamming_12",
                              min_support=0.08, min_confidence=0.5,
                              min_lift=1.2, top_k=30):
    """
    FP-Growth on high-error sub-populations.
    Discovers frequent itemsets associated with model failures.

    Returns
    -------
    rules_or_itemsets : pd.DataFrame or None
    """
    try:
        from mlxtend.frequent_patterns import fpgrowth, association_rules
    except ImportError:
        print("  ⚠ Bibliothèque 'mlxtend' non installée. "
              "Installez-la : pip install mlxtend")
        return None

    if metric not in df.columns:
        return None

    # Binarize features
    items = pd.DataFrame(index=df.index)

    for col in config.BINARY_FEATURES:
        if col in df.columns:
            items[f"{col}=1"] = (df[col] == 1).astype(bool)

    for col in config.QUALITATIVE_FEATURES + ["domain"]:
        if col not in df.columns:
            continue
        series = (
            df[col].astype(str).replace("nan", "MISSING")
            if col != "domain" else df[col].astype(str)
        )
        for level in series.unique():
            if level == "MISSING":
                continue
            items[f"{col}={level}"] = (series == level).astype(bool)

    if "text_length" in df.columns:
        q33, q66 = df["text_length"].quantile([0.33, 0.66])
        items["text_court"] = (df["text_length"] <= q33).astype(bool)
        items["text_long"] = (df["text_length"] >= q66).astype(bool)

    # ── HIGH ERROR subset ─────────────────────────────────────────────
    median_error = df[metric].median()
    threshold = max(median_error, 1.0 / 12)
    high_error_mask = df[metric] > threshold
    n_high = high_error_mask.sum()

    print(f"\n  ═══ Association Rules (FP-Growth) ═══")
    print(f"  Seuil erreur     : {metric} > {threshold:.4f}")
    print(f"  Sous-pop HIGH    : {n_high} / {len(df)} ({100*n_high/len(df):.1f}%)")

    items_high = items[high_error_mask]

    if len(items_high) < 10:
        print("  ⚠ Trop peu d'échantillons en erreur pour l'analyse par règles.")
        return None

    # FP-Growth
    freq_itemsets = fpgrowth(items_high, min_support=min_support, use_colnames=True)
    if len(freq_itemsets) == 0:
        print(f"  ⚠ Aucun itemset fréquent avec min_support={min_support}. "
              "Réduction à 0.05.")
        freq_itemsets = fpgrowth(items_high, min_support=0.05, use_colnames=True)

    if len(freq_itemsets) == 0:
        print("  ⚠ Toujours aucun itemset fréquent.")
        return None

    print(f"  Itemsets fréquents trouvés : {len(freq_itemsets)}")

    # Association rules
    try:
        rules = association_rules(
            freq_itemsets, metric="confidence",
            min_threshold=min_confidence, num_itemsets=len(freq_itemsets)
        )
    except Exception:
        rules = association_rules(
            freq_itemsets, metric="confidence",
            min_threshold=min_confidence
        )

    if len(rules) == 0:
        print(f"  ⚠ Aucune règle avec confidence >= {min_confidence}")
        freq_itemsets = freq_itemsets.sort_values("support", ascending=False)
        print(f"\n  Top itemsets fréquents (sous-population HIGH ERROR) :")
        for _, row in freq_itemsets.head(top_k).iterrows():
            items_str = " ∧ ".join(sorted(row["itemsets"]))
            print(f"    support={row['support']:.3f}  {items_str}")
        return freq_itemsets

    rules = rules[rules["lift"] >= min_lift]
    rules = rules.sort_values("lift", ascending=False)

    print(f"  Règles (lift >= {min_lift}) : {len(rules)}")
    print(f"\n  Top {min(top_k, len(rules))} règles (HIGH ERROR) :")
    for idx, row in rules.head(top_k).iterrows():
        ant = " ∧ ".join(sorted(row["antecedents"]))
        cons = " ∧ ".join(sorted(row["consequents"]))
        print(f"    [{ant}] → [{cons}]")
        print(f"      support={row['support']:.3f}  "
              f"conf={row['confidence']:.3f}  lift={row['lift']:.2f}")

    # ── LOW ERROR analysis (for contrast) ─────────────────────────────
    low_mask = df[metric] == 0
    items_low = items[low_mask]
    n_low = low_mask.sum()
    print(f"\n  Sous-pop EXACT MATCH : {n_low} / {len(df)} "
          f"({100*n_low/len(df):.1f}%)")

    if n_low >= 10:
        freq_low = fpgrowth(items_low, min_support=min_support, use_colnames=True)
        if len(freq_low) > 0:
            freq_low = freq_low.sort_values("support", ascending=False)
            print(f"  Top itemsets (EXACT MATCH) :")
            for _, row in freq_low.head(15).iterrows():
                items_str = " ∧ ".join(sorted(row["itemsets"]))
                print(f"    support={row['support']:.3f}  {items_str}")

    return rules
