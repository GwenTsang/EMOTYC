#!/usr/bin/env python3
"""
EMOTYC Error Analysis Pipeline — Orchestrator
═══════════════════════════════════════════════

Thin orchestrator that calls specialized sub-modules for deep error
analysis of the EMOTYC model on OOD cyberbullying data.

Modules (in experimentations/analysis/):
    config          — Constants, label mappings, thresholds
    data_loader     — Data loading, cleaning, feature engineering
    inference       — EMOTYC model inference + cached predictions
    metrics         — Error metrics (Hamming, Jaccard, Brier, violations)
    conditional     — Conditional error analysis (modes ↔ emotions)
    logit_analysis  — Logit distributions, threshold sweep, calibration
    stratification  — Density / length / domain stratification
    explainability  — RF + SHAP, univariate, bivariate, association rules
    visualization   — All plotting functions
    report          — Report generation

Usage:
    # Full pipeline (inference + analysis) — optimized thresholds, no context
    python experimentations/error_analysis.py

    # Fixed 0.5 thresholds
    python experimentations/error_analysis.py --no-optimized-thresholds

    # Skip inference (use pre-computed JSONL)
    python experimentations/error_analysis.py --skip-inference --predictions-dir outputs

    # Load from pre-computed CSV (skips inference + metric computation)
    python experimentations/error_analysis.py --from-csv error_analysis_results/analysis_data.csv

Dependencies:
    numpy, pandas, scipy, matplotlib, seaborn, scikit-learn
    Optional: shap, mlxtend (for SHAP and association rules)
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Import analysis sub-modules ──────────────────────────────────────────
from analysis import config
from analysis import data_loader
from analysis import inference
from analysis import metrics
from analysis import conditional
from analysis import logit_analysis
from analysis import stratification
from analysis import explainability
from analysis import visualization
from analysis import report


def _parse_domain_path(value):
    """Parse --xlsx DOMAIN=PATH entries."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Expected DOMAIN=PATH, for example: Homophobie=data/homophobie.xlsx"
        )
    domain, path = value.split("=", 1)
    domain = config.canonicalize_domain_name(domain)
    if domain not in config.XLSX_PATHS:
        raise argparse.ArgumentTypeError(
            f"Unknown domain '{domain}'. Expected one of: {', '.join(config.XLSX_PATHS)}"
        )
    return domain, path


def _build_xlsx_paths_from_args(args):
    """Resolve input XLSX paths from CLI arguments."""
    overrides = dict(args.xlsx or [])

    per_domain_args = {
        "Homophobie": args.xlsx_homophobie,
        "Obésité": args.xlsx_obesite,
        "Racisme": args.xlsx_racisme,
        "Religion": args.xlsx_religion,
    }
    overrides.update({domain: path for domain, path in per_domain_args.items() if path})

    return config.resolve_xlsx_paths(xlsx_dir=args.xlsx_dir, overrides=overrides)


def _json_ready(obj):
    """Convert nested objects to JSON-serializable Python values."""
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(data), f, ensure_ascii=False, indent=2)


def _write_df(path, df):
    if df is None or not hasattr(df, "empty") or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keep_index = not isinstance(df.index, pd.RangeIndex)
    df.to_csv(path, index=keep_index, encoding="utf-8-sig")


def _export_structured_outputs(
    out_dir,
    *,
    df,
    eval_labels,
    label_errors_df,
    violations_df,
    brier_df,
    cond_results,
    interaction_results,
    profile_stats,
    logit_df,
    sweep_results,
    calibration_data,
    density_results,
    length_results,
    cross_results,
    domain_density_results,
    univar_results,
    bivar_results,
    rf_model,
    shap_values,
    feature_names,
):
    """Persist machine-readable outputs for downstream analysis."""
    structured_dir = Path(out_dir) / "structured"
    structured_dir.mkdir(parents=True, exist_ok=True)

    _write_df(structured_dir / "label_errors.csv", label_errors_df)
    _write_df(structured_dir / "annotation_violations_per_sample.csv", violations_df)
    _write_df(structured_dir / "brier_scores.csv", brier_df)

    summary = {
        "n_rows": len(df),
        "eval_labels": list(eval_labels),
        "domains": df["domain"].value_counts().sort_index().to_dict() if "domain" in df.columns else {},
    }
    _write_json(structured_dir / "run_summary.json", summary)

    if cond_results:
        _write_df(structured_dir / "conditional" / "delta_f1_emotion_given_mode.csv",
                  cond_results.get("delta_f1_emotion_given_mode"))
        _write_df(structured_dir / "conditional" / "delta_f1_mode_given_emotion.csv",
                  cond_results.get("delta_f1_mode_given_emotion"))
        _write_json(structured_dir / "conditional" / "stratified_metrics_emotion_given_mode.json",
                    cond_results.get("stratified_metrics_emotion_given_mode", {}))
        _write_json(structured_dir / "conditional" / "stratified_metrics_mode_given_emotion.json",
                    cond_results.get("stratified_metrics_mode_given_emotion", {}))
        _write_json(structured_dir / "conditional" / "global_emotion_f1.json",
                    cond_results.get("global_emotion_f1", {}))
        _write_json(structured_dir / "conditional" / "global_mode_f1.json",
                    cond_results.get("global_mode_f1", {}))

    if interaction_results:
        _write_df(structured_dir / "interaction" / "interaction_effects.csv",
                  interaction_results.get("interaction_effects"))
        details = interaction_results.get("interaction_details")
        if details:
            _write_df(structured_dir / "interaction" / "interaction_details.csv", pd.DataFrame(details))
        _write_df(structured_dir / "interaction" / "error_cooccurrence.csv",
                  interaction_results.get("error_cooccurrence"))

    _write_df(structured_dir / "interaction" / "combination_profiles.csv", profile_stats)
    _write_df(structured_dir / "logits" / "logit_distribution.csv", logit_df)

    if sweep_results:
        _write_df(structured_dir / "logits" / "threshold_sweep.csv",
                  sweep_results.get("sweep_results"))
        _write_json(structured_dir / "logits" / "optimal_thresholds.json",
                    sweep_results.get("optimal_thresholds", {}))
        _write_json(structured_dir / "logits" / "pareto_fronts.json",
                    sweep_results.get("pareto_fronts", {}))

    if calibration_data:
        _write_json(structured_dir / "logits" / "calibration_data.json", calibration_data)

    if density_results:
        _write_json(structured_dir / "stratification" / "density_results.json", density_results)
    if length_results:
        _write_json(structured_dir / "stratification" / "length_results.json", length_results)
    if cross_results:
        _write_df(structured_dir / "stratification" / "cross_grid_mean_error.csv",
                  cross_results.get("grid"))
        _write_df(structured_dir / "stratification" / "cross_grid_counts.csv",
                  cross_results.get("grid_n"))
        danger = cross_results.get("danger_zones")
        if danger:
            _write_df(structured_dir / "stratification" / "danger_zones.csv", pd.DataFrame(danger))
    if domain_density_results:
        _write_json(structured_dir / "stratification" / "domain_density_results.json",
                    domain_density_results)

    if univar_results:
        _write_json(structured_dir / "explainability" / "univariate_results.json", univar_results)
    if bivar_results:
        _write_df(structured_dir / "explainability" / "bivariate_results.csv", pd.DataFrame(bivar_results))

    if rf_model is not None and feature_names is not None:
        importances = pd.DataFrame({
            "feature": feature_names,
            "importance": rf_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        _write_df(structured_dir / "explainability" / "rf_feature_importance.csv", importances)

    if shap_values is not None and feature_names is not None:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)
        _write_df(structured_dir / "explainability" / "shap_mean_abs.csv", shap_df)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="EMOTYC Error Analysis Pipeline (modular)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", type=str,
                    default=str(config.DEFAULT_OUTPUT_DIR),
                    help="Output directory")
    p.add_argument("--use-context", action="store_true",
                    help="Use neighboring sentences as context")
    p.add_argument("--no-optimized-thresholds", action="store_true",
                    help="Fixed 0.5 threshold instead of optimized")
    p.add_argument("--skip-inference", action="store_true",
                    help="Skip inference, load from pre-computed JSONL")
    p.add_argument("--predictions-dir", type=str,
                    default=str(config.PROJECT_ROOT / "outputs"),
                    help="Directory with pre-computed JSONL predictions")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--min-support", type=float, default=0.08,
                    help="Minimum support for FP-Growth")
    p.add_argument("--min-confidence", type=float, default=0.5,
                    help="Minimum confidence for association rules")
    p.add_argument("--from-csv", type=str, default=None,
                    help="Load a pre-computed analysis_data.csv "
                         "(skips inference + metrics)")
    p.add_argument("--xlsx-dir", type=str, default=None,
                    help="Directory containing the 4 input XLSX files. Supports both golds/ and new_golds/ layouts.")
    p.add_argument("--xlsx", action="append", type=_parse_domain_path,
                    help="Override one XLSX path with DOMAIN=PATH. Repeatable.")
    p.add_argument("--xlsx-homophobie", type=str, default=None,
                    help="Explicit XLSX path for Homophobie")
    p.add_argument("--xlsx-obesite", type=str, default=None,
                    help="Explicit XLSX path for Obésité")
    p.add_argument("--xlsx-racisme", type=str, default=None,
                    help="Explicit XLSX path for Racisme")
    p.add_argument("--xlsx-religion", type=str, default=None,
                    help="Explicit XLSX path for Religion")
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_paths = None

    if not args.from_csv:
        xlsx_paths = _build_xlsx_paths_from_args(args)

    config_str = (
        f"context={'yes' if args.use_context else 'no'}, "
        f"thresholds={'0.5' if args.no_optimized_thresholds else 'optimized'}"
    )
    print(f"\n{'═' * 70}")
    print(f"  EMOTYC Error Analysis Pipeline (modular)")
    print(f"  Config: {config_str}")
    print(f"{'═' * 70}")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1: DATA LOADING & INFERENCE
    # ══════════════════════════════════════════════════════════════════

    if args.from_csv:
        # ── Load from pre-computed CSV ────────────────────────────────
        print(f"\n▸ Chargement depuis CSV : {args.from_csv}")
        df = pd.read_csv(args.from_csv, encoding="utf-8-sig")
        eval_labels = [
            e for e in config.EMOTION_12
            if e in df.columns and f"pred_{e}" in df.columns
        ]
        print(f"  ✓ {len(df)} lignes, {len(eval_labels)} émotions évaluées")

        # Rebuild text features if missing
        if "text_length" not in df.columns or "word_count" not in df.columns:
            df = data_loader.add_text_features(df)
    else:
        # ── Fresh load ────────────────────────────────────────────────
        print("\n▸ 1. Chargement et nettoyage des données…")
        print("  XLSX utilisés :")
        for domain, path in xlsx_paths.items():
            print(f"    - {domain}: {path}")
        df = data_loader.load_and_clean_data(xlsx_paths=xlsx_paths)
        df = data_loader.add_text_features(df)

        # ── Inference ─────────────────────────────────────────────────
        print("\n▸ 2. Prédictions EMOTYC…")
        df = inference.run_or_load(df, args)

        # ── Error metrics ─────────────────────────────────────────────
        print("\n▸ 3. Calcul des métriques d'erreur…")
        df, eval_labels = metrics.compute_error_metrics(df)

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 2: FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════

    print("\n▸ 4. Feature engineering…")
    df = data_loader.add_density_features(df)
    X_df, feature_names = data_loader.build_analysis_features(df)

    # ── Save augmented DataFrame ──────────────────────────────────────
    csv_path = out_dir / "analysis_data.csv"
    export_cols = (
        ["domain", "_original_idx", "TEXT", "text_length", "word_count"]
        + config.BINARY_FEATURES
        + [c for c in config.QUALITATIVE_FEATURES if c in df.columns]
        + [l for l in eval_labels if l in df.columns]
        + [f"pred_{l}" for l in eval_labels if f"pred_{l}" in df.columns]
        + [f"proba_{l}" for l in eval_labels if f"proba_{l}" in df.columns]
        + [f"err_{l}" for l in eval_labels if f"err_{l}" in df.columns]
        + [c for c in [
            "n_errors_12", "hamming_12", "jaccard_error_12",
            "weighted_hamming_12", "error_category",
            "emotion_density_12", "mode_density_4", "label_density_19",
        ] if c in df.columns]
    )
    export_cols = [c for c in export_cols if c in df.columns]
    df[export_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ DataFrame exporté : {csv_path}")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 3: CORE ANALYSES
    # ══════════════════════════════════════════════════════════════════

    # ── Per-label errors ──────────────────────────────────────────────
    print("\n▸ 5. Analyse par label…")
    label_errors_df = metrics.compute_per_label_errors(df, eval_labels)

    # ── Annotation scheme violations ──────────────────────────────────
    print("\n▸ 6. Violations du schéma d'annotation…")
    violations_df = metrics.compute_annotation_violations(df)

    # ── Brier score decomposition ─────────────────────────────────────
    print("\n▸ 7. Brier score decomposition…")
    all_labels_for_brier = [
        l for l in eval_labels + config.MODES_4
        if l in df.columns and f"proba_{l}" in df.columns
    ]
    brier_df = metrics.compute_brier_scores(df, all_labels_for_brier)

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 4: CONDITIONAL ANALYSIS (Objectives 1 & 2)
    # ══════════════════════════════════════════════════════════════════

    print("\n▸ 8. Analyse conditionnelle (modes ↔ émotions)…")
    cond_results = conditional.conditional_mode_emotion_analysis(df)

    print("\n▸ 9. Analyse des interactions…")
    interaction_results = conditional.interaction_analysis(df)

    print("\n▸ 10. Profils de combinaisons…")
    profile_stats = conditional.combination_profile_analysis(df)

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 5: LOGIT & THRESHOLD ANALYSIS (Objective 3)
    # ══════════════════════════════════════════════════════════════════

    print("\n▸ 11. Analyse des logits…")
    logit_df = logit_analysis.logit_distribution_analysis(df)

    print("\n▸ 12. Threshold sweep (modes)…")
    sweep_results = logit_analysis.threshold_sweep_modes(df)

    print("\n▸ 13. Analyse de calibration…")
    calibration_data = logit_analysis.calibration_analysis(df)

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 6: STRATIFICATION (Objective 4)
    # ══════════════════════════════════════════════════════════════════

    print("\n▸ 14. Stratification par densité…")
    density_results = stratification.density_stratified_analysis(df)

    print("\n▸ 15. Stratification par longueur…")
    length_results = stratification.length_stratified_analysis(df)

    print("\n▸ 16. Cross-stratification (densité × longueur)…")
    cross_results = stratification.cross_stratification(df)

    print("\n▸ 17. Analyse densité contrôlée par domaine…")
    domain_density_results = stratification.domain_controlled_density_analysis(df)

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 7: EXPLAINABILITY
    # ══════════════════════════════════════════════════════════════════

    print("\n▸ 18. Analyse univariée…")
    univar_results = explainability.univariate_analysis(df)

    print("\n▸ 19. Analyse bivariée…")
    bivar_results = explainability.bivariate_analysis(df)

    print("\n▸ 20. Random Forest + SHAP…")
    rf_model, shap_values, feat_names, dt_model = explainability.rf_shap_analysis(
        df, X_df, feature_names
    )

    print("\n▸ 21. Règles d'association (FP-Growth)…")
    rules = explainability.association_rule_analysis(
        df, min_support=args.min_support, min_confidence=args.min_confidence,
    )

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 8: VISUALIZATION
    # ══════════════════════════════════════════════════════════════════

    print("\n▸ 22. Visualisations…")
    visualization.plot_all(
        df, eval_labels, out_dir,
        label_errors_df=label_errors_df,
        univar_results=univar_results,
        bivar_results=bivar_results,
        rf_model=rf_model,
        shap_values=shap_values,
        X_df=X_df,
        feature_names=feat_names,
        cond_results=cond_results,
        interaction_results=interaction_results,
        logit_df=logit_df,
        sweep_results=sweep_results,
        calibration_data=calibration_data,
        density_results=density_results,
        length_results=length_results,
        cross_results=cross_results,
    )

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 9: REPORT
    # ══════════════════════════════════════════════════════════════════

    print("\n▸ 23. Génération du rapport…")
    report.generate_report(
        out_dir, config_str,
        df=df,
        eval_labels=eval_labels,
        label_errors_df=label_errors_df,
        violations_df=violations_df,
        brier_df=brier_df,
        cond_results=cond_results,
        interaction_results=interaction_results,
        profile_stats=profile_stats,
        logit_df=logit_df,
        sweep_results=sweep_results,
        calibration_data=calibration_data,
        density_results=density_results,
        length_results=length_results,
        cross_results=cross_results,
        domain_density_results=domain_density_results,
        univar_results=univar_results,
        bivar_results=bivar_results,
        rf_model=rf_model,
        shap_values=shap_values,
        feature_names=feat_names,
        dt_model=dt_model,
        rules=rules,
    )

    print("\n▸ 24. Export structuré…")
    _export_structured_outputs(
        out_dir,
        df=df,
        eval_labels=eval_labels,
        label_errors_df=label_errors_df,
        violations_df=violations_df,
        brier_df=brier_df,
        cond_results=cond_results,
        interaction_results=interaction_results,
        profile_stats=profile_stats,
        logit_df=logit_df,
        sweep_results=sweep_results,
        calibration_data=calibration_data,
        density_results=density_results,
        length_results=length_results,
        cross_results=cross_results,
        domain_density_results=domain_density_results,
        univar_results=univar_results,
        bivar_results=bivar_results,
        rf_model=rf_model,
        shap_values=shap_values,
        feature_names=feat_names,
    )

    # ── Export association rules if available ──────────────────────────
    if rules is not None and hasattr(rules, "to_csv"):
        rules.to_csv(out_dir / "association_rules_high_error.csv", index=False)

    # ── Export decision tree ──────────────────────────────────────────
    if dt_model is not None and hasattr(dt_model, "tree_text"):
        with open(out_dir / "decision_tree_rules.txt", "w",
                  encoding="utf-8") as f:
            f.write("Decision Tree (max_depth=4) — "
                    "prédiction du Hamming Error\n")
            f.write("=" * 70 + "\n")
            f.write(dt_model.tree_text)

    print(f"\n{'═' * 70}")
    print(f"  ✓ Analyse terminée. Résultats dans : {out_dir}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
