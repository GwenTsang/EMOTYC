#!/usr/bin/env python3
"""
EMOTYC Sanity Check — Script 2/3
═════════════════════════════════

Vérifie la cohérence logique interne des prédictions EMOTYC (ou des
gold labels). Toutes les fonctions de vérification sont réutilisables
et acceptent un préfixe (`"pred_"` pour les prédictions, `""` pour les
gold labels).

Contraintes métier vérifiées :
  1. Emo ↔ Emotions  (implication bidirectionnelle)
  2. Base ou Complexe ↔ il existe une émotion spécifique (implication bidirectionnelle)
  3. Modes ↔ nombre d'émotions  (M ≤ E, E=0⇒M=0, E>0⇒M≥1)

Usage :
    # Vérifier des prédictions
    python scripts/emotyc_sanity_check.py \\
        --input results/predictions/predictions_ctx0_thr1_tpl1.csv \\
        --out-dir results/sanity \\
        --prefix pred_

    # Vérifier des gold labels
    python scripts/emotyc_sanity_check.py \\
        --input golds/homophobie/homophobie_annotations_gold_flat.xlsx \\
        --out-dir results/sanity_gold \\
        --prefix ""
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
#  LABEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

EMOTIONS_BASE = ["Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse"]
EMOTIONS_COMPLEXE = ["Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie"]
EMOTIONS_ALL = EMOTIONS_BASE + EMOTIONS_COMPLEXE + ["Autre"]
MODES = ["Comportementale", "Désignée", "Montrée", "Suggérée"]


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _safe_get(row, col, default=None):
    """Récupère une valeur dans une row, retourne default si absent ou NaN."""
    if col not in row.index:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    return val


def _safe_text(val, max_len=100):
    """Tronque un texte pour l'affichage."""
    if val is None:
        return ""
    s = str(val)
    return s[:max_len] + "…" if len(s) > max_len else s


def _to_python(x):
    """Convertit numpy scalar → python natif pour la sérialisation JSON."""
    if hasattr(x, "item"):
        return x.item()
    return x


def _col(prefix, name):
    """Construit le nom de colonne : prefix + name."""
    return f"{prefix}{name}"


def _has_col(df, prefix, name):
    """Vérifie si une colonne préfixée existe dans le DataFrame."""
    return _col(prefix, name) in df.columns


def _get_val(row, prefix, name, default=None):
    """Récupère la valeur d'une colonne préfixée."""
    return _safe_get(row, _col(prefix, name), default)


# ═══════════════════════════════════════════════════════════════════════════
#  CHECK 1 : Emo ↔ Emotions
# ═══════════════════════════════════════════════════════════════════════════

def check_emo_vs_emotions(df, prefix="pred_"):
    """
    Vérifie la cohérence bidirectionnelle Emo ↔ Emotions.

    Règles :
      - Emo=0 ⇒ toutes les émotions = 0
      - toutes les émotions = 0 ⇒ Emo = 0
      (autrement dit : Emo ↔ OR(emotions))

    Args:
        df: DataFrame avec les colonnes {prefix}Emo, {prefix}{emotion}
        prefix: "pred_" pour les prédictions, "" pour les gold labels

    Returns:
        dict avec :
          - "emotion_sans_emo": cas où une émotion est active mais Emo=0
          - "emo_sans_emotion": cas où Emo=1 mais aucune émotion active
          - "n_violations": total
          - "n_checked": nombre de lignes vérifiées
          - "skipped": True si les colonnes nécessaires sont absentes
    """
    result = {
        "emotion_sans_emo": [],
        "emo_sans_emotion": [],
        "n_violations": 0,
        "n_checked": 0,
        "skipped": False,
        "skip_reason": None,
    }

    # Vérifier que la colonne Emo existe
    if not _has_col(df, prefix, "Emo"):
        result["skipped"] = True
        result["skip_reason"] = f"Colonne '{_col(prefix, 'Emo')}' absente"
        return result

    # Déterminer les colonnes d'émotions disponibles
    available_emotions = [e for e in EMOTIONS_ALL if _has_col(df, prefix, e)]
    if not available_emotions:
        result["skipped"] = True
        result["skip_reason"] = "Aucune colonne d'émotion trouvée"
        return result

    result["n_checked"] = len(df)

    for idx, row in df.iterrows():
        text = _safe_text(row.get("TEXT"))
        emo_val = _get_val(row, prefix, "Emo", default=0)

        # Émotions actives
        detected = [e for e in available_emotions if _get_val(row, prefix, e, 0) == 1]
        any_emotion = len(detected) > 0

        # Règle 1 : émotion détectée mais Emo=0
        if any_emotion and emo_val == 0:
            result["emotion_sans_emo"].append({
                "idx": _to_python(idx),
                "text": text,
                "emotions_detectees": detected,
                "domain": row.get("domain", ""),
            })

        # Règle 2 : Emo=1 mais aucune émotion
        if emo_val == 1 and not any_emotion:
            proba_emo = _get_val(row, "proba_", "Emo") if prefix == "pred_" else None
            result["emo_sans_emotion"].append({
                "idx": _to_python(idx),
                "text": text,
                "proba_Emo": _to_python(proba_emo) if proba_emo is not None else None,
                "domain": row.get("domain", ""),
            })

    result["n_violations"] = len(result["emotion_sans_emo"]) + len(result["emo_sans_emotion"])
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  CHECK 2 : Base/Complexe ↔ émotions spécifiques
# ═══════════════════════════════════════════════════════════════════════════

def check_base_complex_vs_emotions(df, prefix="pred_"):
    """
    Vérifie la cohérence bidirectionnelle Base/Complexe ↔ émotions.

    Règles :
      - Base=1 ⇒ au moins une émotion de base active
      - ∃ émotion de base active ⇒ Base=1
      - Complexe=1 ⇒ au moins une émotion complexe active
      - ∃ émotion complexe active ⇒ Complexe=1

    Returns:
        dict avec les catégories de violations
    """
    result = {
        "base_sans_emotion_base": [],
        "emotion_base_sans_base": [],
        "complexe_sans_emotion_complexe": [],
        "emotion_complexe_sans_complexe": [],
        "n_violations": 0,
        "n_checked": 0,
        "skipped": False,
        "skip_reason": None,
    }

    has_base = _has_col(df, prefix, "Base")
    has_complexe = _has_col(df, prefix, "Complexe")

    if not has_base and not has_complexe:
        result["skipped"] = True
        result["skip_reason"] = f"Colonnes '{_col(prefix, 'Base')}' et '{_col(prefix, 'Complexe')}' absentes"
        return result

    available_base = [e for e in EMOTIONS_BASE if _has_col(df, prefix, e)]
    available_complexe = [e for e in EMOTIONS_COMPLEXE if _has_col(df, prefix, e)]

    result["n_checked"] = len(df)

    for idx, row in df.iterrows():
        text = _safe_text(row.get("TEXT"))
        domain = row.get("domain", "")

        # ── Base ──────────────────────────────────────────────────────
        if has_base:
            pred_base = _get_val(row, prefix, "Base", 0)
            base_detected = [e for e in available_base if _get_val(row, prefix, e, 0) == 1]

            # Base=1 mais aucune émotion de base
            if pred_base == 1 and not base_detected:
                result["base_sans_emotion_base"].append({
                    "idx": _to_python(idx),
                    "text": text,
                    "domain": domain,
                    "proba_Base": _to_python(
                        _get_val(row, "proba_", "Base")
                    ) if prefix == "pred_" else None,
                })

            # Émotion de base détectée mais Base=0
            if base_detected and pred_base == 0:
                result["emotion_base_sans_base"].append({
                    "idx": _to_python(idx),
                    "text": text,
                    "domain": domain,
                    "emotions_base_detectees": base_detected,
                })

        # ── Complexe ──────────────────────────────────────────────────
        if has_complexe:
            pred_complexe = _get_val(row, prefix, "Complexe", 0)
            complexe_detected = [e for e in available_complexe if _get_val(row, prefix, e, 0) == 1]

            # Complexe=1 mais aucune émotion complexe
            if pred_complexe == 1 and not complexe_detected:
                result["complexe_sans_emotion_complexe"].append({
                    "idx": _to_python(idx),
                    "text": text,
                    "domain": domain,
                    "proba_Complexe": _to_python(
                        _get_val(row, "proba_", "Complexe")
                    ) if prefix == "pred_" else None,
                })

            # Émotion complexe détectée mais Complexe=0
            if complexe_detected and pred_complexe == 0:
                result["emotion_complexe_sans_complexe"].append({
                    "idx": _to_python(idx),
                    "text": text,
                    "domain": domain,
                    "emotions_complexes_detectees": complexe_detected,
                })

    result["n_violations"] = (
        len(result["base_sans_emotion_base"])
        + len(result["emotion_base_sans_base"])
        + len(result["complexe_sans_emotion_complexe"])
        + len(result["emotion_complexe_sans_complexe"])
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  CHECK 3 : Modes ↔ Nombre d'émotions
# ═══════════════════════════════════════════════════════════════════════════

def check_modes_vs_emotions(df, prefix="pred_"):
    """
    Vérifie la cohérence Modes ↔ Émotions.

    Règles :
      - E=0 ⇒ M=0  (pas d'émotion → pas de mode)
      - E>0 ⇒ M≥1  (émotion présente → au moins un mode)
      - M ≤ E       (pas plus de modes que d'émotions)

    où E = nombre d'émotions actives, M = nombre de modes actifs.

    Returns:
        dict avec les catégories de violations
    """
    result = {
        "mode_sans_emotion": [],     # M>0, E=0
        "emotion_sans_mode": [],     # E>0, M=0
        "trop_de_modes": [],         # M > E
        "n_violations": 0,
        "n_checked": 0,
        "skipped": False,
        "skip_reason": None,
    }

    # Vérifier les colonnes de modes disponibles
    available_modes = [m for m in MODES if _has_col(df, prefix, m)]
    if not available_modes:
        result["skipped"] = True
        result["skip_reason"] = "Aucune colonne de mode trouvée"
        return result

    available_emotions = [e for e in EMOTIONS_ALL if _has_col(df, prefix, e)]
    if not available_emotions:
        result["skipped"] = True
        result["skip_reason"] = "Aucune colonne d'émotion trouvée"
        return result

    result["n_checked"] = len(df)

    # Vectorisé pour la performance
    emotion_cols = [_col(prefix, e) for e in available_emotions]
    mode_cols = [_col(prefix, m) for m in available_modes]

    E = df[emotion_cols].fillna(0).astype(int).sum(axis=1)
    M = df[mode_cols].fillna(0).astype(int).sum(axis=1)

    # Masques d'anomalies
    mask_mode_sans_emo = (E == 0) & (M > 0)
    mask_emo_sans_mode = (E > 0) & (M == 0)
    mask_trop_de_modes = M > E

    # Collecter les détails pour chaque type
    for idx in df.index[mask_mode_sans_emo]:
        row = df.loc[idx]
        active_modes = [m for m in available_modes if _get_val(row, prefix, m, 0) == 1]
        result["mode_sans_emotion"].append({
            "idx": _to_python(idx),
            "text": _safe_text(row.get("TEXT")),
            "domain": row.get("domain", ""),
            "modes_actifs": active_modes,
            "E": 0,
            "M": int(M[idx]),
        })

    for idx in df.index[mask_emo_sans_mode]:
        row = df.loc[idx]
        active_emotions = [e for e in available_emotions if _get_val(row, prefix, e, 0) == 1]
        result["emotion_sans_mode"].append({
            "idx": _to_python(idx),
            "text": _safe_text(row.get("TEXT")),
            "domain": row.get("domain", ""),
            "emotions_actives": active_emotions,
            "E": int(E[idx]),
            "M": 0,
        })

    for idx in df.index[mask_trop_de_modes]:
        row = df.loc[idx]
        active_emotions = [e for e in available_emotions if _get_val(row, prefix, e, 0) == 1]
        active_modes = [m for m in available_modes if _get_val(row, prefix, m, 0) == 1]
        result["trop_de_modes"].append({
            "idx": _to_python(idx),
            "text": _safe_text(row.get("TEXT")),
            "domain": row.get("domain", ""),
            "emotions_actives": active_emotions,
            "modes_actifs": active_modes,
            "E": int(E[idx]),
            "M": int(M[idx]),
        })

    result["n_violations"] = (
        len(result["mode_sans_emotion"])
        + len(result["emotion_sans_mode"])
        + len(result["trop_de_modes"])
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════

def run_all_checks(df, prefix="pred_"):
    """
    Exécute les 3 vérifications de cohérence et agrège les résultats.

    Args:
        df: DataFrame contenant les colonnes à vérifier
        prefix: "pred_" pour les prédictions, "" pour les gold labels

    Returns:
        dict structuré avec tous les résultats
    """
    checks = {
        "emo_vs_emotions": check_emo_vs_emotions(df, prefix),
        "base_complex": check_base_complex_vs_emotions(df, prefix),
        "modes_vs_emotions": check_modes_vs_emotions(df, prefix),
    }

    # Compter le total des violations (hors checks skippés)
    total_violations = sum(
        c["n_violations"] for c in checks.values() if not c["skipped"]
    )

    # Identifier les indices de lignes avec au moins une violation
    violation_indices = set()
    for check_name, check_result in checks.items():
        if check_result["skipped"]:
            continue
        for sub_key, sub_list in check_result.items():
            if isinstance(sub_list, list) and sub_list and isinstance(sub_list[0], dict):
                for item in sub_list:
                    if "idx" in item:
                        violation_indices.add(item["idx"])

    n_samples = len(df)
    n_clean = n_samples - len(violation_indices)

    summary = {
        "n_samples": n_samples,
        "total_violations": total_violations,
        "n_rows_with_violations": len(violation_indices),
        "n_clean_rows": n_clean,
        "pct_clean": round(100 * n_clean / n_samples, 2) if n_samples > 0 else 0,
        "checks_skipped": [k for k, v in checks.items() if v["skipped"]],
    }

    return {"checks": checks, "summary": summary}


# ═══════════════════════════════════════════════════════════════════════════
#  DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_sanity_report(results, title="", max_examples=3):
    """Affiche un tableau synthétique des résultats de sanity check."""
    checks = results["checks"]
    summary = results["summary"]
    n = summary["n_samples"]

    print(f"\n{'═' * 70}")
    print(f"  SANITY CHECK — {title}" if title else "  SANITY CHECK")
    print(f"{'═' * 70}")

    # Détail par sous-vérification
    SUB_CHECK_LABELS = {
        "emo_vs_emotions": {
            "emotion_sans_emo":  "Émotion détectée mais Emo=0",
            "emo_sans_emotion":  "Emo=1 mais aucune émotion",
        },
        "base_complex": {
            "base_sans_emotion_base":         "Base=1 sans émotion de base",
            "emotion_base_sans_base":         "Émotion de base mais Base=0",
            "complexe_sans_emotion_complexe": "Complexe=1 sans émotion complexe",
            "emotion_complexe_sans_complexe": "Émotion complexe mais Complexe=0",
        },
        "modes_vs_emotions": {
            "mode_sans_emotion":  "Mode sans émotion (M>0, E=0)",
            "emotion_sans_mode":  "Émotion sans mode (E>0, M=0)",
            "trop_de_modes":      "Plus de modes que d'émotions (M>E)",
        },
    }

    print(f"  {'Vérification':<45s} {'Violations':>10s} {'%':>8s}")
    print(f"  {'─' * 65}")

    for check_name, sub_labels in SUB_CHECK_LABELS.items():
        check = checks[check_name]
        if check["skipped"]:
            print(f"  ⊘ {check_name:<43s} {'SKIPPED':>10s}   ({check['skip_reason']})")
            continue

        for sub_key, label in sub_labels.items():
            violations = check.get(sub_key, [])
            count = len(violations)
            pct = f"{100 * count / n:.2f}%" if n > 0 else "N/A"
            marker = "⚠" if count > 0 else "✓"
            print(f"  {marker} {label:<43s} {count:>10d} {pct:>8s}")

            # Exemples
            if count > 0 and max_examples > 0:
                for ex in violations[:max_examples]:
                    print(f"      → idx={ex['idx']}  {ex.get('text', '')[:70]}")

    print(f"  {'─' * 65}")
    print(f"  TOTAL violations                                {summary['total_violations']:>10d}")
    print(f"  Lignes 100% cohérentes                          "
          f"{summary['n_clean_rows']:>10d}  ({summary['pct_clean']:.1f}%)")

    if summary["checks_skipped"]:
        print(f"  Checks ignorés : {', '.join(summary['checks_skipped'])}")

    print(f"{'═' * 70}")


# ═══════════════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_sanity_report(results, out_path, condition_tag=None):
    """Exporte le rapport de sanity check en JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export = {
        "condition": condition_tag,
        "summary": results["summary"],
        "checks": {},
    }

    for check_name, check_data in results["checks"].items():
        export["checks"][check_name] = {
            "n_violations": check_data["n_violations"],
            "n_checked": check_data["n_checked"],
            "skipped": check_data["skipped"],
            "skip_reason": check_data.get("skip_reason"),
        }

        # Sous-vérifications avec comptages
        for key, val in check_data.items():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                export["checks"][check_name][key] = {
                    "n": len(val),
                    "pct": round(100 * len(val) / check_data["n_checked"], 2) if check_data["n_checked"] > 0 else 0,
                    "examples": val[:10],  # Limiter pour la taille du JSON
                }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Rapport sanity check exporté : {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING (supporte CSV et XLSX)
# ═══════════════════════════════════════════════════════════════════════════

def load_input(input_path: str) -> pd.DataFrame:
    """Charge un fichier CSV ou XLSX."""
    p = Path(input_path)
    if not p.exists():
        print(f"✗ Fichier introuvable : {p}")
        sys.exit(1)

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p, encoding="utf-8-sig")
    elif p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        print(f"✗ Format non supporté : {p.suffix} (attendu: .csv ou .xlsx)")
        sys.exit(1)

    print(f"✓ Fichier chargé : {p.name} ({len(df)} lignes, {len(df.columns)} colonnes)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="EMOTYC Sanity Check — Vérification de cohérence logique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True,
                    help="Fichier CSV (prédictions) ou XLSX (gold labels) à vérifier")
    p.add_argument("--out-dir", required=True,
                    help="Dossier de sortie pour les rapports JSON")
    p.add_argument("--prefix", default="pred_",
                    help='Préfixe des colonnes à vérifier : "pred_" pour les prédictions, '
                         '"" pour les gold labels (défaut: pred_)')
    p.add_argument("--max-examples", type=int, default=3,
                    help="Nombre max d'exemples à afficher par type de violation (défaut: 3)")
    return p.parse_args()


def main():
    args = parse_args()

    prefix = args.prefix
    target = "prédictions" if prefix == "pred_" else "gold labels"

    print(f"\n{'═' * 70}")
    print(f"  EMOTYC Sanity Check — Vérification des {target}")
    print(f"{'═' * 70}")

    # ── 1. Chargement ─────────────────────────────────────────────────
    df = load_input(args.input)

    # ── 2. Sanity checks ──────────────────────────────────────────────
    results = run_all_checks(df, prefix=prefix)

    # ── 3. Affichage ──────────────────────────────────────────────────
    title = Path(args.input).name
    print_sanity_report(results, title=title, max_examples=args.max_examples)

    # ── 4. Export ─────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    stem = Path(args.input).stem
    out_path = out_dir / f"sanity_{stem}.json"
    export_sanity_report(results, out_path)

    print(f"\n  ✓ Terminé.")


if __name__ == "__main__":
    main()
