#!/usr/bin/env python3
"""
EMOTYC Pipeline — Script 3/3 (Orchestrateur)
═════════════════════════════════════════════

Orchestre le pipeline complet :
  1. Découvre les fichiers XLSX dans un répertoire
  2. Charge le modèle EMOTYC une seule fois
  3. Itère sur toutes les combinaisons de conditions (2³ = 8)
  4. Pour chaque condition : inférence + sanity check
  5. Produit un rapport comparatif synthétique

Usage :
    # Pipeline complet (8 conditions)
    python scripts/emotyc_pipeline.py \\
        --input-dir golds/ \\
        --out-dir results/sanity_pipeline

    # Une seule condition spécifique
    python scripts/emotyc_pipeline.py \\
        --input-dir golds/ \\
        --out-dir results/sanity_pipeline \\
        --use-context --no-optimized-thresholds --no-template

    # Batch complet avec batch_size ajusté pour T4
    python scripts/emotyc_pipeline.py \\
        --input-dir golds/ \\
        --out-dir results/sanity_pipeline \\
        --batch-size 32
"""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Ajout du répertoire scripts/ au path ─────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from emotyc_predict import load_model
from emotyc_batch_predict import (
    discover_xlsx_files,
    load_all_xlsx,
    run_inference,
    save_predictions,
    build_condition_tag,
    build_condition_description,
)
from emotyc_sanity_check import (
    run_all_checks,
    print_sanity_report,
    export_sanity_report,
)

PROJECT_ROOT = SCRIPT_DIR.parent


# ═══════════════════════════════════════════════════════════════════════════
#  CONDITION GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_conditions():
    """
    Génère les 8 combinaisons de conditions.
    Retourne une liste de dicts avec les flags booléens.
    """
    conditions = []
    for use_context, opt_thresholds, use_template in itertools.product([False, True], repeat=3):
        cond = {
            "use_context": use_context,
            "optimized_thresholds": opt_thresholds,
            "use_template": use_template,
        }
        cond["tag"] = build_condition_tag(use_context, opt_thresholds, use_template)
        cond["desc"] = build_condition_description(use_context, opt_thresholds, use_template)
        conditions.append(cond)
    return conditions


# ═══════════════════════════════════════════════════════════════════════════
#  COMPARATIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def build_comparative_summary(all_results):
    """
    Construit un tableau comparatif des sanity checks à travers toutes
    les conditions.

    Args:
        all_results: liste de dicts {condition, tag, desc, sanity_results}

    Returns:
        DataFrame comparatif, dict structuré
    """
    rows = []
    for entry in all_results:
        cond = entry["condition"]
        sanity = entry["sanity_results"]
        checks = sanity["checks"]
        summary = sanity["summary"]

        row = {
            "condition": cond["tag"],
            "description": cond["desc"],
            "n_samples": summary["n_samples"],
        }

        # Extraire les comptages par check
        # Check 1: Emo ↔ Emotions
        c1 = checks["emo_vs_emotions"]
        if not c1["skipped"]:
            row["emo_sans_emotion"] = len(c1["emo_sans_emotion"])
            row["emotion_sans_emo"] = len(c1["emotion_sans_emo"])
        else:
            row["emo_sans_emotion"] = "—"
            row["emotion_sans_emo"] = "—"

        # Check 2: Base/Complexe
        c2 = checks["base_complex"]
        if not c2["skipped"]:
            row["base_sans_emo_base"] = len(c2["base_sans_emotion_base"])
            row["emo_base_sans_base"] = len(c2["emotion_base_sans_base"])
            row["cpx_sans_emo_cpx"] = len(c2["complexe_sans_emotion_complexe"])
            row["emo_cpx_sans_cpx"] = len(c2["emotion_complexe_sans_complexe"])
        else:
            for k in ["base_sans_emo_base", "emo_base_sans_base",
                       "cpx_sans_emo_cpx", "emo_cpx_sans_cpx"]:
                row[k] = "—"

        # Check 3: Modes
        c3 = checks["modes_vs_emotions"]
        if not c3["skipped"]:
            row["mode_sans_emo"] = len(c3["mode_sans_emotion"])
            row["emo_sans_mode"] = len(c3["emotion_sans_mode"])
            row["trop_modes"] = len(c3["trop_de_modes"])
        else:
            for k in ["mode_sans_emo", "emo_sans_mode", "trop_modes"]:
                row[k] = "—"

        # Totaux
        row["total_violations"] = summary["total_violations"]
        row["pct_clean"] = summary["pct_clean"]

        rows.append(row)

    df_summary = pd.DataFrame(rows)
    return df_summary


def print_comparative_summary(df_summary, n_samples):
    """Affiche le tableau comparatif formaté."""
    print(f"\n{'═' * 100}")
    print(f"  COMPARAISON — SANITY CHECK ACROSS {len(df_summary)} CONDITIONS ({n_samples} samples)")
    print(f"{'═' * 100}")

    # En-tête
    print(f"  {'Condition':<25s} {'Desc':<18s} "
          f"{'Emo↔E':>6s} {'B/C':>6s} {'Modes':>6s} "
          f"{'Total':>7s} {'%Clean':>7s}")
    print(f"  {'─' * 90}")

    for _, row in df_summary.iterrows():
        # Agréger les sous-checks
        emo_total = _safe_add(row.get("emo_sans_emotion"), row.get("emotion_sans_emo"))
        bc_total = _safe_add(
            _safe_add(row.get("base_sans_emo_base"), row.get("emo_base_sans_base")),
            _safe_add(row.get("cpx_sans_emo_cpx"), row.get("emo_cpx_sans_cpx"))
        )
        mode_total = _safe_add(
            _safe_add(row.get("mode_sans_emo"), row.get("emo_sans_mode")),
            row.get("trop_modes")
        )

        # Marqueur pour la condition canonique
        marker = " ★" if row["condition"] == "ctx0_thr1_tpl1" else "  "

        print(f"{marker}{row['condition']:<25s} {row['description']:<18s} "
              f"{_fmt(emo_total):>6s} {_fmt(bc_total):>6s} {_fmt(mode_total):>6s} "
              f"{row['total_violations']:>7} {row['pct_clean']:>6.1f}%")

    print(f"  {'─' * 90}")

    # Meilleure condition
    best = df_summary.loc[df_summary["pct_clean"].idxmax()]
    print(f"  ★ Meilleure cohérence : {best['condition']} ({best['description']}) — "
          f"{best['pct_clean']:.1f}% clean")
    print(f"{'═' * 100}")


def _safe_add(a, b):
    """Addition qui gère les '—' (checks skippés)."""
    if a == "—" or b == "—":
        return "—"
    if a is None:
        a = 0
    if b is None:
        b = 0
    return a + b


def _fmt(val):
    """Formate une valeur pour affichage (gère '—')."""
    if val == "—":
        return "—"
    return str(val)


# ═══════════════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_comparative_summary(df_summary, all_results, out_dir):
    """Exporte le résumé comparatif en JSON et TXT."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "comparative_summary.csv"
    df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # JSON structuré
    json_path = out_dir / "comparative_summary.json"
    export = {
        "n_conditions": len(df_summary),
        "n_samples": int(df_summary.iloc[0]["n_samples"]) if len(df_summary) > 0 else 0,
        "conditions": [],
    }
    for entry in all_results:
        export["conditions"].append({
            "tag": entry["condition"]["tag"],
            "desc": entry["condition"]["desc"],
            "flags": {
                "use_context": entry["condition"]["use_context"],
                "optimized_thresholds": entry["condition"]["optimized_thresholds"],
                "use_template": entry["condition"]["use_template"],
            },
            "summary": entry["sanity_results"]["summary"],
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Résumé comparatif exporté :")
    print(f"    CSV  : {csv_path}")
    print(f"    JSON : {json_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="EMOTYC Pipeline — Orchestrateur (inférence + sanity check)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input-dir", required=True,
                    help="Répertoire contenant les fichiers XLSX gold")
    p.add_argument("--out-dir", required=True,
                    help="Dossier de sortie principal")
    p.add_argument("--batch-size", type=int, default=16,
                    help="Taille du batch pour l'inférence (défaut: 16)")
    p.add_argument("--device", default=None,
                    help="Device PyTorch (défaut: auto-détection cuda/cpu)")

    # Flags pour restreindre à une seule condition
    single = p.add_argument_group("Condition unique (optionnel)",
        "Si au moins un de ces flags est spécifié, seule cette condition sera "
        "exécutée au lieu des 8 combinaisons.")
    single.add_argument("--use-context", action="store_true", default=None,
                         help="Utiliser les phrases voisines comme contexte")
    single.add_argument("--no-optimized-thresholds", action="store_true", default=None,
                         help="Seuil fixe 0.5")
    single.add_argument("--no-template", action="store_true", default=None,
                         help="Phrase brute sans template bca")

    # Option pour ne lancer que le sanity check (skip inference)
    p.add_argument("--sanity-only", action="store_true",
                    help="Ne pas lancer l'inférence, lire les CSV de prédictions existants "
                         "dans {out-dir}/predictions/")

    return p.parse_args()


def _single_condition_requested(args):
    """Vérifie si l'utilisateur veut une seule condition spécifique."""
    # argparse met None par défaut pour les flags non spécifiés
    # et False quand le flag est absent dans la ligne de commande
    # On utilise use_context comme proxy : si la valeur a été
    # explicitement set (True/False) vs laissée à None
    return any([
        args.use_context,
        args.no_optimized_thresholds,
        args.no_template,
    ])


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    sanity_dir = out_dir / "sanity"

    t_start = time.time()

    print(f"\n{'═' * 80}")
    print(f"  EMOTYC Pipeline — Inférence + Sanity Check")
    print(f"{'═' * 80}")

    # ── 1. Découverte des fichiers ────────────────────────────────────
    print("\n▸ 1. Découverte des fichiers XLSX…")
    xlsx_dict = discover_xlsx_files(args.input_dir)

    # ── 2. Chargement des données ─────────────────────────────────────
    print("\n▸ 2. Chargement des données…")
    df_base = load_all_xlsx(xlsx_dict)
    n_samples = len(df_base)

    # ── 3. Déterminer les conditions ──────────────────────────────────
    if _single_condition_requested(args):
        conditions = [{
            "use_context": bool(args.use_context),
            "optimized_thresholds": not bool(args.no_optimized_thresholds),
            "use_template": not bool(args.no_template),
            "tag": build_condition_tag(
                bool(args.use_context),
                not bool(args.no_optimized_thresholds),
                not bool(args.no_template),
            ),
            "desc": build_condition_description(
                bool(args.use_context),
                not bool(args.no_optimized_thresholds),
                not bool(args.no_template),
            ),
        }]
        print(f"\n▸ 3. Condition unique : {conditions[0]['tag']} ({conditions[0]['desc']})")
    else:
        conditions = generate_all_conditions()
        print(f"\n▸ 3. {len(conditions)} conditions à évaluer :")
        for c in conditions:
            marker = "★" if c["tag"] == "ctx0_thr1_tpl1" else " "
            print(f"  {marker} {c['tag']} ({c['desc']})")

    # ── 4. Chargement du modèle (une seule fois) ─────────────────────
    if not args.sanity_only:
        print("\n▸ 4. Chargement du modèle EMOTYC…")
        import torch
        device = torch.device(args.device) if args.device else None
        tokenizer, model, device = load_model(device)
    else:
        print("\n▸ 4. Mode --sanity-only : pas d'inférence")
        tokenizer = model = device = None

    # ── 5. Boucle sur les conditions ──────────────────────────────────
    all_results = []

    for i, cond in enumerate(conditions, 1):
        tag = cond["tag"]
        desc = cond["desc"]

        print(f"\n{'─' * 80}")
        print(f"  [{i}/{len(conditions)}] Condition : {tag} ({desc})")
        print(f"{'─' * 80}")

        pred_csv_path = pred_dir / f"predictions_{tag}.csv"

        if args.sanity_only:
            # Mode sanity-only : charger les prédictions existantes
            if not pred_csv_path.exists():
                print(f"  ✗ Fichier introuvable : {pred_csv_path}")
                print(f"    Lancez d'abord le pipeline sans --sanity-only")
                continue
            print(f"  Chargement des prédictions : {pred_csv_path}")
            df_pred = pd.read_csv(pred_csv_path, encoding="utf-8-sig")
        else:
            # Inférence
            df_pred = df_base.copy()
            df_pred = run_inference(
                df_pred, tokenizer, model, device,
                use_context=cond["use_context"],
                no_template=not cond["use_template"],
                use_optimized_thresholds=cond["optimized_thresholds"],
                batch_size=args.batch_size,
            )

            # Sauvegarder les prédictions
            save_predictions(df_pred, pred_dir, tag)

        # Sanity check
        print(f"\n  Sanity check…")
        sanity_results = run_all_checks(df_pred, prefix="pred_")
        print_sanity_report(sanity_results, title=f"{tag} ({desc})", max_examples=2)

        # Exporter le rapport JSON
        export_sanity_report(
            sanity_results,
            sanity_dir / f"sanity_{tag}.json",
            condition_tag=tag,
        )

        all_results.append({
            "condition": cond,
            "sanity_results": sanity_results,
        })

    # ── 6. Résumé comparatif ──────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'═' * 80}")
        print(f"  RÉSUMÉ COMPARATIF")
        print(f"{'═' * 80}")

        df_summary = build_comparative_summary(all_results)
        print_comparative_summary(df_summary, n_samples)
        export_comparative_summary(df_summary, all_results, out_dir)

    # ── 7. Temps total ────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'═' * 80}")
    print(f"  ✓ Pipeline terminé en {elapsed:.1f}s")
    print(f"  Résultats dans : {out_dir}")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
