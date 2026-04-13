#!/usr/bin/env python3
"""
EMOTYC Batch Predict — Script 1/3
══════════════════════════════════

Charge le modèle EMOTYC une seule fois, parcourt tous les fichiers XLSX
d'un répertoire donné, et exécute l'inférence pour une condition donnée.
Sauvegarde les résultats (probas + prédictions binaires) en CSV.

Usage :
    # Condition par défaut (bca template, seuils optimisés, sans contexte)
    python scripts/emotyc_batch_predict.py \\
        --input-dir golds/ \\
        --out-dir results/predictions

    # Condition spécifique
    python scripts/emotyc_batch_predict.py \\
        --input-dir golds/ \\
        --out-dir results/predictions \\
        --use-context --no-optimized-thresholds --no-template
"""

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Ajout du répertoire parent au path pour import de emotyc_predict ─────
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from emotyc_predict import (
    EMOTYC_LABEL2ID,
    EMOTION_ORDER,
    GOLD_TO_EMOTYC,
    MODE_ORDER,
    MODE_INDICES,
    EMO_INDEX,
    TYPE_INDICES,
    OPTIMIZED_THRESHOLDS,
    load_model,
    format_input,
    predict_batch,
)

PROJECT_ROOT = SCRIPT_DIR.parent

# ═══════════════════════════════════════════════════════════════════════════
#  LABEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

# Reverse mapping : EMOTYC label → model output index
EMOTYC_ID2LABEL = {v: k for k, v in EMOTYC_LABEL2ID.items()}

# All 19 labels in model output order (index 0..18)
ALL_LABELS_ORDERED = [EMOTYC_ID2LABEL[i] for i in range(19)]

# Gold-name mapping for the columns we need to read from XLSX
# Maps gold column name (with accents) → EMOTYC model label (ASCII)
GOLD_NAME_TO_EMOTYC = {
    "Colère":           "Colere",
    "Dégoût":           "Degout",
    "Joie":             "Joie",
    "Peur":             "Peur",
    "Surprise":         "Surprise",
    "Tristesse":        "Tristesse",
    "Admiration":       "Admiration",
    "Culpabilité":      "Culpabilite",
    "Embarras":         "Embarras",
    "Fierté":           "Fierte",
    "Jalousie":         "Jalousie",
    "Autre":            "Autre",
    "Emo":              "Emo",
    "Comportementale":  "Comportementale",
    "Désignée":         "Designee",
    "Montrée":          "Montree",
    "Suggérée":         "Suggeree",
    "Base":             "Base",
    "Complexe":         "Complexe",
}

# Inverse: EMOTYC model label → gold column name
EMOTYC_TO_GOLD_NAME = {v: k for k, v in GOLD_NAME_TO_EMOTYC.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def discover_xlsx_files(input_dir: str) -> dict[str, Path]:
    """
    Découvre tous les fichiers XLSX dans un répertoire (récursif).
    Retourne un dict {domain_name: Path}.
    Le nom de domaine est déduit du nom du fichier (sans suffixe).
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"✗ Répertoire introuvable : {input_path}")
        sys.exit(1)

    xlsx_files = sorted(input_path.rglob("*.xlsx"))
    if not xlsx_files:
        print(f"✗ Aucun fichier XLSX trouvé dans {input_path}")
        sys.exit(1)

    result = {}
    for f in xlsx_files:
        # Ignore les fichiers temporaires Excel
        if f.name.startswith("~$"):
            continue
        # Déduire le nom du domaine depuis le nom du fichier
        domain = f.stem.replace("_annotations_gold_flat", "").replace("_gold_flat", "")
        result[domain] = f

    print(f"✓ {len(result)} fichier(s) XLSX découvert(s) :")
    for domain, path in result.items():
        print(f"  {domain}: {path}")

    return result


def load_single_xlsx(xlsx_path: Path) -> pd.DataFrame:
    """Charge un XLSX et valide la présence de la colonne TEXT."""
    df = pd.read_excel(xlsx_path)

    # Trouver la colonne texte
    text_col = None
    for candidate in ("TEXT", "text", "sentence"):
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError(f"Colonne texte introuvable dans {xlsx_path} (attendu: TEXT/text/sentence)")
    if text_col != "TEXT":
        df = df.rename(columns={text_col: "TEXT"})

    df["TEXT"] = df["TEXT"].fillna("").astype(str)
    return df


def load_all_xlsx(xlsx_dict: dict[str, Path]) -> pd.DataFrame:
    """
    Charge tous les XLSX, ajoute une colonne 'domain' et un index global.
    Retourne un DataFrame concaténé.
    """
    frames = []
    for domain, xlsx_path in xlsx_dict.items():
        df = load_single_xlsx(xlsx_path)
        df["domain"] = domain
        df["_source_file"] = str(xlsx_path)
        df["_domain_idx"] = range(len(df))
        frames.append(df)
        print(f"  ✓ {domain}: {len(df)} lignes")

    df_all = pd.concat(frames, ignore_index=True)
    print(f"  Total : {len(df_all)} lignes")
    return df_all


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def build_condition_tag(use_context: bool, optimized_thresholds: bool,
                        use_template: bool) -> str:
    """Construit un tag descriptif pour une condition d'inférence."""
    ctx = "ctx1" if use_context else "ctx0"
    thr = "thr1" if optimized_thresholds else "thr0"
    tpl = "tpl1" if use_template else "tpl0"
    return f"{ctx}_{thr}_{tpl}"


def build_condition_description(use_context: bool, optimized_thresholds: bool,
                                 use_template: bool) -> str:
    """Description humainement lisible d'une condition."""
    parts = []
    parts.append("bca" if use_template else "raw")
    parts.append("opt" if optimized_thresholds else "0.5")
    if use_context:
        parts.append("+ctx")
    return ", ".join(parts)


def run_inference(df: pd.DataFrame, tokenizer, model, device,
                  use_context: bool = False,
                  no_template: bool = False,
                  use_optimized_thresholds: bool = True,
                  batch_size: int = 16) -> pd.DataFrame:
    """
    Exécute l'inférence EMOTYC sur tout le DataFrame.
    Respecte les frontières de domaine pour le contexte.
    Ajoute les colonnes proba_* et pred_* pour les 19 labels.

    Les colonnes pred_* utilisent les noms gold (avec accents) comme
    convention, cohérent avec error_analysis.py.
    """
    N = len(df)
    texts = df["TEXT"].tolist()
    domains = df["domain"].tolist()

    # ── Formater les textes, domaine par domaine ──────────────────────
    formatted_texts = [""] * N

    for domain in df["domain"].unique():
        mask = df["domain"] == domain
        idxs = df.index[mask].tolist()
        domain_texts = df.loc[mask, "TEXT"].tolist()

        for pos, global_idx in enumerate(idxs):
            prev_s = domain_texts[pos - 1] if (pos > 0 and use_context) else None
            next_s = domain_texts[pos + 1] if (pos < len(domain_texts) - 1 and use_context) else None
            formatted_texts[global_idx] = format_input(
                tokenizer, domain_texts[pos], prev_s, next_s,
                use_context=use_context, no_template=no_template,
            )

    # ── Inférence ─────────────────────────────────────────────────────
    template_name = "raw" if no_template else ("bca_context" if use_context else "bca_no_context")
    print(f"  Template : {template_name}")
    print(f"  Inférence sur {N} textes (batch_size={batch_size})…")

    all_probs_19 = predict_batch(tokenizer, model, device, formatted_texts,
                                  batch_size=batch_size)
    print(f"  ✓ Inférence terminée — shape: {all_probs_19.shape}")

    # ── Seuils ────────────────────────────────────────────────────────
    # Construire la matrice de seuils pour les 19 labels
    thresholds_19 = np.full(19, 0.5)  # défaut : 0.5

    if use_optimized_thresholds:
        for gold_name, threshold in OPTIMIZED_THRESHOLDS.items():
            emotyc_name = GOLD_NAME_TO_EMOTYC.get(gold_name)
            if emotyc_name and emotyc_name in EMOTYC_LABEL2ID:
                idx = EMOTYC_LABEL2ID[emotyc_name]
                thresholds_19[idx] = threshold

    pred_matrix = (all_probs_19 >= thresholds_19).astype(int)

    # ── Stocker dans le DataFrame ─────────────────────────────────────
    # Utiliser les noms gold (avec accents) pour les colonnes
    new_cols = {}
    for model_idx in range(19):
        emotyc_label = EMOTYC_ID2LABEL[model_idx]
        gold_name = EMOTYC_TO_GOLD_NAME.get(emotyc_label, emotyc_label)

        new_cols[f"proba_{gold_name}"] = all_probs_19[:, model_idx]
        new_cols[f"pred_{gold_name}"] = pred_matrix[:, model_idx]
        new_cols[f"threshold_{gold_name}"] = thresholds_19[model_idx]

    result_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return result_df


# ═══════════════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def save_predictions(df: pd.DataFrame, out_dir: Path, condition_tag: str):
    """
    Sauvegarde les prédictions dans un CSV unique par condition.
    Le fichier contient toutes les lignes de tous les domaines.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"predictions_{condition_tag}.csv"

    # Sélectionner les colonnes pertinentes
    meta_cols = ["domain", "_domain_idx", "_source_file", "TEXT"]
    # Colonnes gold (si présentes)
    gold_cols = [c for c in df.columns if c in GOLD_NAME_TO_EMOTYC]
    # Colonnes prédictions
    pred_cols = sorted([c for c in df.columns if c.startswith("pred_")])
    proba_cols = sorted([c for c in df.columns if c.startswith("proba_")])
    threshold_cols = sorted([c for c in df.columns if c.startswith("threshold_")])

    export_cols = meta_cols + gold_cols + proba_cols + pred_cols + threshold_cols
    export_cols = [c for c in export_cols if c in df.columns]

    df[export_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ Prédictions exportées : {out_path} ({len(df)} lignes)")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="EMOTYC Batch Predict — Inférence sur tous les XLSX d'un répertoire",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input-dir", required=True,
                    help="Répertoire contenant les fichiers XLSX gold")
    p.add_argument("--out-dir", required=True,
                    help="Dossier de sortie pour les CSV de prédictions")
    p.add_argument("--use-context", action="store_true",
                    help="Utiliser les phrases voisines (i-1, i+1) comme contexte")
    p.add_argument("--no-optimized-thresholds", action="store_true",
                    help="Utiliser un seuil fixe de 0.5 au lieu des seuils optimisés")
    p.add_argument("--no-template", action="store_true",
                    help="Phrase brute sans template bca")
    p.add_argument("--batch-size", type=int, default=16,
                    help="Taille du batch pour l'inférence (défaut: 16)")
    p.add_argument("--device", default=None,
                    help="Device PyTorch (défaut: auto-détection cuda/cpu)")
    return p.parse_args()


def main():
    args = parse_args()

    use_context = args.use_context
    optimized_thresholds = not args.no_optimized_thresholds
    use_template = not args.no_template

    condition_tag = build_condition_tag(use_context, optimized_thresholds, use_template)
    condition_desc = build_condition_description(use_context, optimized_thresholds, use_template)

    print(f"\n{'═' * 70}")
    print(f"  EMOTYC Batch Predict")
    print(f"  Condition : {condition_tag} ({condition_desc})")
    print(f"{'═' * 70}")

    # ── 1. Découverte des fichiers ────────────────────────────────────
    print("\n▸ 1. Découverte des fichiers XLSX…")
    xlsx_dict = discover_xlsx_files(args.input_dir)

    # ── 2. Chargement des données ─────────────────────────────────────
    print("\n▸ 2. Chargement des données…")
    df = load_all_xlsx(xlsx_dict)

    # ── 3. Chargement du modèle (une seule fois) ─────────────────────
    print("\n▸ 3. Chargement du modèle…")
    import torch
    device = torch.device(args.device) if args.device else None
    tokenizer, model, device = load_model(device)

    # ── 4. Inférence ──────────────────────────────────────────────────
    print(f"\n▸ 4. Inférence ({condition_desc})…")
    df = run_inference(
        df, tokenizer, model, device,
        use_context=use_context,
        no_template=not use_template,
        use_optimized_thresholds=optimized_thresholds,
        batch_size=args.batch_size,
    )

    # ── 5. Export ─────────────────────────────────────────────────────
    print("\n▸ 5. Export…")
    out_path = save_predictions(df, Path(args.out_dir), condition_tag)

    print(f"\n{'═' * 70}")
    print(f"  ✓ Terminé. Fichier : {out_path}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
