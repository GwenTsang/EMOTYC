#!/usr/bin/env python3
"""
EMOTYC Error Analysis Pipeline
═══════════════════════════════

Identifie les sous-populations où le modèle EMOTYC (RoBERTa / CamemBERT-base)
échoue ou réussit sur des données Out-of-Domain de cyberharcèlement.

MÉTHODOLOGIE
────────────
1. **Métrique d'erreur principale : Hamming Error (11 émotions)**
   Pour de la classification multi-labels avec K=19 labels très déséquilibrés,
   la distance de Hamming par échantillon est la plus robuste car :
   - Elle est TOUJOURS définie (pas de division par zéro contrairement au Jaccard
     quand gold et pred sont tous les deux zéro)
   - Elle traite chaque label indépendamment, ce qui est adapté quand la plupart
     des labels sont à 0
   - Elle capture les partial matches de façon granulaire
   Formule : H_i = (1/K) * sum_k |y_ik - ŷ_ik|

2. **Métrique complémentaire : Jaccard Error**
   E_i = 1 - J(y_i, ŷ_i) avec convention J(∅, ∅) = 1
   Utile pour pondérer les erreurs en fonction du nombre de labels actifs.

3. **Métrique pondérée : Inverse-Prevalence-Weighted Hamming**
   Pondère chaque label par w_k = 1 / max(p_k, 0.01) où p_k est la prévalence.
   Cela pénalise davantage les erreurs sur les labels rares.

Pour le RF + SHAP : cible = hamming_error (continue, normalisée).
Pour l'Association Rule Mining : filtre sur hamming_error > médiane.

Usage :
    # Pipeline complet (inférence + analyse) — seuils optimisés, sans contexte
    python experimentations/error_analysis.py

    # Avec contexte voisin (i-1, i, i+1)
    python experimentations/error_analysis.py --use-context

    # Seuil 0.5 fixe
    python experimentations/error_analysis.py --no-optimized-thresholds

    # Sauter l'inférence (utiliser des JSONL pré-calculés)
    python experimentations/error_analysis.py --skip-inference --predictions-dir outputs

Dépendances supplémentaires (pip install) :
    shap, mlxtend
"""

import argparse
import json
import math
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

# Fix torch CUDA DLL loading on Windows — MUST happen before any torch import
if sys.platform == "win32":
    _torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    # Force HuggingFace offline mode on local machine (model already cached)
    os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ═══════════════════════════════════════════════════════════════════════════
#  PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent

XLSX_PATHS = {
    "Homophobie": PROJECT_ROOT / "outputs/homophobie/homophobie_annotations_gold_flat.xlsx",
    "Obésité":    PROJECT_ROOT / "outputs/obésité/obésité_annotations_gold_flat.xlsx",
    "Racisme":    PROJECT_ROOT / "outputs/racisme/racisme_annotations_gold_flat.xlsx",
    "Religion":   PROJECT_ROOT / "outputs/religion/religion_annotations_gold_flat.xlsx",
}

# ── Labels cibles EMOTYC (19 labels) ──────────────────────────────────────

# Mapping gold column → EMOTYC model label → model output index
FULL_GOLD_TO_EMOTYC = {
    # 11 émotions
    "Colère":           ("Colere",           9),
    "Dégoût":           ("Degout",          11),
    "Joie":             ("Joie",            15),
    "Peur":             ("Peur",            16),
    "Surprise":         ("Surprise",        17),
    "Tristesse":        ("Tristesse",       18),
    "Admiration":       ("Admiration",       7),
    "Culpabilité":      ("Culpabilite",     10),
    "Embarras":         ("Embarras",        12),
    "Fierté":           ("Fierte",          13),
    "Jalousie":         ("Jalousie",        14),
    # Autre
    "Autre":            ("Autre",            8),
    # Emo
    "Emo":              ("Emo",              0),
    # Modes
    "Comportementale":  ("Comportementale",  1),
    "Désignée":         ("Designee",         2),
    "Montrée":          ("Montree",          3),
    "Suggérée":         ("Suggeree",         4),
    # Type
    "Base":             ("Base",             5),
    "Complexe":         ("Complexe",         6),
}

# 11 émotions — set commun à tous les XLSX, avec seuils optimisés
EMOTION_11 = [
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie",
]

# Seuils optimisés pour les 11 émotions (issus de emotyc_predict.py)
OPTIMIZED_THRESHOLDS = {
    "Admiration":  0.9531926895718311,
    "Colère":      0.28217218720548165,
    "Culpabilité": 0.12671495241969652,
    "Dégoût":      0.19269005632824862,
    "Embarras":    0.9548280448988165,
    "Fierté":      0.8002327448859459,
    "Jalousie":    0.017136900811277365,
    "Joie":        0.9155047132251537,
    "Peur":        0.9881862235180032,
    "Surprise":    0.9722425408373772,
    "Tristesse":   0.6984491339960737,
}

# ── Features explicatives (non-cibles) ───────────────────────────────────

BINARY_FEATURES = [
    "elongation", "ironie", "insulte", "mépris / haine",
    "argot", "abréviation", "interjection",
]

QUALITATIVE_FEATURES = [
    "ROLE", "HATE", "TARGET", "VERBAL_ABUSE",
    "INTENTION", "CONTEXT", "SENTIMENT",
]

# Valeurs valides connues pour chaque feature qualitative
VALID_VALUES = {
    "ROLE":         {"bully", "victim", "bully_support", "victim_support", "conciliator"},
    "HATE":         {"OAG", "CAG", "NAG"},
    "SENTIMENT":    {"POS", "NEG", "NEU"},
    "TARGET":       {"bully", "victim", "bully_support", "victim_support", "conciliator"},
    "VERBAL_ABUSE": {"BLM", "NCG", "THR", "DNG", "OTH"},
    "INTENTION":    {"ATK", "DFN", "CNS", "AIN", "GSL", "EMP", "CR", "OTH"},
    "CONTEXT":      {"ATK", "DFN", "CNS", "AIN", "GSL", "EMP", "CR", "OTH"},
}


# ═══════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════

def _is_dirty_annotation(val):
    """Détecte les artefacts d'annotation (ex: 'File: scenario_...Majority: NULL')."""
    if not isinstance(val, str):
        return False
    return val.startswith("File: ") or "Majority: NULL" in val


def _clean_qualitative_column(series, valid_set=None):
    """Nettoie une colonne qualitative : strip, normalise, filtre les artefacts."""
    cleaned = series.copy()
    for i, val in enumerate(cleaned):
        if pd.isna(val):
            cleaned.iloc[i] = np.nan
            continue
        s = str(val).strip()
        if _is_dirty_annotation(s):
            cleaned.iloc[i] = np.nan
            continue
        if valid_set is not None:
            # Gérer les valeurs composées (ex: "victim/victim_support")
            parts = [p.strip() for p in s.split("/")]
            if all(p in valid_set for p in parts):
                cleaned.iloc[i] = s
            else:
                cleaned.iloc[i] = np.nan
        else:
            cleaned.iloc[i] = s
    return cleaned.astype("category")


def _clean_target_column(series, valid_roles):
    """Nettoie la colonne TARGET qui peut contenir des valeurs composées."""
    cleaned = series.copy()
    for i, val in enumerate(cleaned):
        if pd.isna(val):
            cleaned.iloc[i] = np.nan
            continue
        s = str(val).strip()
        if _is_dirty_annotation(s):
            cleaned.iloc[i] = np.nan
            continue
        # Extraire le premier rôle cible pour simplifier
        parts = [p.strip() for p in s.split("/")]
        valid_parts = [p for p in parts if p in valid_roles]
        if valid_parts:
            cleaned.iloc[i] = valid_parts[0]  # Premier rôle = cible principale
        else:
            cleaned.iloc[i] = np.nan
    return cleaned.astype("category")


def load_and_clean_data():
    """Charge les 4 XLSX, nettoie, ajoute la colonne 'domain', concatène."""
    frames = []
    for domain, xlsx_path in XLSX_PATHS.items():
        df = pd.read_excel(xlsx_path)
        df["domain"] = domain
        df["_original_idx"] = range(len(df))
        frames.append(df)
        print(f"  ✓ {domain}: {len(df)} lignes, {len(df.columns)} colonnes")

    df_all = pd.concat(frames, ignore_index=True)
    n_total = len(df_all)
    print(f"\n  Total : {n_total} lignes")

    # ── Nettoyage des features qualitatives ───────────────────────────
    for col in QUALITATIVE_FEATURES:
        if col not in df_all.columns:
            continue
        if col == "TARGET":
            df_all[col] = _clean_target_column(df_all[col], VALID_VALUES.get("TARGET", set()))
        else:
            df_all[col] = _clean_qualitative_column(
                df_all[col], VALID_VALUES.get(col)
            )

    # ── Nettoyage des features binaires ───────────────────────────────
    for col in BINARY_FEATURES:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce").fillna(0).astype(int)

    # ── Nettoyage des gold labels ─────────────────────────────────────
    for gold_col in FULL_GOLD_TO_EMOTYC:
        if gold_col in df_all.columns:
            df_all[gold_col] = pd.to_numeric(df_all[gold_col], errors="coerce").fillna(0)
            df_all[gold_col] = (df_all[gold_col] >= 0.5).astype(int)

    # ── Colonne TEXT ──────────────────────────────────────────────────
    text_col = None
    for candidate in ("TEXT", "text", "sentence"):
        if candidate in df_all.columns:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError("Colonne texte introuvable (TEXT/text/sentence)")
    if text_col != "TEXT":
        df_all = df_all.rename(columns={text_col: "TEXT"})
    df_all["TEXT"] = df_all["TEXT"].fillna("").astype(str)

    # ── Features textuelles dérivées ──────────────────────────────────
    df_all["text_length"] = df_all["TEXT"].str.len()
    df_all["word_count"] = df_all["TEXT"].str.split().str.len()
    df_all["pct_uppercase"] = df_all["TEXT"].apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
    )
    df_all["has_exclamation"] = df_all["TEXT"].str.contains("!").astype(int)
    df_all["has_question"] = df_all["TEXT"].str.contains(r"\?").astype(int)

    n_clean = df_all[QUALITATIVE_FEATURES[0]].notna().sum() if QUALITATIVE_FEATURES[0] in df_all.columns else n_total
    print(f"  Après nettoyage : {n_clean}/{n_total} lignes avec {QUALITATIVE_FEATURES[0]} valide")

    return df_all


# ═══════════════════════════════════════════════════════════════════════════
#  2. EMOTYC INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def _load_emotyc_model(device_name=None):
    """Charge le modèle EMOTYC et le tokenizer."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    MODEL_NAME = "TextToKids/CamemBERT-base-EmoTextToKids"
    TOKENIZER_NAME = "camembert-base"

    device = torch.device(device_name) if device_name else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(MODEL_NAME)
        .to(device)
        .eval()
    )
    print(f"  ✓ Modèle EMOTYC chargé sur {device}")
    return tokenizer, model, device


def _format_input(tokenizer, sentence, prev_sentence=None, next_sentence=None,
                  use_context=False):
    """Formate l'input selon le template bca_v3."""
    eos = tokenizer.eos_token
    if use_context:
        prev = prev_sentence or eos
        nxt = next_sentence or eos
        return f"before:{prev}{eos}current:{sentence}{eos}after:{nxt}{eos}"
    return f"before:{eos}current:{sentence}{eos}after:{eos}"


def _predict_batch(tokenizer, model, device, texts, batch_size=16):
    """Inférence par batch → matrice (N, 19) de probas sigmoid."""
    import torch
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(
            batch, return_tensors="pt", truncation=True,
            padding=True, max_length=512, add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            logits = model(**encodings).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def run_emotyc_inference(df, use_context=False, use_optimized_thresholds=True,
                         batch_size=16, device=None):
    """
    Exécute l'inférence EMOTYC sur tout le DataFrame.
    Ajoute les colonnes pred_* et proba_* pour chaque label.
    Respecte les frontières de domaine pour le contexte.
    """
    tokenizer, model, dev = _load_emotyc_model(device)

    # Préparer les textes formatés, domaine par domaine (pour le contexte)
    formatted_texts = [""] * len(df)
    for domain in df["domain"].unique():
        mask = df["domain"] == domain
        idxs = df.index[mask].tolist()
        texts = df.loc[mask, "TEXT"].tolist()
        for pos, global_idx in enumerate(idxs):
            prev_s = texts[pos - 1] if (pos > 0 and use_context) else None
            next_s = texts[pos + 1] if (pos < len(texts) - 1 and use_context) else None
            formatted_texts[global_idx] = _format_input(
                tokenizer, texts[pos], prev_s, next_s, use_context
            )

    # Inférence
    print(f"\n  Inférence sur {len(df)} textes (batch_size={batch_size})…")
    all_probs_19 = _predict_batch(tokenizer, model, dev, formatted_texts, batch_size)
    print(f"  ✓ Inférence terminée — shape: {all_probs_19.shape}")

    # Stocker probabilités et prédictions
    for gold_col, (emotyc_name, model_idx) in FULL_GOLD_TO_EMOTYC.items():
        proba_col = f"proba_{gold_col}"
        pred_col = f"pred_{gold_col}"
        df[proba_col] = all_probs_19[:, model_idx]

        # Seuil : optimisé pour les 11 émotions, 0.5 sinon
        if use_optimized_thresholds and gold_col in OPTIMIZED_THRESHOLDS:
            threshold = OPTIMIZED_THRESHOLDS[gold_col]
        else:
            threshold = 0.5
        df[pred_col] = (all_probs_19[:, model_idx] >= threshold).astype(int)

    return df


def load_cached_predictions(df, predictions_dir):
    """
    Charge des JSONL de prédictions produites par emotyc_predict.py.
    Joint les prédictions au DataFrame d'analyse.
    """
    # Mapping domain → sous-dossier de prédictions
    domain_dirs = {
        "Homophobie": "homophobie",
        "Obésité":    "obésité",
        "Racisme":    "racisme",
        "Religion":   "religion",
    }

    for domain, subdir in domain_dirs.items():
        # Chercher le JSONL dans le dossier emotyc_eval
        jsonl_candidates = list(Path(predictions_dir).glob(f"{subdir}/**/emotyc_predictions.jsonl"))
        if not jsonl_candidates:
            raise FileNotFoundError(
                f"JSONL introuvable pour {domain} dans {predictions_dir}/{subdir}/. "
                f"Exécutez d'abord: python scripts/emotyc_predict.py --xlsx ... --out_dir outputs/{subdir}/emotyc_eval"
            )
        jsonl_path = jsonl_candidates[0]

        records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        mask = df["domain"] == domain
        domain_idx = df.index[mask].tolist()

        if len(records) != len(domain_idx):
            raise ValueError(
                f"Mismatch {domain}: {len(records)} prédictions vs {len(domain_idx)} lignes gold"
            )

        for i, (global_idx, rec) in enumerate(zip(domain_idx, records)):
            # Émotions
            for emo in EMOTION_11:
                if emo in rec.get("preds", {}):
                    df.at[global_idx, f"pred_{emo}"] = rec["preds"][emo]
                    df.at[global_idx, f"proba_{emo}"] = rec["probas"].get(emo, np.nan)
            # Modes
            for mode in ["Comportementale", "Désignée", "Montrée", "Suggérée"]:
                emotyc_mode = mode.replace("é", "e").replace("è", "e")
                if emotyc_mode in rec.get("preds_mode", {}):
                    df.at[global_idx, f"pred_{mode}"] = rec["preds_mode"][emotyc_mode]
                    df.at[global_idx, f"proba_{mode}"] = rec["probas_mode"].get(emotyc_mode, np.nan)
            # Emo
            if "pred_emo" in rec:
                df.at[global_idx, "pred_Emo"] = rec["pred_emo"]
                df.at[global_idx, "proba_Emo"] = rec.get("proba_emo", np.nan)
            # Type
            for t in ["Base", "Complexe"]:
                if t in rec.get("preds_type", {}):
                    df.at[global_idx, f"pred_{t}"] = rec["preds_type"][t]
                    df.at[global_idx, f"proba_{t}"] = rec["probas_type"].get(t, np.nan)
            # Autre
            if "Autre" in rec.get("preds", {}):
                df.at[global_idx, "pred_Autre"] = rec["preds"]["Autre"]

    print(f"  ✓ Prédictions chargées depuis {predictions_dir}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  3. ERROR METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_error_metrics(df):
    """
    Calcule 4 métriques d'erreur par échantillon :
      - n_errors_11 : nombre de labels (sur 11 émotions) mal prédits
      - hamming_11  : n_errors_11 / 11
      - jaccard_error_11 : 1 - Jaccard(gold_11, pred_11)
      - weighted_hamming_11 : Hamming pondéré par 1/prevalence
    """
    # Matrice gold et pred pour les 11 émotions
    gold_cols = [e for e in EMOTION_11 if e in df.columns]
    pred_cols = [f"pred_{e}" for e in gold_cols if f"pred_{e}" in df.columns]
    eval_emotions = [e for e in gold_cols if f"pred_{e}" in df.columns]

    if not eval_emotions:
        raise ValueError("Aucune prédiction trouvée. Lancez l'inférence d'abord.")

    K = len(eval_emotions)
    gold_mat = df[eval_emotions].values.astype(int)
    pred_mat = df[[f"pred_{e}" for e in eval_emotions]].values.astype(int)

    # 1. Nombre d'erreurs brut
    errors = np.abs(gold_mat - pred_mat)
    df["n_errors_11"] = errors.sum(axis=1)

    # 2. Hamming error normalisé
    df["hamming_11"] = df["n_errors_11"] / K

    # 3. Jaccard error
    intersection = (gold_mat & pred_mat).sum(axis=1).astype(float)
    union = (gold_mat | pred_mat).sum(axis=1).astype(float)
    jaccard_score = np.where(union > 0, intersection / union, 1.0)  # J(∅,∅)=1
    df["jaccard_error_11"] = 1.0 - jaccard_score

    # 4. Prevalence-weighted Hamming
    prevalences = gold_mat.mean(axis=0)
    weights = 1.0 / np.maximum(prevalences, 0.01)  # floor à 1% pour éviter ÷0
    weights = weights / weights.sum()  # normalisation
    df["weighted_hamming_11"] = (errors * weights[np.newaxis, :]).sum(axis=1)

    # 5. Décomposition par label : FP, FN par sample
    for j, emo in enumerate(eval_emotions):
        g = gold_mat[:, j]
        p = pred_mat[:, j]
        df[f"err_{emo}"] = np.where(
            g == p, "OK",
            np.where(p > g, "FP", "FN")
        )

    # 6. Catégorie d'erreur globale
    df["error_category"] = pd.cut(
        df["hamming_11"],
        bins=[-0.001, 0, 0.1, 0.3, 1.0],
        labels=["exact_match", "low_error", "medium_error", "high_error"],
    )

    # Résumé
    print(f"\n  ═══ Résumé des erreurs (11 émotions, K={K}) ═══")
    print(f"  Hamming moyen        : {df['hamming_11'].mean():.4f}")
    print(f"  Hamming médian       : {df['hamming_11'].median():.4f}")
    print(f"  Jaccard error moyen  : {df['jaccard_error_11'].mean():.4f}")
    print(f"  Exact match rate     : {(df['n_errors_11'] == 0).mean():.4f}")
    print(f"  Weighted Hamming moy : {df['weighted_hamming_11'].mean():.4f}")
    print(f"  Distribution n_errors:")
    for ne in sorted(df["n_errors_11"].unique()):
        pct = (df["n_errors_11"] == ne).mean() * 100
        print(f"    {ne} erreurs: {pct:5.1f}%")

    return df, eval_emotions


# ═══════════════════════════════════════════════════════════════════════════
#  4. FEATURE ENGINEERING (for analysis)
# ═══════════════════════════════════════════════════════════════════════════

def build_analysis_features(df):
    """
    Construit la matrice de features explicatives pour le modèle de diagnostic.
    Retourne (X_df, feature_names) avec X numérique (pas de NaN).
    """
    parts = []
    feature_names = []

    # A) Features binaires
    for col in BINARY_FEATURES:
        if col in df.columns:
            parts.append(df[col].fillna(0).astype(int).values.reshape(-1, 1))
            feature_names.append(col)

    # B) Features textuelles dérivées
    for col in ["text_length", "word_count", "pct_uppercase",
                "has_exclamation", "has_question"]:
        if col in df.columns:
            parts.append(df[col].fillna(0).values.reshape(-1, 1))
            feature_names.append(col)

    # C) Domain (one-hot)
    domain_dummies = pd.get_dummies(df["domain"], prefix="domain")
    parts.append(domain_dummies.values)
    feature_names.extend(domain_dummies.columns.tolist())

    # D) Features qualitatives (one-hot, avec catégorie MISSING)
    for col in QUALITATIVE_FEATURES:
        if col not in df.columns:
            continue
        series = df[col].copy()
        series = series.fillna("MISSING")
        dummies = pd.get_dummies(series, prefix=col)
        parts.append(dummies.values)
        feature_names.extend(dummies.columns.tolist())

    X = np.hstack(parts)
    X_df = pd.DataFrame(X, columns=feature_names, index=df.index)
    print(f"  Matrice de features : {X_df.shape[0]} lignes × {X_df.shape[1]} features")
    return X_df, feature_names


# ═══════════════════════════════════════════════════════════════════════════
#  5. UNIVARIATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def univariate_analysis(df, out_dir):
    """
    Pour chaque feature explicative, compare la distribution de hamming_11
    entre les différents niveaux. Test statistique + box plot.
    """
    results = []
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metric = "hamming_11"
    all_features = BINARY_FEATURES + QUALITATIVE_FEATURES + ["domain"]

    for feat in all_features:
        if feat not in df.columns:
            continue

        col = df[feat].copy()
        if feat in BINARY_FEATURES:
            col = col.astype(str)
        else:
            col = col.fillna("MISSING").astype(str)

        # Grouper par niveau
        groups = {}
        for level in col.unique():
            mask = col == level
            vals = df.loc[mask, metric].dropna()
            if len(vals) >= 3:
                groups[level] = vals

        if len(groups) < 2:
            continue

        # Test statistique
        group_arrays = list(groups.values())
        if len(groups) == 2:
            stat_val, p_val = stats.mannwhitneyu(*group_arrays, alternative="two-sided")
            test_name = "Mann-Whitney U"
        else:
            stat_val, p_val = stats.kruskal(*group_arrays)
            test_name = "Kruskal-Wallis"

        # Effect size : eta² pour Kruskal-Wallis
        n_total_kw = sum(len(g) for g in group_arrays)
        eta_sq = (stat_val - len(groups) + 1) / (n_total_kw - len(groups)) if n_total_kw > len(groups) else 0

        # Stats par niveau
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

        # Box plot
        fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 5))
        plot_data = pd.DataFrame({"feature": col, "error": df[metric]})
        plot_data = plot_data[plot_data["feature"].isin(groups.keys())]
        order = sorted(groups.keys(), key=lambda l: groups[l].mean(), reverse=True)
        sns.boxplot(data=plot_data, x="feature", y="error", order=order, ax=ax,
                    palette="RdYlGn_r", showfliers=True)
        # Ajouter les moyennes
        for i, level in enumerate(order):
            m = groups[level].mean()
            ax.plot(i, m, "D", color="black", markersize=6, zorder=5)
        ax.set_title(f"Hamming Error by {feat}\n({test_name}: p={p_val:.2e}, η²={max(eta_sq,0):.3f})")
        ax.set_xlabel(feat)
        ax.set_ylabel("Hamming Error (11 émotions)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(plot_dir / f"univariate_{feat.replace(' ', '_').replace('/', '_')}.png", dpi=150)
        plt.close(fig)

    # Trier par p-value
    results.sort(key=lambda r: r["p_value"])

    print(f"\n  ═══ Analyse Univariée (top 10) ═══")
    for r in results[:10]:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        print(f"  {r['feature']:<25s}  {r['test']:<18s}  p={r['p_value']:.2e} {sig}  η²={r['eta_squared']:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  6. BIVARIATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def bivariate_analysis(df, out_dir, top_n_pairs=10):
    """
    Analyse l'interaction entre paires de features sur le hamming error.
    Produit des heatmaps pour les paires les plus informatives.
    """
    metric = "hamming_11"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cat_features = []
    for feat in QUALITATIVE_FEATURES + ["domain"]:
        if feat in df.columns:
            col = df[feat].fillna("MISSING").astype(str) if feat != "domain" else df[feat].astype(str)
            n_levels = col.nunique()
            if 2 <= n_levels <= 8:
                cat_features.append(feat)

    # Ajouter les features binaires qui ont de la variance
    for feat in BINARY_FEATURES:
        if feat in df.columns and df[feat].nunique() >= 2:
            cat_features.append(feat)

    interaction_scores = []

    for i, f1 in enumerate(cat_features):
        for f2 in cat_features[i + 1:]:
            col1 = df[f1].fillna("MISSING").astype(str) if f1 not in BINARY_FEATURES else df[f1].astype(str)
            col2 = df[f2].fillna("MISSING").astype(str) if f2 not in BINARY_FEATURES else df[f2].astype(str)

            # Crosstab des erreurs moyennes
            combined = pd.DataFrame({
                "f1": col1, "f2": col2, "error": df[metric]
            }).dropna(subset=["error"])
            ct = combined.groupby(["f1", "f2"])["error"].agg(["mean", "count"])
            ct = ct[ct["count"] >= 5]  # au moins 5 échantillons par cellule

            if len(ct) < 4:
                continue

            # Score d'interaction : variance des moyennes par cellule
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

    # Heatmaps pour les top paires
    for rank, info in enumerate(interaction_scores[:top_n_pairs]):
        f1, f2 = info["f1"], info["f2"]
        col1 = df[f1].fillna("MISSING").astype(str) if f1 not in BINARY_FEATURES else df[f1].astype(str)
        col2 = df[f2].fillna("MISSING").astype(str) if f2 not in BINARY_FEATURES else df[f2].astype(str)

        combined = pd.DataFrame({"f1": col1, "f2": col2, "error": df[metric]}).dropna(subset=["error"])
        pivot_mean = combined.pivot_table(values="error", index="f1", columns="f2", aggfunc="mean")
        pivot_count = combined.pivot_table(values="error", index="f1", columns="f2", aggfunc="count")

        fig, ax = plt.subplots(figsize=(max(7, pivot_mean.shape[1] * 1.5),
                                        max(5, pivot_mean.shape[0] * 1.0)))
        # Annotations : mean (n=count)
        annot = pivot_mean.copy().astype(str)
        for r in annot.index:
            for c in annot.columns:
                m = pivot_mean.loc[r, c] if not pd.isna(pivot_mean.loc[r, c]) else np.nan
                n = pivot_count.loc[r, c] if not pd.isna(pivot_count.loc[r, c]) else 0
                if pd.isna(m):
                    annot.loc[r, c] = ""
                else:
                    annot.loc[r, c] = f"{m:.3f}\n(n={int(n)})"

        sns.heatmap(pivot_mean, annot=annot, fmt="", cmap="RdYlGn_r",
                    vmin=0, ax=ax, linewidths=0.5)
        ax.set_title(f"Hamming Error: {f1} × {f2}\n(range={info['error_range']:.3f})")
        ax.set_xlabel(f2)
        ax.set_ylabel(f1)
        plt.tight_layout()

        fname = f"bivariate_{rank+1:02d}_{f1}_{f2}".replace(" ", "_").replace("/", "_")
        fig.savefig(plot_dir / f"{fname}.png", dpi=150)
        plt.close(fig)

    print(f"\n  ═══ Analyse Bivariée — Top {min(top_n_pairs, len(interaction_scores))} paires ═══")
    for r in interaction_scores[:top_n_pairs]:
        print(f"  {r['f1']:<20s} × {r['f2']:<20s}  "
              f"range={r['error_range']:.3f}  max={r['max_error']:.3f}  min={r['min_error']:.3f}")

    return interaction_scores


# ═══════════════════════════════════════════════════════════════════════════
#  7. RANDOM FOREST + SHAP
# ═══════════════════════════════════════════════════════════════════════════

def rf_shap_analysis(df, X_df, feature_names, out_dir):
    """
    Entraîne un Random Forest Regressor pour prédire hamming_11 à partir
    des features explicatives, puis utilise SHAP pour interpréter.
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeRegressor, export_text

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    y = df["hamming_11"].values
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

    # Feature importance bar plot
    top_k = min(25, len(feature_names))
    top_idx = sorted_idx[:top_k]
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.35)))
    ax.barh(range(top_k), importances[top_idx][::-1],
            color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_k)))
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in top_idx][::-1])
    ax.set_xlabel("MDI Importance")
    ax.set_title("Random Forest — Feature Importance (MDI)")
    plt.tight_layout()
    fig.savefig(plot_dir / "rf_feature_importance.png", dpi=150)
    plt.close(fig)

    # ── Decision Tree (interprétable) ─────────────────────────────────
    dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=15, random_state=42)
    dt.fit(X, y)
    tree_text = export_text(dt, feature_names=feature_names, max_depth=4)

    dt_path = out_dir / "decision_tree_rules.txt"
    with open(dt_path, "w", encoding="utf-8") as f:
        f.write("Decision Tree (max_depth=4) — prédiction du Hamming Error\n")
        f.write("=" * 70 + "\n")
        f.write(tree_text)
    print(f"  ✓ Arbre de décision exporté : {dt_path}")

    # ── SHAP ──────────────────────────────────────────────────────────
    try:
        import shap

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_df)

        # Summary plot
        fig, ax = plt.subplots(figsize=(12, max(8, top_k * 0.35)))
        shap.summary_plot(shap_values, X_df, max_display=25, show=False)
        plt.tight_layout()
        plt.savefig(plot_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        print("  ✓ SHAP summary plot sauvegardé")

        # SHAP bar plot (mean |SHAP|)
        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.35)))
        shap.summary_plot(shap_values, X_df, plot_type="bar", max_display=25, show=False)
        plt.tight_layout()
        plt.savefig(plot_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        print("  ✓ SHAP bar plot sauvegardé")

        # Top SHAP features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_rank = np.argsort(mean_abs_shap)[::-1]
        print(f"\n  Top 15 features (mean |SHAP|) :")
        for rank, idx in enumerate(shap_rank[:15]):
            print(f"    {rank+1:2d}. {feature_names[idx]:<35s}  {mean_abs_shap[idx]:.4f}")

        return rf, shap_values, feature_names

    except ImportError:
        print("  ⚠ Bibliothèque 'shap' non installée. Installez-la : pip install shap")
        print("    L'analyse SHAP est sautée. Les importances MDI du Random Forest restent disponibles.")
        return rf, None, feature_names


# ═══════════════════════════════════════════════════════════════════════════
#  8. ASSOCIATION RULE MINING
# ═══════════════════════════════════════════════════════════════════════════

def association_rule_analysis(df, out_dir, min_support=0.08, min_confidence=0.5,
                              min_lift=1.2, top_k=30):
    """
    FP-Growth sur les sous-populations à forte erreur.
    Découvre les itemsets fréquents associés aux échecs du modèle.
    """
    try:
        from mlxtend.frequent_patterns import fpgrowth, association_rules
    except ImportError:
        print("  ⚠ Bibliothèque 'mlxtend' non installée. Installez-la : pip install mlxtend")
        print("    L'analyse par règles d'association est sautée.")
        return None

    # Binariser les features
    items = pd.DataFrame(index=df.index)

    # Features binaires
    for col in BINARY_FEATURES:
        if col in df.columns:
            items[f"{col}=1"] = (df[col] == 1).astype(bool)

    # Features qualitatives
    for col in QUALITATIVE_FEATURES + ["domain"]:
        if col not in df.columns:
            continue
        series = df[col].fillna("MISSING").astype(str) if col != "domain" else df[col].astype(str)
        for level in series.unique():
            if level == "MISSING":
                continue
            items[f"{col}={level}"] = (series == level).astype(bool)

    # Features textuelles discrétisées
    if "text_length" in df.columns:
        q33, q66 = df["text_length"].quantile([0.33, 0.66])
        items["text_court"] = (df["text_length"] <= q33).astype(bool)
        items["text_long"] = (df["text_length"] >= q66).astype(bool)

    # ── Analyse sur le sous-ensemble HIGH ERROR ───────────────────────
    # Seuil = médiane + on prend le subset au-dessus
    median_error = df["hamming_11"].median()
    threshold = max(median_error, 1.0 / 11)  # Au moins 1 erreur
    high_error_mask = df["hamming_11"] > threshold
    n_high = high_error_mask.sum()
    print(f"\n  ═══ Association Rules (FP-Growth) ═══")
    print(f"  Seuil erreur     : hamming > {threshold:.4f}")
    print(f"  Sous-pop HIGH    : {n_high} / {len(df)} ({100*n_high/len(df):.1f}%)")

    items_high = items[high_error_mask]

    if len(items_high) < 10:
        print("  ⚠ Trop peu d'échantillons en erreur pour l'analyse par règles.")
        return None

    # FP-Growth
    freq_itemsets = fpgrowth(items_high, min_support=min_support, use_colnames=True)
    if len(freq_itemsets) == 0:
        print(f"  ⚠ Aucun itemset fréquent avec min_support={min_support}. Réduction à 0.05.")
        freq_itemsets = fpgrowth(items_high, min_support=0.05, use_colnames=True)

    if len(freq_itemsets) == 0:
        print("  ⚠ Toujours aucun itemset fréquent. Analyse sautée.")
        return None

    print(f"  Itemsets fréquents trouvés : {len(freq_itemsets)}")

    # Règles d'association
    try:
        rules = association_rules(freq_itemsets, metric="confidence",
                                  min_threshold=min_confidence, num_itemsets=len(freq_itemsets))
    except Exception:
        rules = association_rules(freq_itemsets, metric="confidence",
                                  min_threshold=min_confidence)

    if len(rules) == 0:
        print(f"  ⚠ Aucune règle avec confidence >= {min_confidence}")
        # Afficher les top itemsets à la place
        freq_itemsets = freq_itemsets.sort_values("support", ascending=False)
        print(f"\n  Top itemsets fréquents (sous-population HIGH ERROR) :")
        for _, row in freq_itemsets.head(top_k).iterrows():
            items_str = " ∧ ".join(sorted(row["itemsets"]))
            print(f"    support={row['support']:.3f}  {items_str}")
        freq_itemsets.to_csv(out_dir / "frequent_itemsets_high_error.csv", index=False)
        return freq_itemsets

    rules = rules[rules["lift"] >= min_lift]
    rules = rules.sort_values("lift", ascending=False)

    print(f"  Règles (lift >= {min_lift}) : {len(rules)}")
    print(f"\n  Top {min(top_k, len(rules))} règles (HIGH ERROR) :")
    for idx, row in rules.head(top_k).iterrows():
        ant = " ∧ ".join(sorted(row["antecedents"]))
        cons = " ∧ ".join(sorted(row["consequents"]))
        print(f"    [{ant}] → [{cons}]")
        print(f"      support={row['support']:.3f}  conf={row['confidence']:.3f}  lift={row['lift']:.2f}")

    rules.to_csv(out_dir / "association_rules_high_error.csv", index=False)

    # ── Même analyse sur LOW ERROR (pour le contraste) ────────────────
    low_error_mask = df["hamming_11"] == 0  # exact match
    items_low = items[low_error_mask]
    n_low = low_error_mask.sum()
    print(f"\n  Sous-pop EXACT MATCH (0 erreur) : {n_low} / {len(df)} ({100*n_low/len(df):.1f}%)")

    if n_low >= 10:
        freq_low = fpgrowth(items_low, min_support=min_support, use_colnames=True)
        if len(freq_low) > 0:
            freq_low = freq_low.sort_values("support", ascending=False)
            print(f"  Itemsets fréquents (EXACT MATCH) : {len(freq_low)}")
            print(f"\n  Top itemsets (sous-population EXACT MATCH) :")
            for _, row in freq_low.head(15).iterrows():
                items_str = " ∧ ".join(sorted(row["itemsets"]))
                print(f"    support={row['support']:.3f}  {items_str}")
            freq_low.to_csv(out_dir / "frequent_itemsets_exact_match.csv", index=False)

    return rules


# ═══════════════════════════════════════════════════════════════════════════
#  9. ADDITIONAL ANALYSES
# ═══════════════════════════════════════════════════════════════════════════

def per_label_error_analysis(df, eval_emotions, out_dir):
    """
    Analyse les taux FP/FN par label et par feature explicative.
    Identifie quels labels sont problématiques dans quelles sous-populations.
    """
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Taux de FP et FN globaux par émotion
    label_errors = []
    for emo in eval_emotions:
        err_col = f"err_{emo}"
        if err_col not in df.columns:
            continue
        n = len(df)
        n_fp = (df[err_col] == "FP").sum()
        n_fn = (df[err_col] == "FN").sum()
        n_ok = (df[err_col] == "OK").sum()
        prev = df[emo].mean() if emo in df.columns else np.nan
        label_errors.append({
            "label": emo,
            "FP_rate": round(n_fp / n, 4),
            "FN_rate": round(n_fn / n, 4),
            "accuracy": round(n_ok / n, 4),
            "prevalence": round(prev, 4),
            "n_FP": n_fp,
            "n_FN": n_fn,
        })
    label_errors_df = pd.DataFrame(label_errors)

    # Stacked bar chart : FP / FN / OK par label
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [r["label"] for r in label_errors]
    fp_rates = [r["FP_rate"] for r in label_errors]
    fn_rates = [r["FN_rate"] for r in label_errors]
    ok_rates = [r["accuracy"] for r in label_errors]
    x = np.arange(len(labels))
    width = 0.6
    bars_ok = ax.bar(x, ok_rates, width, label="OK (correct)", color="#4CAF50", alpha=0.8)
    bars_fp = ax.bar(x, fp_rates, width, bottom=ok_rates, label="FP (faux positif)", color="#FF9800", alpha=0.8)
    bars_fn = ax.bar(x, fn_rates, width, bottom=[o + f for o, f in zip(ok_rates, fp_rates)],
                     label="FN (faux négatif)", color="#F44336", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("Décomposition FP / FN / OK par label émotionnel")
    ax.legend(loc="upper right")
    # Ajouter prevalence en annotation
    for i, r in enumerate(label_errors):
        ax.annotate(f"p={r['prevalence']:.2f}", (i, 1.01), ha="center", fontsize=7, color="gray")
    plt.tight_layout()
    fig.savefig(plot_dir / "per_label_error_decomposition.png", dpi=150)
    plt.close(fig)

    # 2. Heatmap : taux d'erreur par label × domain
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, err_type in enumerate(["FP", "FN"]):
        mat = []
        domains_found = sorted(df["domain"].unique())
        for domain in domains_found:
            mask = df["domain"] == domain
            row = []
            for emo in eval_emotions:
                err_col = f"err_{emo}"
                if err_col in df.columns:
                    rate = (df.loc[mask, err_col] == err_type).mean()
                else:
                    rate = np.nan
                row.append(rate)
            mat.append(row)
        mat = np.array(mat)
        im = axes[ax_idx].imshow(mat, cmap="Reds", aspect="auto", vmin=0)
        axes[ax_idx].set_xticks(range(len(eval_emotions)))
        axes[ax_idx].set_xticklabels(eval_emotions, rotation=45, ha="right", fontsize=8)
        axes[ax_idx].set_yticks(range(len(domains_found)))
        axes[ax_idx].set_yticklabels(domains_found)
        axes[ax_idx].set_title(f"Taux de {err_type} par label × domaine")
        # Annotations
        for i in range(len(domains_found)):
            for j in range(len(eval_emotions)):
                val = mat[i, j]
                if not np.isnan(val):
                    axes[ax_idx].text(j, i, f"{val:.2f}", ha="center", va="center",
                                      fontsize=7, color="white" if val > 0.3 else "black")
        plt.colorbar(im, ax=axes[ax_idx], shrink=0.8)
    plt.tight_layout()
    fig.savefig(plot_dir / "error_by_label_domain.png", dpi=150)
    plt.close(fig)

    return label_errors_df


def error_distribution_plots(df, out_dir):
    """Visualisations de la distribution globale des erreurs."""
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogramme hamming_11
    ax = axes[0, 0]
    for domain in sorted(df["domain"].unique()):
        vals = df.loc[df["domain"] == domain, "hamming_11"]
        ax.hist(vals, bins=12, alpha=0.5, label=domain, density=True)
    ax.set_xlabel("Hamming Error (11 émotions)")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution de l'erreur par domaine")
    ax.legend()

    # 2. Box plot par domaine
    ax = axes[0, 1]
    sns.boxplot(data=df, x="domain", y="hamming_11", ax=ax, palette="Set2")
    ax.set_title("Hamming Error par domaine")
    ax.set_xlabel("Domaine")
    ax.set_ylabel("Hamming Error")

    # 3. Distribution du nombre d'erreurs
    ax = axes[1, 0]
    error_counts = df["n_errors_11"].value_counts().sort_index()
    ax.bar(error_counts.index, error_counts.values, color="steelblue", alpha=0.8)
    ax.set_xlabel("Nombre d'erreurs (sur 11)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution du nombre d'erreurs par exemple")

    # 4. Scatter : jaccard error vs hamming error
    ax = axes[1, 1]
    ax.scatter(df["hamming_11"], df["jaccard_error_11"],
               c=df["domain"].astype("category").cat.codes, cmap="Set2",
               alpha=0.5, s=15)
    ax.set_xlabel("Hamming Error")
    ax.set_ylabel("Jaccard Error")
    ax.set_title("Hamming vs Jaccard Error")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(plot_dir / "error_distributions.png", dpi=150)
    plt.close(fig)
    print("  ✓ Plots de distribution des erreurs sauvegardés")


# ═══════════════════════════════════════════════════════════════════════════
#  10. REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(df, eval_emotions, univar_results, bivar_results,
                    rf_model, label_errors_df, out_dir, config_str):
    """Génère un rapport textuel synthétique des résultats."""
    report_path = out_dir / "rapport_error_analysis.txt"

    lines = []
    lines.append("=" * 80)
    lines.append("  RAPPORT D'ANALYSE DES ERREURS — EMOTYC (Out-of-Domain)")
    lines.append("=" * 80)
    lines.append(f"\nConfiguration : {config_str}")
    lines.append(f"Nombre total d'exemples : {len(df)}")
    for domain in sorted(df["domain"].unique()):
        n = (df["domain"] == domain).sum()
        lines.append(f"  {domain}: {n} exemples")
    lines.append(f"Labels évalués : {len(eval_emotions)} émotions")

    # ── Métriques globales ────────────────────────────────────────────
    lines.append(f"\n{'─' * 40}")
    lines.append("MÉTRIQUES D'ERREUR GLOBALES")
    lines.append(f"{'─' * 40}")
    lines.append(f"  Hamming Error moyen    : {df['hamming_11'].mean():.4f}")
    lines.append(f"  Hamming Error médian   : {df['hamming_11'].median():.4f}")
    lines.append(f"  Jaccard Error moyen    : {df['jaccard_error_11'].mean():.4f}")
    lines.append(f"  Weighted Hamming moyen : {df['weighted_hamming_11'].mean():.4f}")
    lines.append(f"  Exact Match rate       : {(df['n_errors_11'] == 0).mean():.4f}")

    lines.append(f"\n  Par domaine :")
    for domain in sorted(df["domain"].unique()):
        m = df.loc[df["domain"] == domain, "hamming_11"]
        lines.append(f"    {domain:<15s}: mean={m.mean():.4f}  median={m.median():.4f}  "
                      f"exact_match={(m == 0).mean():.4f}")

    # ── Erreurs par label ─────────────────────────────────────────────
    if label_errors_df is not None:
        lines.append(f"\n{'─' * 40}")
        lines.append("ERREURS PAR LABEL ÉMOTIONNEL")
        lines.append(f"{'─' * 40}")
        lines.append(f"  {'Label':<15s} {'Prév':>6s} {'FP%':>6s} {'FN%':>6s} {'Acc%':>6s}")
        for _, r in label_errors_df.iterrows():
            lines.append(f"  {r['label']:<15s} {r['prevalence']:>6.2f} "
                          f"{r['FP_rate']:>6.3f} {r['FN_rate']:>6.3f} {r['accuracy']:>6.3f}")

    # ── Analyse univariée (top résultats) ─────────────────────────────
    if univar_results:
        lines.append(f"\n{'─' * 40}")
        lines.append("ANALYSE UNIVARIÉE — FEATURES SIGNIFICATIVES")
        lines.append(f"{'─' * 40}")
        for r in univar_results[:15]:
            sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
            lines.append(f"\n  Feature: {r['feature']} ({r['test']}, p={r['p_value']:.2e} {sig}, η²={r['eta_squared']:.3f})")
            for ls in r["levels"][:5]:
                lines.append(f"    {ls['level']:<25s}  n={ls['n']:>4d}  mean_err={ls['mean_error']:.4f}  ±{ls['std_error']:.4f}")

    # ── Analyse bivariée (top) ────────────────────────────────────────
    if bivar_results:
        lines.append(f"\n{'─' * 40}")
        lines.append("ANALYSE BIVARIÉE — INTERACTIONS SIGNIFICATIVES")
        lines.append(f"{'─' * 40}")
        for r in bivar_results[:10]:
            lines.append(f"  {r['f1']} × {r['f2']}: range={r['error_range']:.3f} "
                          f"(min={r['min_error']:.3f}, max={r['max_error']:.3f})")

    # ── Random Forest ─────────────────────────────────────────────────
    if rf_model is not None:
        lines.append(f"\n{'─' * 40}")
        lines.append("RANDOM FOREST — FEATURE IMPORTANCE")
        lines.append(f"{'─' * 40}")
        lines.append(f"  OOB R² : {rf_model.oob_score_:.4f}")
        imp = rf_model.feature_importances_
        sorted_idx = np.argsort(imp)[::-1]
        for rank, idx in enumerate(sorted_idx[:20]):
            fname = rf_model.feature_names_in_[idx] if hasattr(rf_model, "feature_names_in_") else f"feature_{idx}"
            lines.append(f"    {rank+1:2d}. {fname:<35s}  {imp[idx]:.4f}")

    lines.append(f"\n{'=' * 80}")
    lines.append("  FIN DU RAPPORT")
    lines.append(f"{'=' * 80}")

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n  ✓ Rapport sauvegardé : {report_path}")
    return report_text


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="EMOTYC Error Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", type=str,
                    default=str(PROJECT_ROOT / "experimentations" / "error_analysis_results"),
                    help="Dossier de sortie")
    p.add_argument("--use-context", action="store_true",
                    help="Utiliser les phrases voisines comme contexte")
    p.add_argument("--no-optimized-thresholds", action="store_true",
                    help="Seuil fixe 0.5 au lieu des seuils optimisés")
    p.add_argument("--skip-inference", action="store_true",
                    help="Sauter l'inférence, utiliser des JSONL pré-calculés")
    p.add_argument("--predictions-dir", type=str,
                    default=str(PROJECT_ROOT / "outputs"),
                    help="Dossier contenant les JSONL de prédictions (si --skip-inference)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--min-support", type=float, default=0.08,
                    help="Support minimum pour FP-Growth")
    p.add_argument("--min-confidence", type=float, default=0.5,
                    help="Confidence minimum pour les règles d'association")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_str = (
        f"context={'yes' if args.use_context else 'no'}, "
        f"thresholds={'0.5' if args.no_optimized_thresholds else 'optimized'}"
    )
    print(f"\n{'═' * 70}")
    print(f"  EMOTYC Error Analysis Pipeline")
    print(f"  Config: {config_str}")
    print(f"{'═' * 70}")

    # ── 1. Chargement des données ─────────────────────────────────────
    print("\n▸ 1. Chargement et nettoyage des données…")
    df = load_and_clean_data()

    # ── 2. Inférence (ou chargement des prédictions) ──────────────────
    print("\n▸ 2. Prédictions EMOTYC…")
    if args.skip_inference:
        df = load_cached_predictions(df, args.predictions_dir)
    else:
        df = run_emotyc_inference(
            df,
            use_context=args.use_context,
            use_optimized_thresholds=not args.no_optimized_thresholds,
            batch_size=args.batch_size,
            device=args.device,
        )

    # ── 3. Métriques d'erreur ─────────────────────────────────────────
    print("\n▸ 3. Calcul des métriques d'erreur…")
    df, eval_emotions = compute_error_metrics(df)

    # ── 4. Features d'analyse ─────────────────────────────────────────
    print("\n▸ 4. Construction des features d'analyse…")
    X_df, feature_names = build_analysis_features(df)

    # ── 5. Sauvegarde du DataFrame complet ────────────────────────────
    csv_path = out_dir / "analysis_data.csv"
    export_cols = (
        ["domain", "_original_idx", "TEXT", "text_length", "word_count"]
        + BINARY_FEATURES
        + [c for c in QUALITATIVE_FEATURES if c in df.columns]
        + eval_emotions
        + [f"pred_{e}" for e in eval_emotions if f"pred_{e}" in df.columns]
        + [f"proba_{e}" for e in eval_emotions if f"proba_{e}" in df.columns]
        + [f"err_{e}" for e in eval_emotions if f"err_{e}" in df.columns]
        + ["n_errors_11", "hamming_11", "jaccard_error_11", "weighted_hamming_11", "error_category"]
    )
    export_cols = [c for c in export_cols if c in df.columns]
    df[export_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ DataFrame exporté : {csv_path}")

    # ── 6. Visualisations de distribution ─────────────────────────────
    print("\n▸ 5. Distribution des erreurs…")
    error_distribution_plots(df, out_dir)

    # ── 7. Analyse par label ──────────────────────────────────────────
    print("\n▸ 6. Analyse des erreurs par label…")
    label_errors_df = per_label_error_analysis(df, eval_emotions, out_dir)

    # ── 8. Analyse univariée ──────────────────────────────────────────
    print("\n▸ 7. Analyse univariée…")
    univar_results = univariate_analysis(df, out_dir)

    # ── 9. Analyse bivariée ───────────────────────────────────────────
    print("\n▸ 8. Analyse bivariée…")
    bivar_results = bivariate_analysis(df, out_dir)

    # ── 10. Random Forest + SHAP ──────────────────────────────────────
    print("\n▸ 9. Random Forest + SHAP…")
    rf_model, shap_values, feat_names = rf_shap_analysis(df, X_df, feature_names, out_dir)

    # Stocker feature_names pour le rapport
    if rf_model is not None:
        rf_model.feature_names_in_ = np.array(feature_names)

    # ── 11. Association Rules ─────────────────────────────────────────
    print("\n▸ 10. Règles d'association (FP-Growth)…")
    rules = association_rule_analysis(
        df, out_dir,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
    )

    # ── 12. Rapport ───────────────────────────────────────────────────
    print("\n▸ 11. Génération du rapport…")
    report = generate_report(
        df, eval_emotions, univar_results, bivar_results,
        rf_model, label_errors_df, out_dir, config_str,
    )

    print(f"\n{'═' * 70}")
    print(f"  ✓ Analyse terminée. Résultats dans : {out_dir}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
