# -*- coding: utf-8 -*-
"""
Data Loader — Loading, Cleaning, Feature Engineering
══════════════════════════════════════════════════════

Handles all data I/O and preparation:
  - Loading 4 OOD XLSX files (cyberbullying domains)
  - Cleaning qualitative/binary/gold columns
  - Deriving text-level features
  - Computing label density features
  - Building the analysis feature matrix for RF/SHAP
"""

import numpy as np
import pandas as pd

from . import config


# ═══════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
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


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def load_and_clean_data(xlsx_paths=None):
    """
    Charge les 4 XLSX OOD, nettoie, ajoute la colonne 'domain', concatène.

    Parameters
    ----------
    xlsx_paths : dict, optional
        Mapping domain name → xlsx path. Defaults to config.XLSX_PATHS.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with domain, gold labels, and features.
    """
    if xlsx_paths is None:
        xlsx_paths = config.XLSX_PATHS

    frames = []
    for domain, xlsx_path in xlsx_paths.items():
        df = pd.read_excel(xlsx_path)
        df["domain"] = domain
        df["_original_idx"] = range(len(df))
        frames.append(df)
        print(f"  ✓ {domain}: {len(df)} lignes, {len(df.columns)} colonnes")

    df_all = pd.concat(frames, ignore_index=True)
    n_total = len(df_all)
    print(f"\n  Total : {n_total} lignes")

    # ── Nettoyage des features qualitatives ───────────────────────────
    for col in config.QUALITATIVE_FEATURES:
        if col not in df_all.columns:
            continue
        if col == "TARGET":
            df_all[col] = _clean_target_column(
                df_all[col], config.VALID_VALUES.get("TARGET", set())
            )
        else:
            df_all[col] = _clean_qualitative_column(
                df_all[col], config.VALID_VALUES.get(col)
            )

    # ── Nettoyage des features binaires ───────────────────────────────
    for col in config.BINARY_FEATURES:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce").fillna(0).astype(int)

    # ── Nettoyage des gold labels ─────────────────────────────────────
    for gold_col in config.FULL_GOLD_TO_EMOTYC:
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

    n_clean = (
        df_all[config.QUALITATIVE_FEATURES[0]].notna().sum()
        if config.QUALITATIVE_FEATURES[0] in df_all.columns
        else n_total
    )
    print(f"  Après nettoyage : {n_clean}/{n_total} lignes avec "
          f"{config.QUALITATIVE_FEATURES[0]} valide")

    return df_all


def add_text_features(df):
    """
    Ajoute des features textuelles dérivées.

    Features added: text_length, word_count, pct_uppercase,
                    has_exclamation, has_question
    """
    df["text_length"] = df["TEXT"].str.len()
    df["word_count"] = df["TEXT"].str.split().str.len()
    df["pct_uppercase"] = df["TEXT"].apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
    )
    df["has_exclamation"] = df["TEXT"].str.contains("!").astype(int)
    df["has_question"] = df["TEXT"].str.contains(r"\?").astype(int)
    return df


def add_density_features(df):
    """
    Computes label density per sample across different label groups.

    Features added:
        label_density_19   — total active labels (out of 19)
        emotion_density_12 — active emotions (out of 12)
        mode_density_4     — active modes (out of 4)
        gold_density_19    — gold label density (separate from pred)
    """
    # Gold densities
    emo_cols = [e for e in config.EMOTION_12 if e in df.columns]
    mode_cols = [m for m in config.MODES_4 if m in df.columns]
    all_cols = [l for l in config.ALL_19 if l in df.columns]

    if emo_cols:
        df["emotion_density_12"] = df[emo_cols].sum(axis=1)
    if mode_cols:
        df["mode_density_4"] = df[mode_cols].sum(axis=1)
    if all_cols:
        df["label_density_19"] = df[all_cols].sum(axis=1)

    # Prediction densities (if pred columns exist)
    pred_emo_cols = [f"pred_{e}" for e in config.EMOTION_12 if f"pred_{e}" in df.columns]
    pred_mode_cols = [f"pred_{m}" for m in config.MODES_4 if f"pred_{m}" in df.columns]
    if pred_emo_cols:
        df["pred_emotion_density"] = df[pred_emo_cols].sum(axis=1)
    if pred_mode_cols:
        df["pred_mode_density"] = df[pred_mode_cols].sum(axis=1)

    return df


def build_analysis_features(df):
    """
    Construit la matrice de features explicatives pour le modèle de diagnostic.

    Returns
    -------
    X_df : pd.DataFrame
        Numeric feature matrix (no NaN).
    feature_names : list[str]
        Ordered feature names.
    """
    parts = []
    feature_names = []

    # A) Features binaires
    for col in config.BINARY_FEATURES:
        if col in df.columns:
            parts.append(df[col].fillna(0).astype(int).values.reshape(-1, 1))
            feature_names.append(col)

    # B) Features textuelles dérivées
    for col in config.TEXT_FEATURES:
        if col in df.columns:
            parts.append(df[col].fillna(0).values.reshape(-1, 1))
            feature_names.append(col)

    # C) Domain (one-hot)
    domain_dummies = pd.get_dummies(df["domain"], prefix="domain")
    parts.append(domain_dummies.values)
    feature_names.extend(domain_dummies.columns.tolist())

    # D) Features qualitatives (one-hot, with MISSING category)
    for col in config.QUALITATIVE_FEATURES:
        if col not in df.columns:
            continue
        series = df[col].astype(str).replace("nan", "MISSING").fillna("MISSING")
        dummies = pd.get_dummies(series, prefix=col)
        parts.append(dummies.values)
        feature_names.extend(dummies.columns.tolist())

    X = np.hstack(parts)
    X_df = pd.DataFrame(X, columns=feature_names, index=df.index)
    print(f"  Matrice de features : {X_df.shape[0]} lignes × {X_df.shape[1]} features")
    return X_df, feature_names
