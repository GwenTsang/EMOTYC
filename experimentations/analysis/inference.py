# -*- coding: utf-8 -*-
"""
Inference — EMOTYC Model Inference & Cached Prediction Loading
═══════════════════════════════════════════════════════════════

Handles model loading, batch inference, and loading pre-computed
predictions from JSONL files.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# ═══════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _load_emotyc_model(device_name=None):
    """Charge le modèle EMOTYC et le tokenizer."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = (
        torch.device(device_name) if device_name
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(config.EMOTYC_TOKENIZER_NAME)
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(config.EMOTYC_MODEL_NAME)
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


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def run_emotyc_inference(df, use_context=False, use_optimized_thresholds=True,
                         batch_size=16, device=None):
    """
    Exécute l'inférence EMOTYC sur tout le DataFrame.
    Ajoute les colonnes pred_* et proba_* pour chaque label.
    Respecte les frontières de domaine pour le contexte.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'TEXT' and 'domain' columns.
    use_context : bool
        Use neighboring sentences (i-1, i+1) as context.
    use_optimized_thresholds : bool
        Use optimized thresholds for the 11 emotions.
    batch_size : int
        Inference batch size.
    device : str or None
        PyTorch device string.

    Returns
    -------
    pd.DataFrame
        Input df augmented with pred_* and proba_* columns.
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
    for gold_col, (emotyc_name, model_idx) in config.FULL_GOLD_TO_EMOTYC.items():
        proba_col = f"proba_{gold_col}"
        pred_col = f"pred_{gold_col}"
        df[proba_col] = all_probs_19[:, model_idx]

        # Seuil : optimisé pour les 11 émotions, 0.5 sinon
        if use_optimized_thresholds and gold_col in config.OPTIMIZED_THRESHOLDS:
            threshold = config.OPTIMIZED_THRESHOLDS[gold_col]
        else:
            threshold = config.DEFAULT_THRESHOLD
        df[pred_col] = (all_probs_19[:, model_idx] >= threshold).astype(int)

    return df


def load_cached_predictions(df, predictions_dir):
    """
    Charge des JSONL de prédictions produites par emotyc_predict.py.
    Joint les prédictions au DataFrame d'analyse.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'domain' column.
    predictions_dir : str or Path
        Directory containing domain subdirectories with JSONL files.

    Returns
    -------
    pd.DataFrame
        Input df augmented with pred_* and proba_* columns.
    """
    domain_dirs = {
        "Homophobie": "homophobie",
        "Obésité":    "obésité",
        "Racisme":    "racisme",
        "Religion":   "religion",
    }

    for domain, subdir in domain_dirs.items():
        # Chercher le JSONL dans le dossier emotyc_eval
        jsonl_candidates = list(
            Path(predictions_dir).glob(f"{subdir}/**/emotyc_predictions.jsonl")
        )
        if not jsonl_candidates:
            raise FileNotFoundError(
                f"JSONL introuvable pour {domain} dans {predictions_dir}/{subdir}/. "
                f"Exécutez d'abord: python scripts/emotyc_predict.py --xlsx ... "
                f"--out_dir outputs/{subdir}/emotyc_eval"
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
                f"Mismatch {domain}: {len(records)} prédictions "
                f"vs {len(domain_idx)} lignes gold"
            )

        for i, (global_idx, rec) in enumerate(zip(domain_idx, records)):
            # Émotions
            for emo in config.EMOTION_11:
                if emo in rec.get("preds", {}):
                    df.at[global_idx, f"pred_{emo}"] = rec["preds"][emo]
                    df.at[global_idx, f"proba_{emo}"] = rec["probas"].get(emo, np.nan)
            # Modes
            for mode in config.MODES_4:
                emotyc_mode = mode.replace("é", "e").replace("è", "e")
                if emotyc_mode in rec.get("preds_mode", {}):
                    df.at[global_idx, f"pred_{mode}"] = rec["preds_mode"][emotyc_mode]
                    df.at[global_idx, f"proba_{mode}"] = rec["probas_mode"].get(
                        emotyc_mode, np.nan
                    )
            # Emo
            if "pred_emo" in rec:
                df.at[global_idx, "pred_Emo"] = rec["pred_emo"]
                df.at[global_idx, "proba_Emo"] = rec.get("proba_emo", np.nan)
            # Type
            for t in config.TYPES_2:
                if t in rec.get("preds_type", {}):
                    df.at[global_idx, f"pred_{t}"] = rec["preds_type"][t]
                    df.at[global_idx, f"proba_{t}"] = rec["probas_type"].get(t, np.nan)
            # Autre
            if "Autre" in rec.get("preds", {}):
                df.at[global_idx, "pred_Autre"] = rec["preds"]["Autre"]
                df.at[global_idx, "proba_Autre"] = rec["probas"].get("Autre", np.nan)

    print(f"  ✓ Prédictions chargées depuis {predictions_dir}")
    return df


def run_or_load(df, args):
    """
    Convenience wrapper: runs inference or loads cached predictions
    based on CLI args.

    Parameters
    ----------
    df : pd.DataFrame
    args : argparse.Namespace
        Must have: skip_inference, predictions_dir, use_context,
        no_optimized_thresholds, batch_size, device

    Returns
    -------
    pd.DataFrame
    """
    if args.skip_inference:
        return load_cached_predictions(df, args.predictions_dir)
    else:
        return run_emotyc_inference(
            df,
            use_context=args.use_context,
            use_optimized_thresholds=not args.no_optimized_thresholds,
            batch_size=args.batch_size,
            device=args.device,
        )
