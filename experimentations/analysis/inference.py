# -*- coding: utf-8 -*-
"""
Inference — EMOTYC Model Inference & Cached Prediction Loading
═══════════════════════════════════════════════════════════════

Handles model loading, batch inference, and loading pre-computed
predictions from JSONL files.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model_onnx"
MODEL_DOWNLOAD_HINT = "Run `bash setup.sh` from the EMOTYC repository root."


# ═══════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_model_paths():
    """Resolve local ONNX artifacts, with environment overrides."""
    model_dir = Path(os.environ.get("EMOTYC_MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser()
    onnx_path = Path(os.environ.get("EMOTYC_ONNX_PATH", model_dir / "model.onnx")).expanduser()
    tokenizer_path = Path(
        os.environ.get("EMOTYC_TOKENIZER_PATH", model_dir / "tokenizer.json")
    ).expanduser()
    return onnx_path, tokenizer_path


def _require_model_artifacts(onnx_path, tokenizer_path):
    missing = [str(path) for path in (onnx_path, tokenizer_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing EMOTYC ONNX artifact(s): "
            + ", ".join(missing)
            + f". {MODEL_DOWNLOAD_HINT}"
        )


def _load_emotyc_model(device_name=None):
    """Charge le modèle EMOTYC ONNX et le tokenizer."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    onnx_path, tokenizer_path = _resolve_model_paths()
    _require_model_artifacts(onnx_path, tokenizer_path)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=512)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available_providers = ort.get_available_providers()
    requested_device = str(device_name).lower() if device_name is not None else None
    use_cuda = (
        requested_device.startswith("cuda")
        if requested_device is not None
        else "CUDAExecutionProvider" in available_providers
    )

    providers = ["CPUExecutionProvider"]
    if use_cuda and "CUDAExecutionProvider" in available_providers:
        providers.insert(0, "CUDAExecutionProvider")

    model = ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)

    device_str = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
    print(f"  ✓ Modèle EMOTYC ONNX chargé sur {device_str} ({onnx_path})")
    return tokenizer, model


def _format_input(tokenizer, sentence, prev_sentence=None, next_sentence=None,
                  use_context=False):
    """Formate l'input selon le template bca_v3."""
    eos = "</s>"
    if use_context:
        prev = prev_sentence or eos
        nxt = next_sentence or eos
        return f"before:{prev}{eos}current:{sentence}{eos}after:{nxt}{eos}"
    return f"before:{eos}current:{sentence}{eos}after:{eos}"


def _predict_batch(tokenizer, model, texts, batch_size=16):
    """Inférence par batch → matrice (N, 19) de probas sigmoid."""
    import numpy as np

    all_probs = []

    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 1

    input_names = {inp.name for inp in model.get_inputs()}

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
        max_len = max((len(enc.ids) for enc in encodings), default=1)
        max_len = max(max_len, 1)

        input_ids = np.full((len(encodings), max_len), pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)

        for row_index, encoding in enumerate(encodings):
            ids = encoding.ids or [pad_id]
            input_ids[row_index, : len(ids)] = ids
            attention_mask[row_index, : len(ids)] = 1

        inputs = {"input_ids": input_ids}
        if "attention_mask" in input_names:
            inputs["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        logits = np.asarray(model.run(None, inputs)[0], dtype=np.float32)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        probs = np.empty_like(logits, dtype=np.float32)
        positive = logits >= 0
        probs[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_x = np.exp(logits[~positive])
        probs[~positive] = exp_x / (1.0 + exp_x)

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
    tokenizer, model = _load_emotyc_model(device)

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
    all_probs_19 = _predict_batch(tokenizer, model, formatted_texts, batch_size)
    print(f"  ✓ Inférence terminée — shape: {all_probs_19.shape}")

    # Stocker probabilités et prédictions
    for gold_col, (emotyc_name, model_idx) in config.FULL_GOLD_TO_EMOTYC.items():
        proba_col = f"proba_{gold_col}"
        pred_col = f"pred_{gold_col}"
        df[proba_col] = all_probs_19[:, model_idx]

        # Seuil : fixe de 0.06 pour les modes d'expression, optimisé pour les émotions, 0.5 sinon
        if gold_col in ["Suggérée", "Désignée", "Comportementale", "Montrée"]:
            threshold = 0.06
        elif use_optimized_thresholds and gold_col in config.OPTIMIZED_THRESHOLDS:
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
