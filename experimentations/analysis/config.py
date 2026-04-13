# -*- coding: utf-8 -*-
"""
Configuration — Constants, Paths, Label Mappings, Thresholds
═══════════════════════════════════════════════════════════════

Single source of truth for all constants used across the pipeline.
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

XLSX_PATHS = {
    "Homophobie": PROJECT_ROOT / "outputs/homophobie/homophobie_annotations_gold_flat.xlsx",
    "Obésité":    PROJECT_ROOT / "outputs/obésité/obésité_annotations_gold_flat.xlsx",
    "Racisme":    PROJECT_ROOT / "outputs/racisme/racisme_annotations_gold_flat.xlsx",
    "Religion":   PROJECT_ROOT / "outputs/religion/religion_annotations_gold_flat.xlsx",
}

TRAINING_DATA_PATH = PROJECT_ROOT / "Documentation" / "emotexttokids_gold_flat.xlsx"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experimentations" / "error_analysis_results"

# ═══════════════════════════════════════════════════════════════════════════
#  LABEL MAPPINGS  (19 labels total)
# ═══════════════════════════════════════════════════════════════════════════

# Mapping: gold column name → (EMOTYC model label name, model output index)
FULL_GOLD_TO_EMOTYC = {
    # 11 core emotions
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
    # Meta: emotional character
    "Emo":              ("Emo",              0),
    # Expression modes
    "Comportementale":  ("Comportementale",  1),
    "Désignée":         ("Designee",         2),
    "Montrée":          ("Montree",          3),
    "Suggérée":         ("Suggeree",         4),
    # Emotion types
    "Base":             ("Base",             5),
    "Complexe":         ("Complexe",         6),
}

# ── Structured label groups ───────────────────────────────────────────────

EMOTION_11 = [
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie",
]

EMOTION_12 = EMOTION_11 + ["Autre"]

MODES_4 = ["Comportementale", "Désignée", "Montrée", "Suggérée"]

TYPES_2 = ["Base", "Complexe"]

META_LABELS = ["Emo"]

ALL_19 = META_LABELS + MODES_4 + TYPES_2 + EMOTION_12

# Semantic groups for group-level operations
ANNOTATION_GROUPS = {
    "emotions_11": EMOTION_11,
    "emotions_12": EMOTION_12,
    "modes":       MODES_4,
    "types":       TYPES_2,
    "meta":        META_LABELS,
    "all_19":      ALL_19,
}

# Basic emotions (subset for Base/Complexe logic)
BASIC_EMOTIONS = ["Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse"]
COMPLEX_EMOTIONS = ["Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie", "Autre"]

# ═══════════════════════════════════════════════════════════════════════════
#  OPTIMIZED THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════

# Optimized thresholds for the 11 emotions (from emotyc_predict.py calibration)
# Template: bca_v3, add_special_tokens=False
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

# Default threshold for labels without optimized thresholds
DEFAULT_THRESHOLD = 0.5

# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

BINARY_FEATURES = [
    "elongation", "ironie", "insulte", "mépris / haine",
    "argot", "abréviation", "interjection",
]

QUALITATIVE_FEATURES = [
    "ROLE", "HATE", "TARGET", "VERBAL_ABUSE",
    "INTENTION", "CONTEXT", "SENTIMENT",
]

TEXT_FEATURES = [
    "text_length", "word_count", "pct_uppercase",
    "has_exclamation", "has_question",
]

# Valid values for each qualitative feature (for cleaning)
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
#  MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

EMOTYC_MODEL_NAME = "TextToKids/CamemBERT-base-EmoTextToKids"
EMOTYC_TOKENIZER_NAME = "camembert-base"
