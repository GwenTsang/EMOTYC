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
    "CyberAggAdo": PROJECT_ROOT / "data" / "CyberAdoAgg_gold_global_total_latest.xlsx",
}

XLSX_FILENAMES_BY_DOMAIN = {
    "Homophobie": [
        "homophobie_annotations_gold_flat.xlsx",
        "homophobie_annotations_gold_flat_updated.xlsx",
    ],
    "Obésité": [
        "obésité_annotations_gold_flat.xlsx",
        "obésité_annotations_gold_flat_updated.xlsx",
    ],
    "Racisme": [
        "racisme_annotations_gold_flat.xlsx",
        "racisme_annotations_gold_flat_updated.xlsx",
    ],
    "Religion": [
        "religion_gold_flat.xlsx",
        "religion_annotations_gold_flat.xlsx",
        "religion_annotations_gold_flat_updated.xlsx",
    ],
}

DOMAIN_ALIASES = {
    "homophobie": "Homophobie",
    "obesite": "Obésité",
    "obésité": "Obésité",
    "racisme": "Racisme",
    "religion": "Religion",
    "cyberaggado": "CyberAggAdo",
}


def canonicalize_domain_name(name):
    """Return the canonical display name for a domain."""
    key = str(name).strip()
    return DOMAIN_ALIASES.get(key.lower(), key)


def resolve_xlsx_paths(xlsx_dir=None, overrides=None):
    """Resolve the four domain XLSX paths from defaults, a directory, and/or explicit overrides."""
    paths = {domain: Path(path) for domain, path in XLSX_PATHS.items()}

    if xlsx_dir is not None:
        root = Path(xlsx_dir)
        if not root.exists():
            raise FileNotFoundError(f"Input XLSX directory not found: {root}")

        for domain, filenames in XLSX_FILENAMES_BY_DOMAIN.items():
            slug = domain.lower()
            candidates = [root / filename for filename in filenames]
            candidates.extend(root.glob(f"{slug}/**/*.xlsx"))
            candidates.extend(root.glob(f"**/{slug}*.xlsx"))

            chosen = next((candidate for candidate in candidates if Path(candidate).exists()), None)
            if chosen is None:
                raise FileNotFoundError(
                    f"Could not resolve an XLSX file for domain '{domain}' from directory: {root}"
                )
            paths[domain] = Path(chosen)

    if overrides:
        for domain, path in overrides.items():
            if path is None:
                continue
            canonical_domain = canonicalize_domain_name(domain)
            if canonical_domain not in XLSX_PATHS:
                raise KeyError(f"Unknown domain for XLSX override: {domain}")
            paths[canonical_domain] = Path(path)

    missing = [f"{domain}: {path}" for domain, path in paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Some XLSX inputs do not exist:\n - " + "\n - ".join(missing)
        )

    return paths
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

# Some consolidated XLSX exports use the model's ASCII label names for gold
# columns. Normalize them to the canonical annotation names above.
GOLD_COLUMN_ALIASES = {
    "Colere": "Colère",
    "Degout": "Dégoût",
    "Culpabilite": "Culpabilité",
    "Fierte": "Fierté",
    "Designee": "Désignée",
    "Montree": "Montrée",
    "Suggeree": "Suggérée",
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
    "Admiration":  0.95,
    "Colère":      0.28,
    "Culpabilité": 0.12,
    "Dégoût":      0.19,
    "Embarras":    0.95,
    "Fierté":      0.80,
    "Jalousie":    0.01,
    "Joie":        0.91,
    "Peur":        0.98,
    "Surprise":    0.97,
    "Tristesse":   0.69,
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
    "INTENTION", "CONTEXT", "SENTIMENT", "nature_linguistique",
]

TEXT_FEATURES = [
    "text_length", "word_count", "pct_uppercase",
    "has_exclamation", "has_question",
    "mean_span_avg_tok_len", "n_frag_words_in_spans",
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
