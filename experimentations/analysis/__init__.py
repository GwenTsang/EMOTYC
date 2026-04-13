# -*- coding: utf-8 -*-
"""
EMOTYC Error Analysis — Modular Pipeline
═════════════════════════════════════════

Sub-package providing specialized modules for deep error analysis
of the EMOTYC (CamemBERT-base-EmoTextToKids) model on OOD
cyberbullying data.

Modules:
    config          — Constants, paths, label mappings, thresholds
    data_loader     — Data loading, cleaning, feature engineering
    inference       — EMOTYC model inference + cached prediction loading
    metrics         — Error metrics (Hamming, Jaccard, Brier, violations)
    conditional     — Conditional error analysis (modes ↔ emotions)
    logit_analysis  — Logit distributions, threshold sweep, calibration
    stratification  — Density / length / domain stratification
    explainability  — RF + SHAP, univariate, bivariate, association rules
    visualization   — All plotting functions
    report          — Report generation
"""
