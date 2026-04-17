# EMOTYC — Évaluation et Analyse d'Erreurs du modèle CamemBERT-base-EmoTextToKids

Pipeline d'évaluation, d'analyse d'erreurs et de vérification de cohérence du modèle [TextToKids/CamemBERT-base-EmoTextToKids](https://huggingface.co/TextToKids/CamemBERT-base-EmoTextToKids), appliqué au corpus [CyberAgression-Large](https://github.com/aollagnier/CyberAgression-Large) qui contient des messages de cyberharcèlement en français, rédigés par des jeunes entre 11 et 28 ans.


## Table des matières

- [Contexte](#contexte)
- [Le modèle EMOTYC](#le-modèle-emotyc)
- [Architecture du repository](#architecture-du-repository)
- [Données](#données)
- [Scripts principaux](#scripts-principaux)
  - [`inference.py` — Inférence unitaire et comparaison au gold](#1-inferencepy--inférence-unitaire-et-comparaison-au-gold)
  - [`emotyc_batch_predict.py` — Inférence batch multi-fichiers](#2-emotyc_batch_predictpy--inférence-batch-multi-fichiers)
  - [`emotyc_sanity_check.py` — Vérification de cohérence logique](#3-emotyc_sanity_checkpy--vérification-de-cohérence-logique)
  - [`emotyc_pipeline.py` — Orchestrateur du pipeline complet](#4-emotyc_pipelinepy--orchestrateur-du-pipeline-complet)
  - [`ArgillaAnnotations.py` — Export vers Argilla](#5-argillaannotationspy--export-vers-argilla)
- [Scripts d'expérimentation](#scripts-dexpérimentation)
  - [`distribution_analysis.py` — Analyse distributionnelle comparative](#6-distribution_analysispy--analyse-distributionnelle-comparative)
  - [`error_analysis.py` — Pipeline d'analyse d'erreurs modulaire](#7-error_analysispy--pipeline-danalyse-derreurs-modulaire)
- [Conditions expérimentales](#conditions-expérimentales)
- [Template BCA](#template-bca)
- [Seuils de décision](#seuils-de-décision)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Documentation](#documentation)


## Contexte

Ce repository a pour objectif d'évaluer les performances et la cohérence structurelle du modèle EMOTYC lorsqu'il est appliqué à des données hors de sa distribution d'entraînement. Le modèle a été entraîné sur le corpus EmoTextToKids (textes narratifs destinés aux enfants) et est ici testé sur quatre corpus de cyberharcèlement en français, couvrant quatre domaines thématiques : homophobie, obésité, racisme et religion.

L'analyse porte sur trois axes principaux :

1. **Évaluation des performances** — comparaison des prédictions EMOTYC aux annotations gold (humaines) via des métriques standard (F1, précision, rappel, kappa de Cohen, exact match).
2. **Vérification de cohérence logique** — *sanity checks* vérifiant que les prédictions du modèle respectent les contraintes logiques internes du schéma d'annotation (ex. : si une émotion est détectée, le flag `Emo` doit être activé).
3. **Analyse d'erreurs approfondie** — identification des profils d'erreurs, des facteurs explicatifs, et des pistes d'amélioration.

## Le modèle EMOTYC

**Modèle** : [`TextToKids/CamemBERT-base-EmoTextToKids`](https://huggingface.co/TextToKids/CamemBERT-base-EmoTextToKids)
**Tokenizer** : `camembert-base`
**Architecture** : CamemBERT-base avec une tête de classification multi-label (sigmoid, `problem_type=multi_label_classification`)
**Fine-tuning** : `add_special_tokens=False` (la tête de classification lit le premier subword token du texte, pas le token `<s>`)

### Les 19 labels du modèle

Le modèle produit un vecteur de 19 logits, organisés en 5 groupes sémantiques :

| Index | Label EMOTYC | Groupe | Description |
|:-----:|:-------------|:-------|:------------|
| 0 | `Emo` | Méta | Caractère émotionnel global du segment |
| 1 | `Comportementale` | Mode d'expression | Émotion exprimée par un comportement |
| 2 | `Designee` | Mode d'expression | Émotion nommée explicitement |
| 3 | `Montree` | Mode d'expression | Émotion montrée par des indices verbaux |
| 4 | `Suggeree` | Mode d'expression | Émotion sous-entendue, implicite |
| 5 | `Base` | Type | Émotion de base (primaire) |
| 6 | `Complexe` | Type | Émotion complexe (secondaire) |
| 7 | `Admiration` | Émotion complexe | — |
| 8 | `Autre` | Émotion | Catégorie résiduelle |
| 9 | `Colere` | Émotion de base | — |
| 10 | `Culpabilite` | Émotion complexe | — |
| 11 | `Degout` | Émotion de base | — |
| 12 | `Embarras` | Émotion complexe | — |
| 13 | `Fierte` | Émotion complexe | — |
| 14 | `Jalousie` | Émotion complexe | — |
| 15 | `Joie` | Émotion de base | — |
| 16 | `Peur` | Émotion de base | — |
| 17 | `Surprise` | Émotion de base | — |
| 18 | `Tristesse` | Émotion de base | — |

> **Note** : Les noms des colonnes dans les fichiers gold utilisent les accents français (ex. `Colère`, `Dégoût`, `Fierté`, `Désignée`, `Montrée`, `Suggérée`), tandis que les labels internes du modèle sont en ASCII. Les scripts gèrent automatiquement ce mapping.

### Contraintes logiques du schéma d'annotation

Les 19 labels ne sont pas indépendants. Ils sont liés par des implications logiques :

1. **`Emo` ↔ Émotions** : `Emo=1` si et seulement s'il existe au moins une émotion active.
2. **`Base` ↔ Émotions de base** : `Base=1` si et seulement s'il existe au moins une émotion de base (Colère, Dégoût, Joie, Peur, Surprise, Tristesse).
3. **`Complexe` ↔ Émotions complexes** : `Complexe=1` si et seulement s'il existe au moins une émotion complexe (Admiration, Culpabilité, Embarras, Fierté, Jalousie).
4. **Modes ↔ Émotions** : si des émotions sont actives (E>0), au moins un mode doit l'être (M≥1) ; si aucune émotion (E=0), aucun mode (M=0) ; le nombre de modes ne doit pas excéder le nombre d'émotions (M ≤ E).

---

## Architecture du repository

```
EMOTYC/
├── scripts/                          # Scripts principaux du pipeline
│   ├── inference.py                  # Inférence unitaire + comparaison gold
│   ├── emotyc_batch_predict.py       # Inférence batch multi-fichiers (Script 1/3)
│   ├── emotyc_sanity_check.py        # Vérification de cohérence logique (Script 2/3)
│   ├── emotyc_pipeline.py            # Orchestrateur pipeline complet (Script 3/3)
│   └── ArgillaAnnotations.py         # Export des annotations vers Argilla
│
├── experimentations/                 # Scripts d'expérimentation et d'analyse
│   ├── distribution_analysis.py      # Analyse distributionnelle comparative
│   ├── error_analysis.py             # Orchestrateur de l'analyse d'erreurs
│   ├── analysis/                     # Sous-modules de l'analyse d'erreurs
│   │   ├── __init__.py
│   │   ├── config.py                 # Constantes, chemins, mappings de labels
│   │   ├── data_loader.py            # Chargement, nettoyage, feature engineering
│   │   ├── inference.py              # Inférence EMOTYC + cache des prédictions
│   │   ├── metrics.py                # Métriques d'erreur (Hamming, Jaccard, Brier)
│   │   ├── conditional.py            # Analyse conditionnelle modes ↔ émotions
│   │   ├── logit_analysis.py         # Distribution des logits, threshold sweep
│   │   ├── stratification.py         # Stratification densité / longueur / domaine
│   │   ├── explainability.py         # Random Forest + SHAP, règles d'association
│   │   ├── visualization.py          # Toutes les fonctions de visualisation
│   │   └── report.py                 # Génération de rapports textuels
│   ├── error_analysis_results/       # Résultats de l'analyse d'erreurs
│   └── AnalysePeformanceEmotycLLMJudge/  # Évaluation complémentaire par LLM-judge
│
├── data/                             # Données sources (corpus bruts)
│   ├── CyberBullyingExperiment.parquet   # Corpus complet cyberharcèlement
│   ├── homophobie.xlsx               # Sous-corpus homophobie
│   ├── obésité.xlsx                  # Sous-corpus obésité
│   ├── racisme.xlsx                  # Sous-corpus racisme
│   └── religion.xlsx                 # Sous-corpus religion
│
├── golds/                            # Annotations gold (vérité terrain)
│   ├── homophobie/
│   │   └── homophobie_annotations_gold_flat.xlsx
│   ├── obésité/
│   │   └── obésité_annotations_gold_flat.xlsx
│   ├── racisme/
│   │   └── racisme_annotations_gold_flat.xlsx
│   └── religion/
│       └── religion_gold_flat.xlsx
│
├── results/                          # Résultats des évaluations
│   └── sanity_gold/                  # Rapports sanity check sur les gold labels
│       ├── sanity_homophobie_annotations_gold_flat.json
│       ├── sanity_obésité_annotations_gold_flat.json
│       ├── sanity_racisme_annotations_gold_flat.json
│       └── sanity_religion_gold_flat.json
│
├── Documentation/                    # Documentation de référence
│   ├── Features.md                   # Description des features du corpus CyberBullying
│   ├── EMOTYC_template.md            # Expérimentation H1 vs H2 (add_special_tokens)
│   ├── sanity_checks.md              # Analyse détaillée des résultats de sanity check
│   └── emotexttokids_gold_flat.xlsx  # Gold labels du corpus d'entraînement EmoTextToKids
│
├── requirements.txt                  # Dépendances Python
├── .gitignore
└── README.md
```

---

## Données

### Corpus source (`data/`)

Le corpus `CyberBullyingExperiment.parquet` contient des conversations de cyberharcèlement en français, annotées sur plusieurs niveaux (rôle, agressivité, cible, abus verbal, intention, contexte, sentiment). Il est découpé en quatre sous-corpus thématiques au format XLSX.

Chaque ligne correspond à un message dans une conversation et contient :
- **`TEXT`** — le texte du message
- **`ROLE`** — le rôle de l'auteur (bully, victim, bystander-defender, bystander-assistant, conciliator)
- **`HATE`** — le niveau d'agressivité (OAG = overt, CAG = covert, NAG = non-agressif)
- **`TARGET`**, **`VERBAL_ABUSE`**, **`INTENTION`**, **`CONTEXT`**, **`SENTIMENT`** — annotations sociolinguistiques complémentaires
- **Features binaires** — `elongation`, `ironie`, `insulte`, `mépris / haine`, `argot`, `abréviation`, `interjection`
- **11 émotions** + **`Autre`** — annotations gold binaires (0/1)
- **4 modes d'expression** — `Comportementale`, `Désignée`, `Montrée`, `Suggérée`
- **Méta-labels** — `Emo`, `Base`, `Complexe`

### Annotations gold (`golds/`)

Les fichiers gold sont des XLSX « aplatis » (*gold flat*) contenant les annotations de référence humaines, avec une colonne binaire par label. Ces fichiers servent de vérité terrain pour l'évaluation des prédictions EMOTYC.

| Domaine | Fichier |
|:--------|:--------|
| Homophobie | `homophobie_annotations_gold_flat.xlsx` |
| Obésité | `obésité_annotations_gold_flat.xlsx` |
| Racisme | `racisme_annotations_gold_flat.xlsx` |
| Religion | `religion_gold_flat.xlsx` |

---

## Scripts principaux

### 1. `inference.py` — Inférence unitaire et comparaison au gold

**Rôle** : Charge le modèle EMOTYC, applique les prédictions sur chaque ligne d'un fichier gold label, compare les résultats aux annotations humaines, et exporte des métriques détaillées.

**Fonctionnalités** :
- Inférence batch sur GPU/CPU avec `torch.no_grad()`
- Support de trois modes de formatage du texte :
  - **`raw`** — phrase brute sans transformation
  - **`bca_no_context`** — template BCA sans phrases voisines
  - **`bca_context`** — template BCA avec phrases voisines (i-1, i+1)
- Évaluation sur 4 dimensions : émotions (11), Autre, modes d'expression (4), Emo, Base/Complexe
- Calcul de métriques par label : accuracy, kappa de Cohen, F1, précision, rappel
- Métriques globales : macro-F1, micro-F1, exact match
- Identification des divergences individuelles (faux positifs / faux négatifs) avec probabilités
- Seuil configurable pour les modes d'expression (`--mode-threshold`)

**Sorties** :
- `emotyc_predictions.jsonl` — résultats détaillés ligne par ligne
- `emotyc_predictions_output.xlsx` — prédictions binaires exportées
- `emotyc_predictions_summary.json` — résumé des métriques

**Usage** :

```bash
# Condition recommandée pour les données OOD
python scripts/inference.py \
    --xlsx golds/racisme/racisme_annotations_gold_flat.xlsx \
    --out_dir results/racisme_eval \
    --no-optimized-thresholds \
    --mode-threshold 0.06

# Avec seuils optimisés et contexte
python scripts/inference.py \
    --xlsx golds/homophobie/homophobie_annotations_gold_flat.xlsx \
    --out_dir results/homophobie_eval \
    --use-context \
    --batch-size 32
```

**Arguments** :

| Argument | Description | Défaut |
|:---------|:------------|:-------|
| `--xlsx` | Chemin vers le fichier gold label (.xlsx) | *requis* |
| `--out_dir` | Dossier de sortie | *requis* |
| `--use-context` | Utiliser les phrases voisines comme contexte | `False` |
| `--no-template` | Phrase brute sans template BCA | `False` |
| `--batch-size` | Taille du batch d'inférence | `16` |
| `--device` | Device PyTorch (`cuda`, `cpu`) | auto-détection |
| `--mode-threshold` | Seuil pour les modes d'expression | `0.5` |


### 2. `emotyc_batch_predict.py` — Inférence batch multi-fichiers

**Rôle** : Script 1/3 du pipeline modulaire. Charge le modèle une seule fois, parcourt tous les fichiers XLSX d'un répertoire, et exécute l'inférence pour une condition donnée. Respecte les frontières de domaine pour le contexte (les phrases voisines ne traversent pas les limites entre domaines).

**Fonctionnalités** :
- Découverte récursive des fichiers XLSX (`rglob("*.xlsx")`)
- Attribution automatique du domaine à partir du nom de fichier
- Génération d'un tag de condition descriptif (ex. `ctx0_thr1_tpl1`)
- Stockage des probabilités, prédictions binaires et seuils pour les 19 labels
- Export en CSV unique par condition

**Usage** :

```bash
# Condition par défaut (bca template, seuils optimisés, sans contexte)
python scripts/emotyc_batch_predict.py \
    --input-dir golds/ \
    --out-dir results/predictions

# Condition spécifique
python scripts/emotyc_batch_predict.py \
    --input-dir golds/ \
    --out-dir results/predictions \
    --use-context --no-optimized-thresholds --no-template
```


### 3. `emotyc_sanity_check.py` — Vérification de cohérence logique

**Rôle** : Script 2/3 du pipeline modulaire. Vérifie que les prédictions (ou les gold labels) respectent les contraintes logiques internes du schéma d'annotation EMOTYC. Fonctionne de manière agnostique via un préfixe configurable (`pred_` pour les prédictions, `""` pour les gold labels).

**3 vérifications implémentées** :

| Check | Règle | Violations détectées |
|:------|:------|:---------------------|
| **Emo ↔ Émotions** | `Emo=1` ⟺ `∃ émotion active` | Émotion sans Emo, Emo sans émotion |
| **Base/Complexe ↔ Émotions** | `Base=1` ⟺ `∃ émotion de base` ; idem Complexe | Base sans émotion de base, émotion de base sans Base, etc. |
| **Modes ↔ Émotions** | `E=0 ⇒ M=0`, `E>0 ⇒ M≥1`, `M ≤ E` | Mode sans émotion, émotion sans mode, M > E |

**Sorties** :
- Tableau synthétique dans la console avec nombre de violations par sous-check et exemples
- Rapport JSON exporté avec détails et comptages

**Usage** :

```bash
# Vérifier des prédictions (préfixe "pred_" par défaut)
python scripts/emotyc_sanity_check.py \
    --input results/predictions/predictions_ctx0_thr1_tpl1.csv \
    --out-dir results/sanity \
    --prefix pred_

# Vérifier gold labels (préfixe vide)
python scripts/emotyc_sanity_check.py \
    --input golds/homophobie/homophobie_annotations_gold_flat.xlsx \
    --out-dir results/sanity_gold \
    --prefix ""
```


### 4. `emotyc_pipeline.py` — Orchestrateur du pipeline complet

**Rôle** : Script 3/3. Orchestre le pipeline complet en chaînant les scripts 1 et 2. Charge le modèle une seule fois, puis itère sur toutes les combinaisons de conditions expérimentales (2³ = 8, voir [Conditions expérimentales](#conditions-expérimentales)), exécute l'inférence + le sanity check pour chacune, et produit un rapport comparatif synthétique.

**Fonctionnalités** :
- Génération automatique des 8 conditions (produit cartésien de 3 paramètres booléens)
- Ou restriction à une seule condition via les flags CLI
- Mode `--sanity-only` pour relancer uniquement les sanity checks sans réinférer
- Rapport comparatif avec tableau des violations, pourcentage de cohérence, et identification de la meilleure condition
- Export en CSV et JSON structuré

**Usage** :

```bash
# Pipeline complet (8 conditions)
python scripts/emotyc_pipeline.py \
    --input-dir golds/ \
    --out-dir results/sanity_pipeline

# Une seule condition
python scripts/emotyc_pipeline.py \
    --input-dir golds/ \
    --out-dir results/sanity_pipeline \
    --use-context --no-optimized-thresholds

# Sanity check uniquement (sans ré-inférer)
python scripts/emotyc_pipeline.py \
    --input-dir golds/ \
    --out-dir results/sanity_pipeline \
    --sanity-only
```


### 5. `ArgillaAnnotations.py` — Export vers Argilla

**Rôle** : Migre les annotations gold (fichier XLSX avec variables binaires 0/1) vers un dataset [Argilla v2](https://docs.argilla.io/) pour permettre la révision et la correction des annotations dans une interface web dédiée.

**Fonctionnalités** :
- Connexion à une instance Argilla via API
- Création d'un dataset avec deux `MultiLabelQuestion` (émotions et modes)
- Import des suggestions pré-remplies depuis les colonnes binaires du XLSX
- Métadonnées : index de ligne, fichier source, rôle de l'auteur
- Option `--force` pour écraser un dataset existant

**Usage** :

```bash
python scripts/ArgillaAnnotations.py \
    --xlsx golds/racisme/racisme_annotations_gold_flat.xlsx \
    --api_url http://localhost:6900 \
    --api_key admin.apikey \
    --dataset racisme_gold_review \
    --workspace argilla \
    --force
```


## Scripts d'expérimentation

### 6. `distribution_analysis.py` — Analyse distributionnelle comparative

**Rôle** : Compare les distributions des 19 labels entre le corpus d'entraînement (EmoTextToKids) et les corpus OOD de cyberharcèlement. Quantifie le *distribution shift* pour expliquer les écarts de performance.

**Analyses réalisées** :
1. **Distribution par label** — prévalence (%) de chaque label dans chaque dataset
2. **Analyse de densité** — nombre moyen/médian de labels actifs par instance, distribution des densités
3. **Co-occurrence** — matrices de co-occurrence émotion×émotion et émotion×mode
4. **Profils de labels** — identification des configurations de labels rares ou jamais vues dans l'entraînement
5. **Tests Chi² par label** — significativité statistique des différences de prévalence avec V de Cramér
6. **Divergence Jensen-Shannon** — mesure de la divergence entre les distributions de prévalence

---

### 7. `error_analysis.py` — Pipeline d'analyse d'erreurs modulaire

**Rôle** : Orchestrateur d'un pipeline d'analyse d'erreurs en 9 phases et 23 étapes, structuré en sous-modules spécialisés dans `experimentations/analysis/`.

**Phases du pipeline** :

| Phase | Étapes | Module | Description |
|:-----:|:------:|:-------|:------------|
| 1 | 1–3 | `data_loader`, `inference`, `metrics` | Chargement, inférence, métriques d'erreur |
| 2 | 4 | `data_loader` | Feature engineering (densité, longueur, features textuelles) |
| 3 | 5–7 | `metrics` | Erreurs par label, violations du schéma, décomposition de Brier |
| 4 | 8–10 | `conditional` | Analyse conditionnelle modes ↔ émotions, interactions, profils de combinaisons |
| 5 | 11–13 | `logit_analysis` | Distribution des logits, threshold sweep pour les modes, calibration |
| 6 | 14–17 | `stratification` | Stratification par densité, longueur, cross-stratification, contrôle par domaine |
| 7 | 18–21 | `explainability` | Analyses univariée/bivariée, Random Forest + SHAP, règles d'association (FP-Growth) |
| 8 | 22 | `visualization` | Génération de toutes les visualisations (plots) |
| 9 | 23 | `report` | Génération du rapport textuel final |

**Sous-modules** (`experimentations/analysis/`) :

| Module | Responsabilité |
|:-------|:---------------|
| `config.py` | Constantes, chemins, mappings des 19 labels, seuils optimisés, définitions des features |
| `data_loader.py` | Chargement des 4 XLSX OOD, nettoyage des colonnes qualitatives/binaires, features textuelles dérivées (longueur, % majuscules, ponctuation), construction de la matrice de features pour RF/SHAP |
| `inference.py` | Inférence EMOTYC avec cache des prédictions pré-calculées (JSONL) |
| `metrics.py` | Hamming loss, Jaccard error, Brier score decomposition, violations du schéma d'annotation |
| `conditional.py` | Analyse conditionnelle des erreurs modes ↔ émotions, matrice d'interaction, profils de combinaisons |
| `logit_analysis.py` | Distribution des probabilités sigmoid par label, sweep de seuils pour les modes, courbes de calibration |
| `stratification.py` | Stratification des erreurs par densité de labels, longueur textuelle, et croisement densité × longueur |
| `explainability.py` | Random Forest entraîné sur les features pour prédire l'erreur Hamming, SHAP values, analyse univariée/bivariée, règles d'association FP-Growth |
| `visualization.py` | Toutes les fonctions `matplotlib`/`seaborn` de visualisation |
| `report.py` | Génération du rapport textuel final (`.txt`) |

**Usage** :

```bash
# Pipeline complet (inférence + analyse)
python experimentations/error_analysis.py

# Avec contexte
python experimentations/error_analysis.py --use-context

# Seuils fixes
python experimentations/error_analysis.py --no-optimized-thresholds

# Depuis un CSV pré-calculé (skip inférence + métriques)
python experimentations/error_analysis.py \
    --from-csv experimentations/error_analysis_results/analysis_data.csv
```


## Conditions expérimentales

Le pipeline explore systématiquement 8 conditions, correspondant au produit cartésien de 3 paramètres binaires :

| Paramètre | Valeurs | Description |
|:-----------|:--------|:------------|
| **Contexte** | `ctx0` / `ctx1` | Sans / avec phrases voisines (i-1, i+1) |
| **Seuils** | `thr0` / `thr1` | Seuil fixe 0.5 / seuils optimisés par émotion |
| **Template** | `tpl0` / `tpl1` | Phrase brute / template BCA |

**Convention de nommage** : `ctx{0|1}_thr{0|1}_tpl{0|1}`, ex. : `ctx0_thr1_tpl1` = sans contexte, seuils optimisés, avec template BCA.

La **condition canonique** est `ctx0_thr1_tpl1` (marquée ★), correspondant à la configuration la plus proche de l'usage officiel du modèle.


## Template BCA

Le template *Before-Current-After* (BCA) est le format d'entrée utilisé lors du fine-tuning d'EMOTYC. Il encadre la phrase cible avec des marqueurs de contexte :

```
before:{phrase_précédente}</s>current:{phrase_cible}</s>after:{phrase_suivante}</s>
```

**Sans contexte** (défaut pour OOD) :
```
before:</s>current:{phrase_cible}</s>after:</s>
```

**Avec contexte** :
```
before:{phrase i-1}</s>current:{phrase i}</s>after:{phrase i+1}</s>
```

> **Résultat expérimental clé** : le template BCA améliore la cohérence de +9pp en in-domain et de **+34pp en OOD**. C'est un stabilisateur structurel critique pour les données hors-domaine. En revanche, le contexte dégrade la cohérence en OOD (−9pp) alors qu'il l'améliore légèrement en in-domain (+0.3pp).



### Seuil des modes et méta-labels

Les modes d'expression (`Comportementale`, `Désignée`, `Montrée`, `Suggérée`), `Emo`, `Base`, `Complexe` et `Autre` utilisent un seuil fixe de **0.5** par défaut. Le seuil des modes est configurable via `--mode-threshold`.

---

## Installation

### Prérequis

- Python ≥ 3.10
- GPU NVIDIA avec CUDA (recommandé, mais fonctionne en CPU)

### Dépendances

```bash
pip install -r requirements.txt
```

Le modèle EMOTYC est téléchargé automatiquement depuis le HuggingFace Hub lors de la première exécution.

---

## Utilisation

### Workflow typique

```bash
# 1. Lancer le pipeline complet (8 conditions × 4 domaines)
python scripts/emotyc_pipeline.py \
    --input-dir golds/ \
    --out-dir results/sanity_pipeline \
    --batch-size 32

# 2. Analyser les erreurs en profondeur
python experimentations/error_analysis.py \
    --out-dir experimentations/error_analysis_results

# 3. Vérifier la cohérence des gold labels eux-mêmes
python scripts/emotyc_sanity_check.py \
    --input golds/religion/religion_gold_flat.xlsx \
    --out-dir results/sanity_gold \
    --prefix ""

# 4. (Optionnel) Exporter vers Argilla pour révision humaine
python scripts/ArgillaAnnotations.py \
    --xlsx golds/racisme/racisme_annotations_gold_flat.xlsx \
    --api_url http://localhost:6900 \
    --api_key admin.apikey \
    --dataset racisme_review
```

---

## Documentation

Le dossier `Documentation/` contient :

| Fichier | Description |
|:--------|:------------|
| `Features.md` | Description détaillée des 7 niveaux d'annotation du corpus CyberBullying (rôle, agressivité, cible, abus verbal, intention, contexte, sentiment) |
| `EMOTYC_template.md` | Expérimentation complète testant H1 (`add_special_tokens=True`) vs H2 (`add_special_tokens=False`) sur 8 conditions pour déterminer la configuration de fine-tuning du modèle. Verdict : **H2 confirmée** |
| `sanity_checks.md` | Analyse comparative détaillée des résultats de sanity check : in-domain (27 911 samples) vs OOD (781 samples), effet du template, du contexte, des seuils, analyse par type de violation, recommandations |
| `emotexttokids_gold_flat.xlsx` | Gold labels du corpus d'entraînement EmoTextToKids (27 911 phrases) pour comparaison distributionnelle |