#!/usr/bin/env python3
"""
Enrichit les fichiers gold XLSX avec deux colonnes binaires : `typo` et `elongation`.

Les verdicts proviennent du fichier JSONL annoté :
  experimentations/elongations/elongations_annotated.jsonl

La jointure se fait par correspondance exacte du champ `texte_brut` (JSONL)
et de la colonne `TEXT` (gold XLSX), après stripping des espaces.

Usage :
  python scripts/enrich_gold_typo_elongation.py
  python scripts/enrich_gold_typo_elongation.py --dry-run
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSONL = REPO_ROOT / "experimentations" / "elongations" / "elongations_annotated.jsonl"
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"

GOLD_SUFFIX = "_annotations_gold_flat.xlsx"


def build_verdict_index(jsonl_path: Path) -> dict[str, dict[str, int]]:
    """Parse le JSONL et construit un index texte → {typo: 0|1, elongation: 0|1}.

    Un texte obtient typo=1 si au moins une ligne JSONL a verdict=="typo",
    et elongation=1 si au moins une ligne a verdict=="elongation".
    """
    index: dict[str, dict[str, int]] = defaultdict(lambda: {"typo": 0, "elongation": 0})

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text_key = row.get("texte_brut", "").strip()
            verdict = row.get("verdict", "").strip()
            if not text_key or verdict not in ("typo", "elongation"):
                continue
            index[text_key][verdict] = 1

    return dict(index)


def enrich_gold_file(
    xlsx_path: Path,
    verdict_index: dict[str, dict[str, int]],
    dry_run: bool = False,
) -> dict:
    """Ajoute les colonnes typo et elongation à un fichier gold XLSX.

    Retourne un dict avec les statistiques de matching.
    """
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    n_rows = len(df)

    # Lookup par texte strippé
    typo_vals = []
    elong_vals = []
    matched = 0

    for _, row in df.iterrows():
        text_key = str(row["TEXT"]).strip()
        entry = verdict_index.get(text_key)
        if entry is not None:
            typo_vals.append(entry["typo"])
            elong_vals.append(entry["elongation"])
            matched += 1
        else:
            typo_vals.append(0)
            elong_vals.append(0)

    df["typo"] = typo_vals
    df["elongation"] = elong_vals

    # Forcer en int
    df["typo"] = df["typo"].astype(int)
    df["elongation"] = df["elongation"].astype(int)

    stats = {
        "file": xlsx_path.name,
        "rows": n_rows,
        "matched": matched,
        "typo_1": sum(typo_vals),
        "elongation_1": sum(elong_vals),
    }

    if not dry_run:
        df.to_excel(xlsx_path, index=False, engine="openpyxl")
        print(f"  ✓ Sauvegardé : {xlsx_path}")
    else:
        print(f"  [dry-run] Non sauvegardé : {xlsx_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Enrichit les gold XLSX avec les colonnes typo et elongation"
    )
    parser.add_argument(
        "--jsonl", type=Path, default=DEFAULT_JSONL,
        help=f"Chemin du JSONL annoté (défaut : {DEFAULT_JSONL})",
    )
    parser.add_argument(
        "--outputs", type=Path, default=DEFAULT_OUTPUTS,
        help=f"Dossier contenant les sous-dossiers gold (défaut : {DEFAULT_OUTPUTS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Ne pas sauvegarder les fichiers, afficher seulement les stats",
    )
    args = parser.parse_args()

    # 1. Charger l'index des verdicts
    print(f"Lecture de {args.jsonl}...")
    verdict_index = build_verdict_index(args.jsonl)
    n_typo = sum(1 for v in verdict_index.values() if v["typo"])
    n_elong = sum(1 for v in verdict_index.values() if v["elongation"])
    n_both = sum(1 for v in verdict_index.values() if v["typo"] and v["elongation"])
    print(f"  {len(verdict_index)} textes indexés (typo={n_typo}, elongation={n_elong}, les deux={n_both})")

    # 2. Trouver et enrichir chaque gold XLSX
    print(f"\nRecherche des fichiers gold dans {args.outputs}/...")
    all_stats = []

    for folder in sorted(args.outputs.iterdir()):
        if not folder.is_dir():
            continue
        gold_file = folder / f"{folder.name}{GOLD_SUFFIX}"
        if not gold_file.exists():
            print(f"  ⚠ Pas de fichier gold trouvé : {gold_file}")
            continue

        print(f"\nTraitement de {gold_file.name}...")
        stats = enrich_gold_file(gold_file, verdict_index, dry_run=args.dry_run)
        all_stats.append(stats)

    # 3. Résumé
    print(f"\n{'='*60}")
    print(f"{'Fichier':<45} {'Rows':>5} {'Match':>5} {'Typo':>5} {'Elong':>5}")
    print(f"{'-'*60}")
    for s in all_stats:
        print(f"{s['file']:<45} {s['rows']:>5} {s['matched']:>5} {s['typo_1']:>5} {s['elongation_1']:>5}")
    print(f"{'-'*60}")
    print(f"{'TOTAL':<45} {sum(s['rows'] for s in all_stats):>5} "
          f"{sum(s['matched'] for s in all_stats):>5} "
          f"{sum(s['typo_1'] for s in all_stats):>5} "
          f"{sum(s['elongation_1'] for s in all_stats):>5}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
