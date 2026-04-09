#!/usr/bin/env python3
"""
Associe les textes du fichier Parquet aux annotations LLM des fichiers JSONL.

La jointure se fait par correspondance exacte du texte brut :
  Parquet["TEXT"].strip()  ==  target_text extrait du prompt JSONL

Usage :
  python match_parquet_jsonl.py
  python match_parquet_jsonl.py --parquet chemin/vers/fichier.parquet
  python match_parquet_jsonl.py --outputs chemin/vers/outputs/
"""

import argparse
import json
import re
import pandas as pd
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_PARQUET = REPO_ROOT / "data" / "CyberBullyingExperiment.parquet"
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"

# Regex pour extraire le texte cible depuis le champ "prompt" des JSONL
TARGET_RE = re.compile(
    r'TARGET:\s*\[.*?\]\s*\(role=[^)]*\)\s*\(time=[^)]*\)\s*"(.+)"\s*$',
    re.MULTILINE,
)


def extract_target_text(prompt: str) -> str:
    """Extrait le texte cible depuis le prompt LLM."""
    m = TARGET_RE.search(prompt)
    return m.group(1).strip() if m else ""


def build_llm_index(outputs_dir: Path) -> dict[str, dict]:
    """Parse les JSONL dans outputs/ et construit un index texte → annotations.

    Retourne un dict indexé par le texte cible (stripped) :
        {
            "has_elong_kw": bool,
            "has_typo_kw": bool,
            "span_elong_words": set[str],
            "span_typo_words": set[str],
            "justifications": [str, ...],
            "sources": [str, ...],          # fichiers JSONL sources
        }
    Plusieurs lignes JSONL pour le même texte sont fusionnées (union).
    """
    elong_kw = {"répétition", "étirement", "allongement", "redoublement"}
    typo_kw = {"faute", "orthographe", "coquille"}

    index: dict[str, dict] = {}

    for folder in outputs_dir.iterdir():
        if not folder.is_dir():
            continue
        for jsonl_file in folder.glob("*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    parsed = row.get("parsed_json")
                    if not parsed:
                        continue

                    target_text = extract_target_text(row.get("prompt", ""))
                    if not target_text:
                        continue

                    entry = index.setdefault(target_text, {
                        "has_elong_kw": False,
                        "has_typo_kw": False,
                        "span_elong_words": set(),
                        "span_typo_words": set(),
                        "justifications": [],
                        "sources": [],
                    })
                    entry["sources"].append(str(jsonl_file))

                    for unit in parsed.get("sitemo_units", []):
                        justif = unit.get("justification", "")
                        if not justif:
                            continue
                        justif_lower = justif.lower()
                        span_text = unit.get("span_text", "").lower().strip()

                        has_e = any(kw in justif_lower for kw in elong_kw)
                        has_t = any(kw in justif_lower for kw in typo_kw)

                        if has_e:
                            entry["has_elong_kw"] = True
                            if span_text:
                                entry["span_elong_words"].add(span_text)
                        if has_t:
                            entry["has_typo_kw"] = True
                            if span_text:
                                entry["span_typo_words"].add(span_text)

                        entry["justifications"].append(justif)

    return index


def match_parquet_to_jsonl(
    df: pd.DataFrame,
    llm_index: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Associe chaque ligne du Parquet à son entrée dans l'index LLM.

    Retourne :
      - matched : DataFrame des lignes ayant une correspondance JSONL
      - unmatched : DataFrame des lignes sans correspondance
    """
    match_flags = []
    llm_data = []

    for _, row in df.iterrows():
        text_key = str(row["TEXT"]).strip()
        llm = llm_index.get(text_key)

        if llm is not None:
            match_flags.append(True)
            llm_data.append({
                "has_elong_kw": llm["has_elong_kw"],
                "has_typo_kw": llm["has_typo_kw"],
                "n_justifications": len(llm["justifications"]),
                "span_elong_words": ", ".join(sorted(llm["span_elong_words"])),
                "span_typo_words": ", ".join(sorted(llm["span_typo_words"])),
                "source_files": ", ".join(sorted(set(llm["sources"]))),
            })
        else:
            match_flags.append(False)
            llm_data.append({})

    df = df.copy()
    df["llm_matched"] = match_flags
    llm_df = pd.DataFrame(llm_data, index=df.index)
    df = pd.concat([df, llm_df], axis=1)

    matched = df[df["llm_matched"]].copy()
    unmatched = df[~df["llm_matched"]].copy()
    return matched, unmatched


def main():
    parser = argparse.ArgumentParser(description="Association textes Parquet ↔ JSONL")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                        help=f"Chemin du fichier Parquet (défaut : {DEFAULT_PARQUET})")
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS,
                        help=f"Dossier contenant les JSONL (défaut : {DEFAULT_OUTPUTS})")
    parser.add_argument("--save", type=Path, default=None,
                        help="Sauvegarder le résultat dans un fichier JSONL")
    args = parser.parse_args()

    # 1. Charger le Parquet
    print(f"Lecture de {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    df = df[df["TEXT"].astype(str).str.strip().str.len() > 0].copy()
    print(f"  {len(df)} lignes avec TEXT non vide")

    # 2. Construire l'index LLM
    print(f"\nParsing des JSONL dans {args.outputs}/...")
    llm_index = build_llm_index(args.outputs)
    print(f"  {len(llm_index)} textes distincts indexés")

    # 3. Faire l'association
    print("\nAssociation Parquet ↔ JSONL...")
    matched, unmatched = match_parquet_to_jsonl(df, llm_index)

    pct = len(matched) / len(df) * 100 if len(df) > 0 else 0
    print(f"\n{'='*50}")
    print(f"  Lignes Parquet       : {len(df)}")
    print(f"  Textes JSONL indexés : {len(llm_index)}")
    print(f"  Associées            : {len(matched)} ({pct:.1f}%)")
    print(f"  Non associées        : {len(unmatched)}")
    print(f"{'='*50}")

    # Détail des non-associées
    if len(unmatched) > 0:
        print(f"\nExemples de textes Parquet sans correspondance JSONL (max 5) :")
        for _, row in unmatched.head(5).iterrows():
            text_preview = str(row["TEXT"])[:80]
            print(f"  ID={row.get('ID', '?')} : \"{text_preview}...\"")

    # Textes JSONL orphelins (présents dans JSONL mais pas dans le Parquet)
    parquet_texts = set(df["TEXT"].astype(str).str.strip())
    orphan_jsonl = {t for t in llm_index if t not in parquet_texts}
    if orphan_jsonl:
        print(f"\nTextes JSONL sans correspondance Parquet : {len(orphan_jsonl)}")
        for t in list(orphan_jsonl)[:5]:
            print(f"  \"{t[:80]}...\"")

    # Statistiques sur les mots-clés
    if len(matched) > 0:
        n_elong = matched["has_elong_kw"].sum()
        n_typo = matched["has_typo_kw"].sum()
        print(f"\nParmi les {len(matched)} textes associés :")
        print(f"  Avec mot-clé élongation : {n_elong}")
        print(f"  Avec mot-clé typo       : {n_typo}")
        print(f"  Les deux                : {(matched['has_elong_kw'] & matched['has_typo_kw']).sum()}")
        print(f"  Aucun des deux          : {(~matched['has_elong_kw'] & ~matched['has_typo_kw']).sum()}")

    # 4. Sauvegarde optionnelle
    if args.save:
        out = args.save
        print(f"\nSauvegarde dans {out}...")
        records = []
        for _, row in df.iterrows():
            text_key = str(row["TEXT"]).strip()
            llm = llm_index.get(text_key)
            rec = {
                "id": row.get("ID"),
                "text": row["TEXT"],
                "llm_matched": llm is not None,
            }
            if llm:
                rec["has_elong_kw"] = llm["has_elong_kw"]
                rec["has_typo_kw"] = llm["has_typo_kw"]
                rec["span_elong_words"] = sorted(llm["span_elong_words"])
                rec["span_typo_words"] = sorted(llm["span_typo_words"])
                rec["n_justifications"] = len(llm["justifications"])
            records.append(rec)

        with open(out, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {len(records)} lignes écrites")


if __name__ == "__main__":
    main()
