#!/usr/bin/env python3
"""
Script pour migrer un fichier XLSX comprenant des variables 
booléennes (0/1) d'émotions et de modes d'expression vers un dataset Argilla v2.
"""

import argparse
import sys
import os
import pandas as pd
import argilla as rg
from tqdm import tqdm

EMOTIONS = [
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie", "Autre"
]

MODES = ["Comportementale", "Désignée", "Montrée", "Suggérée"]

def connect_argilla(api_url: str, api_key: str) -> rg.Argilla:
    """Se connecte à l'instance Argilla."""
    return rg.Argilla(api_url=api_url, api_key=api_key)

def build_dataset(client: rg.Argilla, dataset_name: str, workspace: str, force: bool) -> rg.Dataset:
    """Construit et configure le dataset Argilla."""
    settings = rg.Settings(
        guidelines="Examen des annotations 'gold' d'émotions et de modes d'expression.",
        fields=[
            rg.TextField(name="text", title="Texte de la conversation"),
        ],
        questions=[
            rg.MultiLabelQuestion(
                name="emotions",
                title="Émotions présentes",
                labels=EMOTIONS,
                required=False
            ),
            rg.MultiLabelQuestion(
                name="modes",
                title="Modes d'expression",
                labels=MODES,
                required=False
            )
        ],
        metadata=[
            rg.IntegerMetadataProperty(name="idx", title="Index de ligne"),
            rg.TermsMetadataProperty(name="source", title="Fichier source"),
            rg.TermsMetadataProperty(name="role", title="Role"),
        ]
    )

    try:
        existing = client.datasets(name=dataset_name, workspace=workspace)
        if force:
            print(f"[*] Suppression du dataset '{dataset_name}' existant...")
            existing.delete()
        else:
            print(f"⚠ Le dataset '{dataset_name}' existe déjà. Utilisez --force pour l'écraser.")
            sys.exit(1)
    except Exception:
        pass

    dataset = rg.Dataset(
        name=dataset_name,
        workspace=workspace,
        settings=settings,
        client=client
    )
    dataset.create()
    print(f"✓ Dataset '{dataset_name}' créé avec succès.")
    return dataset

def prepare_records(df: pd.DataFrame, source_filename: str) -> list[rg.Record]:
    """Prépare des objets rg.Record avec les suggestions construites selon les colonnes (0/1)."""
    records = []
    
    # Validation colonnes
    missing_cols = [c for c in (EMOTIONS + MODES) if c not in df.columns]
    if missing_cols:
        print(f"⚠ Les colonnes suivantes manquent dans le fichier XLSX : {missing_cols}")
        print("Poursuite de l'exécution avec les colonnes disponibles...")

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Préparation des entrées"):
        text = str(row.get("TEXT", ""))
        
        # Ignorer les lignes sans texte pertinent
        if not text.strip() or pd.isna(row.get("TEXT")):
            continue
            
        # Extraction des émotions/modes. Si la colonne existe et la valeur == 1
        active_emotions = [em for em in EMOTIONS if em in df.columns and row.get(em) == 1]
        active_modes = [mode for mode in MODES if mode in df.columns and row.get(mode) == 1]
        
        # Metadata implicites, pas mises en évidence dans l'interface principale
        metadata = {
            "idx": int(row.get("idx", row_idx)),
            "source": source_filename,
        }
        
        if "ROLE" in df.columns and not pd.isna(row["ROLE"]):
            metadata["role"] = str(row["ROLE"])[0:100] # troncature sécurité
            
        record = rg.Record(
            fields={"text": text},
            suggestions=[
                rg.Suggestion("emotions", active_emotions),
                rg.Suggestion("modes", active_modes)
            ],
            metadata=metadata
        )
        records.append(record)
        
    return records

def main():
    parser = argparse.ArgumentParser(description="Importer des données structurées Gold vers Argilla.")
    parser.add_argument("--xlsx", required=True, help="Chemin vers le fichier XLSX contenant les annotations.")
    parser.add_argument("--api_url", required=True, help="URL de l'instance Argilla.")
    parser.add_argument("--api_key", required=True, help="Clé API de l'instance Argilla.")
    parser.add_argument("--dataset", required=True, help="Nom du dataset Argilla à créer.")
    parser.add_argument("--workspace", default="argilla", help="Workspace Argilla de destination.")
    parser.add_argument("--force", action="store_true", help="Forcer la création complète s'il existe déjà.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.xlsx):
        print(f"⚠ Le fichier {args.xlsx} n'a pas été trouvé.")
        sys.exit(1)
        
    print(f"[*] Chargement du fichier: {args.xlsx}")
    try:
        df = pd.read_excel(args.xlsx, engine="openpyxl")
    except Exception as e:
        print(f"⚠ Erreur lors du chargement du fichier Excel : {e}")
        sys.exit(1)
        
    if "TEXT" not in df.columns:
        print("⚠ Impossible de trouver la colonne 'TEXT' dans le fichier.")
        sys.exit(1)
        
    client = connect_argilla(args.api_url, args.api_key)
    dataset = build_dataset(client, args.dataset, args.workspace, args.force)
    
    records = prepare_records(df, os.path.basename(args.xlsx))
    
    if len(records) > 0:
        print(f"[*] Publication de {len(records)} entrées vers Argilla en cours...")
        dataset.records.log(records)
        print("✓ Importation terminée avec succès !")
    else:
        print("⚠ Aucun record valide n'a pu être extrait.")

if __name__ == "__main__":
    main()
