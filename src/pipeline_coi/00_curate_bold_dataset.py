# =============================================================================
# ATLAS - COI PIPELINE - SCRIPT 0: CURATE BOLD DATASET
# =============================================================================
#
# OBJECTIVE:
#   To address the extreme size of the full BOLD database by creating a
#   smaller, taxonomically balanced, and high-quality subset for training.
#
# MODIFICATION:
#   The sampling fraction has been adjusted to target approximately 500,000
#   sequences to ensure a practical processing time on standard hardware.
#
# =============================================================================

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---
project_root = Path(__file__).parent.parent.parent
RAW_DATA_DIR = project_root / "data" / "raw"

FULL_BOLD_PATH = RAW_DATA_DIR / "BOLD_Public.29-Aug-2025.fasta"
# --- MODIFICATION: Updated output filename for the 500k subset ---
CURATED_SUBSET_PATH = RAW_DATA_DIR / "BOLD_curated_subset_500k.fasta"

# --- Parameters ---
# --- MODIFICATION: Changed fraction to target ~500k sequences (1/40 ~ 2.5%) ---
SAMPLING_FRACTION = 1/40
STRATIFY_COLUMN = 'class'
MIN_CLASS_MEMBERS = 10

# --- Helper Function ---
def parse_bold_taxonomy_v2(description):
    parsed_ranks = {'kingdom': None, 'phylum': None, 'class': None, 'order': None, 'family': None, 'genus': None, 'species': None}
    try:
        parts = description.split('|')
        taxonomy_str = ""
        for part in parts:
            if ',' in part and 'Animalia' in part:
                taxonomy_str = part
                break
        if taxonomy_str:
            ranks = taxonomy_str.split(',')
            if len(ranks) > 2: parsed_ranks['class'] = ranks[2]
            if len(ranks) > 5: parsed_ranks['genus'] = ranks[5]
    except Exception:
        pass
    return parsed_ranks

if __name__ == "__main__":
    print("--- Starting BOLD Dataset Curation (Targeting ~500k sequences) ---")

    if not FULL_BOLD_PATH.exists():
        print(f"[ERROR] Full BOLD FASTA file not found at: {FULL_BOLD_PATH}")
        exit()

    # --- Step 1: First Pass - Extract Metadata ---
    print("\n--- Step 1: Parsing metadata from full BOLD dataset... ---")
    metadata = []
    with open(FULL_BOLD_PATH, "r") as handle:
        for record in tqdm(SeqIO.parse(handle, "fasta"), desc="  - Reading records"):
            tax_info = parse_bold_taxonomy_v2(record.description)
            if tax_info[STRATIFY_COLUMN] and tax_info['genus']:
                metadata.append({'id': record.id, STRATIFY_COLUMN: tax_info[STRATIFY_COLUMN]})
    
    df_meta = pd.DataFrame(metadata)
    print(f"  - Found {len(df_meta)} records with valid metadata.")

    # --- Step 2: Stratified Sampling ---
    print(f"\n--- Step 2: Performing stratified sampling to select {SAMPLING_FRACTION:.1%} of the data... ---")
    
    class_counts = df_meta[STRATIFY_COLUMN].value_counts()
    classes_to_sample = class_counts[class_counts >= MIN_CLASS_MEMBERS].index
    df_to_sample = df_meta[df_meta[STRATIFY_COLUMN].isin(classes_to_sample)]

    df_sampled = df_to_sample.groupby(STRATIFY_COLUMN, group_keys=False).apply(lambda x: x.sample(frac=SAMPLING_FRACTION, random_state=42))
    
    ids_to_keep = set(df_sampled['id'])
    print(f"  - Selected {len(ids_to_keep)} sequences for the curated subset.")

    # --- Step 3: Second Pass - Create Curated FASTA File ---
    print("\n--- Step 3: Writing the new curated FASTA file... ---")
    
    with open(FULL_BOLD_PATH, "r") as handle_in, open(CURATED_SUBSET_PATH, "w") as handle_out:
        for record in tqdm(SeqIO.parse(handle_in, "fasta"), desc="  - Writing sequences"):
            if record.id in ids_to_keep:
                SeqIO.write(record, handle_out, "fasta")

    print(f"\n[SUCCESS] Curated dataset created successfully.")
    print(f"  - Location: {CURATED_SUBSET_PATH}")
    print("  - You can now run the main `01_prepare_data.py` script.")

