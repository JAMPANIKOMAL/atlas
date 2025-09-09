# =============================================================================
# ATLAS v3 - 18S PIPELINE - SCRIPT 1: PREPARE DATA
# =============================================================================
# This script prepares the eukaryotic data from the SILVA database.
#
# WORKFLOW:
# 1.  Checks for a pre-filtered `SILVA_eukaryotes_only.fasta`. If not found,
#     it creates one by filtering the full SILVA database, a one-time operation.
# 2.  Reads from the eukaryote-only file (or a sample for testing).
# 3.  Uses a robust parser to handle complex eukaryotic taxonomy strings.
# 4.  Cleans and filters the data.
# 5.  Engineers k-mer features.
# 6.  Vectorizes, splits, and saves all final 18S-specific artifacts.
# =============================================================================

# --- Imports ---
import pandas as pd
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path
import pickle
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz

# --- Configuration ---
project_root = Path(__file__).parent.parent.parent
RAW_DATA_DIR = project_root / "data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "processed"
MODELS_DIR = project_root / "models"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- SCRIPT BEHAVIOR SWITCH ---
USE_SAMPLE = True
# USE_SAMPLE = False

# --- File Paths ---
FULL_SILVA_PATH = RAW_DATA_DIR / "SILVA_138.1_SSURef_NR99_tax_silva.fasta"
EUKARYOTE_ONLY_PATH = RAW_DATA_DIR / "SILVA_eukaryotes_only.fasta"
SAMPLE_EUKARYOTE_PATH = RAW_DATA_DIR / "SILVA_eukaryotes_sample_10k.fasta"

# --- Parameters ---
KMER_SIZE = 6
TARGET_RANK = 'genus'
MIN_CLASS_MEMBERS = 3
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

# --- Helper Functions ---
DISCARD_RANKS = {
    'cellular organisms', 'Opisthokonta', 'Holozoa', 'Metazoa (Animalia)', 'Eumetazoa',
    'Bilateria', 'Protostomia', 'Deuterostomia', 'Sar', 'Stramenopila', 'Alveolata',
    'Rhizaria', 'Archaeplastida', 'Glaucophyta', 'Chloroplastida', 'Rhodophyceae',
    'Streptophyta', 'Embryophyta', 'Tracheophyta', 'Phragmoplastophyta', 'Excavata',
    'Discoba', 'Metamonada'
}

def parse_eukaryote_taxonomy(taxonomy_str):
    """A robust parser for complex SILVA Eukaryote taxonomy."""
    parsed_ranks = {'kingdom': None, 'phylum': None, 'class': None, 'order': None, 'family': None, 'genus': None, 'species': None}
    discard_lower = {r.lower() for r in DISCARD_RANKS}
    ranks = [rank.strip() for rank in taxonomy_str.split(';') if rank.strip() and rank.strip().lower() not in discard_lower]
    if not ranks: return parsed_ranks
    parsed_ranks['kingdom'] = ranks[0]
    if len(ranks) > 1:
        last_item = ranks[-1]
        if ' ' in last_item:
            parsed_ranks['species'] = last_item
            if len(ranks) > 2: parsed_ranks['genus'] = ranks[-2]
            remaining_ranks = ranks[1:-2]
        else:
            parsed_ranks['genus'] = last_item
            remaining_ranks = ranks[1:-1]
        if len(remaining_ranks) > 0: parsed_ranks['family'] = remaining_ranks[-1]
        if len(remaining_ranks) > 1: parsed_ranks['order'] = remaining_ranks[-2]
        if len(remaining_ranks) > 2: parsed_ranks['class'] = remaining_ranks[-3]
        if len(remaining_ranks) > 3: parsed_ranks['phylum'] = remaining_ranks[-4]
        elif not parsed_ranks['phylum'] and len(ranks) > 1: parsed_ranks['phylum'] = ranks[1]
    return parsed_ranks

def get_kmer_counts(sequence, k):
    """Calculates k-mer counts for a sequence."""
    counts = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if "N" not in kmer.upper(): counts[kmer] += 1
    return dict(counts)

# =============================================================================
# --- Main Script Execution ---
# =============================================================================

if __name__ == "__main__":
    # --- Step 1: Ensure Eukaryote-only file exists ---
    if not EUKARYOTE_ONLY_PATH.exists():
        print(f"Eukaryote-only file not found. Creating it now from {FULL_SILVA_PATH.name}...")
        print("This is a one-time operation and may take a very long time.")
        eukaryote_count = 0
        with open(FULL_SILVA_PATH, "r") as handle_in, open(EUKARYOTE_ONLY_PATH, "w") as handle_out:
            for record in tqdm(SeqIO.parse(handle_in, "fasta"), desc="Filtering for Eukaryotes"):
                if "Eukaryota" in record.description:
                    SeqIO.write(record, handle_out, "fasta")
                    eukaryote_count += 1
        print(f"Found and wrote {eukaryote_count:,} Eukaryote sequences.")
    else:
        print(f"Found existing Eukaryote-only file.")

    # --- Step 2: Select Input File ---
    if USE_SAMPLE:
        input_fasta_path = SAMPLE_EUKARYOTE_PATH
        print(f"--- Running in SAMPLE mode on: {input_fasta_path.name} ---")
    else:
        input_fasta_path = EUKARYOTE_ONLY_PATH
        print(f"--- Running in FULL DATASET mode on: {input_fasta_path.name} ---")

    # --- Step 3: Parse Taxonomy ---
    print("\nStep 3: Parsing taxonomy...")
    parsed_data = []
    with open(input_fasta_path, "r") as handle:
        for record in tqdm(SeqIO.parse(handle, "fasta"), desc="Parsing sequences"):
            accession, taxonomy_str = record.description.split(' ', 1)
            taxonomy_dict = parse_eukaryote_taxonomy(taxonomy_str)
            taxonomy_dict['id'] = record.id
            taxonomy_dict['sequence'] = str(record.seq)
            parsed_data.append(taxonomy_dict)
    df = pd.DataFrame(parsed_data)
    print(f"Created DataFrame with {len(df)} rows.")

    # --- Step 4: Clean and Filter ---
    print("\nStep 4: Cleaning and filtering data...")
    df_cleaned = df.dropna(subset=[TARGET_RANK]).copy()
    class_counts = df_cleaned[TARGET_RANK].value_counts()
    classes_to_keep = class_counts[class_counts >= MIN_CLASS_MEMBERS].index
    df_filtered = df_cleaned[df_cleaned[TARGET_RANK].isin(classes_to_keep)].copy()
    print(f"Final DataFrame has {len(df_filtered)} sequences after cleaning.")

    # --- Step 5: Feature Engineering ---
    print(f"\nStep 5: Engineering {KMER_SIZE}-mer features...")
    df_filtered['kmer_counts'] = list(tqdm((get_kmer_counts(seq, KMER_SIZE) for seq in df_filtered['sequence']), total=len(df_filtered), desc="Calculating k-mers"))

    # --- Step 6: Vectorize ---
    print("\nStep 6: Vectorizing data...")
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(df_filtered['kmer_counts'])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_filtered[TARGET_RANK])
    print(f"Feature matrix shape: {X.shape}")

    # --- Step 7: Split ---
    print("\nStep 7: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"Training set shape: {X_train.shape}")

    # --- Step 8: Save Artifacts ---
    print("\nStep 8: Saving all 18S artifacts to disk...")
    save_npz(PROCESSED_DATA_DIR / "X_train_18s.npz", X_train)
    save_npz(PROCESSED_DATA_DIR / "X_test_18s.npz", X_test)
    np.save(PROCESSED_DATA_DIR / "y_train_18s.npy", y_train)
    np.save(PROCESSED_DATA_DIR / "y_test_18s.npy", y_test)
    with open(MODELS_DIR / "18s_genus_vectorizer.pkl", 'wb') as f: pickle.dump(vectorizer, f)
    with open(MODELS_DIR / "18s_genus_label_encoder.pkl", 'wb') as f: pickle.dump(label_encoder, f)
    print("All artifacts saved successfully.")
    print("\n--- 18S DATA PREPARATION COMPLETE ---")
