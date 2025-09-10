# =============================================================================
# ATLAS - COI PIPELINE - SCRIPT 1: PREPARE DATA
# =============================================================================
# This script prepares the animal (Metazoa) data from the BOLD database.
#
# WORKFLOW:
# 1.  Reads the source BOLD FASTA file (or a sample for testing).
# 2.  Uses a robust, two-stage parser to handle the complex BOLD headers.
# 3.  Cleans, filters, engineers k-mer features, vectorizes, splits, and
#     saves all final COI-specific artifacts.
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
# NOTE: The exact filename for BOLD may change. Update if necessary.
FULL_BOLD_PATH = RAW_DATA_DIR / "BOLD_Public.29-Aug-2025.fasta"
SAMPLE_BOLD_PATH = RAW_DATA_DIR / "BOLD_sample_10k.fasta"

# --- Parameters ---
KMER_SIZE = 8
TARGET_RANK = 'genus'
MIN_CLASS_MEMBERS = 3
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

# --- Helper Functions ---
def parse_bold_taxonomy_v2(description):
    """Parses the complex, pipe-and-comma-separated BOLD FASTA header."""
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
            if len(ranks) > 0: parsed_ranks['kingdom'] = ranks[0]
            if len(ranks) > 1: parsed_ranks['phylum'] = ranks[1]
            if len(ranks) > 2: parsed_ranks['class'] = ranks[2]
            if len(ranks) > 3: parsed_ranks['order'] = ranks[3]
            if len(ranks) > 4: parsed_ranks['family'] = ranks[4]
            if len(ranks) > 5: parsed_ranks['genus'] = ranks[5]
            if len(ranks) > 6: parsed_ranks['species'] = ranks[6]
    except Exception:
        pass
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
    # --- Step 1: Select Input File ---
    if USE_SAMPLE:
        input_fasta_path = SAMPLE_BOLD_PATH
        print(f"--- Running in SAMPLE mode on: {input_fasta_path.name} ---")
    else:
        input_fasta_path = FULL_BOLD_PATH
        print(f"--- Running in FULL DATASET mode on: {input_fasta_path.name} ---")

    # --- Step 2: Parse Taxonomy ---
    print("\nStep 2: Parsing taxonomy...")
    parsed_data = []
    with open(input_fasta_path, "r") as handle:
        for record in tqdm(SeqIO.parse(handle, "fasta"), desc="Parsing sequences"):
            taxonomy_dict = parse_bold_taxonomy_v2(record.description)
            taxonomy_dict['id'] = record.id
            taxonomy_dict['sequence'] = str(record.seq)
            parsed_data.append(taxonomy_dict)
    df = pd.DataFrame(parsed_data)
    print(f"Created DataFrame with {len(df)} rows.")

    # --- Step 3: Clean and Filter ---
    print("\nStep 3: Cleaning and filtering data...")
    df_cleaned = df.dropna(subset=[TARGET_RANK]).copy()
    class_counts = df_cleaned[TARGET_RANK].value_counts()
    classes_to_keep = class_counts[class_counts >= MIN_CLASS_MEMBERS].index
    df_filtered = df_cleaned[df_cleaned[TARGET_RANK].isin(classes_to_keep)].copy()
    print(f"Final DataFrame has {len(df_filtered)} sequences after cleaning.")

    # --- Step 4: Feature Engineering ---
    print(f"\nStep 4: Engineering {KMER_SIZE}-mer features...")
    df_filtered['kmer_counts'] = list(tqdm((get_kmer_counts(seq, KMER_SIZE) for seq in df_filtered['sequence']), total=len(df_filtered), desc="Calculating k-mers"))

    # --- Step 5: Vectorize ---
    print("\nStep 5: Vectorizing data...")
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(df_filtered['kmer_counts'])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_filtered[TARGET_RANK])
    print(f"Feature matrix shape: {X.shape}")

    # --- Step 6: Split ---
    print("\nStep 6: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"Training set shape: {X_train.shape}")

    # --- Step 7: Save Artifacts ---
    print("\nStep 7: Saving all COI artifacts to disk...")
    save_npz(PROCESSED_DATA_DIR / "X_train_coi.npz", X_train)
    save_npz(PROCESSED_DATA_DIR / "X_test_coi.npz", X_test)
    np.save(PROCESSED_DATA_DIR / "y_train_coi.npy", y_train)
    np.save(PROCESSED_DATA_DIR / "y_test_coi.npy", y_test)
    with open(MODELS_DIR / "coi_genus_vectorizer.pkl", 'wb') as f: pickle.dump(vectorizer, f)
    with open(MODELS_DIR / "coi_genus_label_encoder.pkl", 'wb') as f: pickle.dump(label_encoder, f)
    print("All artifacts saved successfully.")
    print("\n--- COI DATA PREPARATION COMPLETE ---")
