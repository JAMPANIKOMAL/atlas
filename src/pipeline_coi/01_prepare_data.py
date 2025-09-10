# =============================================================================
# ATLAS - COI PIPELINE - SCRIPT 1: PREPARE DATA (HIGH-PERFORMANCE VERSION)
# =============================================================================
# This script prepares the animal (Metazoa) data from the BOLD database.
#
# NOTE: This version is designed to run on the `BOLD_curated_subset.fasta`
#       file, which should be generated first by the `00_curate_bold_dataset.py`
#       script.
#
# WORKFLOW:
#   1.  Reads the curated BOLD FASTA subset.
#   2.  Uses a memory-efficient generator to process sequences one by one,
#       avoiding high RAM usage.
#   3.  Cleans and filters the data.
#   4.  Uses HashingVectorizer to convert k-mers directly to a sparse
#       matrix without storing a vocabulary in memory.
#   5.  Splits and saves all final COI-specific artifacts.
# =============================================================================

# --- Imports ---
import pandas as pd
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import save_npz
import gc

# --- Configuration ---
project_root = Path(__file__).parent.parent.parent
RAW_DATA_DIR = project_root / "data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "processed"
MODELS_DIR = project_root / "models"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- SCRIPT BEHAVIOR SWITCH ---
# This script is intended to run on the full curated subset.
# The USE_SAMPLE flag can be set to True to test on the smaller 10k sample.
USE_SAMPLE = False
# USE_SAMPLE = True

# --- File Paths ---
# --- MODIFICATION: This script now points to the new curated subset file ---
FULL_BOLD_PATH = RAW_DATA_DIR / "BOLD_curated_subset.fasta"
SAMPLE_BOLD_PATH = RAW_DATA_DIR / "BOLD_sample_10k.fasta" # Kept for legacy/testing

# --- Parameters ---
KMER_SIZE = 8
TARGET_RANK = 'genus'
MIN_CLASS_MEMBERS = 3
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
# Use a power of 2 for the number of features for optimal hashing
HASHING_FEATURES = 2**20

# --- Helper Functions ---
def parse_bold_taxonomy_v2(description):
    """A memory-efficient parser for BOLD headers."""
    parsed_ranks = {'genus': None} # Only extract what we need
    try:
        parts = description.split('|')
        taxonomy_str = ""
        for part in parts:
            if ',' in part and 'Animalia' in part:
                taxonomy_str = part
                break
        if taxonomy_str:
            ranks = taxonomy_str.split(',')
            if len(ranks) > 5: parsed_ranks['genus'] = ranks[5]
    except Exception:
        pass
    return parsed_ranks

def kmer_generator(sequence, k):
    """Yields k-mers from a sequence string."""
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if "N" not in kmer.upper():
            yield kmer

def sequence_to_kmer_string(sequence_record, k):
    """Converts a Biopython sequence record into a space-separated k-mer string."""
    return " ".join(kmer_generator(str(sequence_record.seq), k))

# =============================================================================
# --- Main Script Execution ---
# =============================================================================

if __name__ == "__main__":
    if USE_SAMPLE:
        input_fasta_path = SAMPLE_BOLD_PATH
        print(f"--- Running in SAMPLE mode on: {input_fasta_path.name} ---")
    else:
        input_fasta_path = FULL_BOLD_PATH
        print(f"--- Running in FULL DATASET mode on: {input_fasta_path.name} ---")

    if not input_fasta_path.exists():
        print(f"[ERROR] Input FASTA file not found at: {input_fasta_path}")
        print("Please ensure you have run the curation script `00_curate_bold_dataset.py` first.")
        exit()

    # --- Step 1 & 2: Parse Taxonomy and Clean Data in a Memory-Efficient Way ---
    print("\n--- Step 1 & 2: Parsing taxonomy and cleaning data... ---")
    labels = []
    # Create a generator expression for memory efficiency
    record_generator = (record for record in SeqIO.parse(input_fasta_path, "fasta"))
    
    for record in tqdm(record_generator, desc="  - Reading records"):
        genus = parse_bold_taxonomy_v2(record.description)['genus']
        labels.append(genus)

    df = pd.DataFrame({'id': [rec.id for rec in SeqIO.parse(input_fasta_path, "fasta")], 'genus': labels})
    
    # Perform the cleaning on the DataFrame
    df_cleaned = df.dropna(subset=[TARGET_RANK]).copy()
    class_counts = df_cleaned[TARGET_RANK].value_counts()
    classes_to_keep = class_counts[class_counts >= MIN_CLASS_MEMBERS].index
    df_filtered = df_cleaned[df_cleaned[TARGET_RANK].isin(classes_to_keep)].copy()
    
    # Get the final set of IDs to keep for vectorization
    final_ids_to_keep = set(df_filtered['id'])
    y = df_filtered[TARGET_RANK].values
    print(f"  - Final dataset has {len(df_filtered)} sequences after cleaning.")
    del df, df_cleaned, df_filtered, labels, class_counts, classes_to_keep
    gc.collect()

    # --- Step 3 & 4: Feature Engineering and Vectorization with HashingVectorizer ---
    print(f"\n--- Step 3 & 4: Engineering {KMER_SIZE}-mer features and vectorizing... ---")
    
    # Define the HashingVectorizer
    vectorizer = HashingVectorizer(
        analyzer='char',
        ngram_range=(KMER_SIZE, KMER_SIZE),
        n_features=HASHING_FEATURES,
        alternate_sign=False # Important for compatibility with some downstream models
    )

    # Create a new generator to read the FASTA file again for vectorization
    kmer_string_generator = (
        sequence_to_kmer_string(record, KMER_SIZE)
        for record in SeqIO.parse(input_fasta_path, "fasta")
        if record.id in final_ids_to_keep
    )
    
    # Fit and transform the data in a streaming fashion
    X = vectorizer.fit_transform(tqdm(kmer_string_generator, desc="  - Vectorizing", total=len(final_ids_to_keep)))
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"  - Feature matrix shape: {X.shape}")

    # --- Step 5: Split Data ---
    print("\n--- Step 5: Splitting data... ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y_encoded)
    print(f"  - Training set shape: {X_train.shape}")

    # --- Step 6: Save Artifacts ---
    print("\n--- Step 6: Saving all COI artifacts to disk... ---")
    save_npz(PROCESSED_DATA_DIR / "X_train_coi.npz", X_train)
    save_npz(PROCESSED_DATA_DIR / "X_test_coi.npz", X_test)
    np.save(PROCESSED_DATA_DIR / "y_train_coi.npy", y_train)
    np.save(PROCESSED_DATA_DIR / "y_test_coi.npy", y_test)
    # Note: We do not save the HashingVectorizer as it is stateless
    with open(MODELS_DIR / "coi_genus_label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    print("  - All artifacts saved successfully.")
    print("\n--- COI DATA PREPARATION COMPLETE ---")

