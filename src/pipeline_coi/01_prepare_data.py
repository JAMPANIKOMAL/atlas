# =============================================================================
# ATLAS - COI PIPELINE - SCRIPT 1: PREPARE DATA (FINAL BATCH PROCESSING)
# =============================================================================
# This script prepares the animal (Metazoa) data from the BOLD database.
#
# FINAL VERSION - SCALABILITY FIX:
#   This version implements a true "out-of-core" batch processing workflow
#   to handle the massive, curated BOLD dataset without crashing due to
#   memory errors. It processes the data in smaller chunks and then stacks
#   the resulting sparse matrices.
#
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
# --- BATCH PROCESSING MODIFICATION: Import vstack ---
from scipy.sparse import save_npz, vstack
import gc

# --- Configuration ---
project_root = Path(__file__).parent.parent.parent
RAW_DATA_DIR = project_root / "data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "processed"
MODELS_DIR = project_root / "models"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- File Paths ---
INPUT_FASTA_PATH = RAW_DATA_DIR / "BOLD_curated_subset.fasta"

# --- Parameters ---
KMER_SIZE = 8
TARGET_RANK = 'genus'
MIN_CLASS_MEMBERS = 3
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
HASHING_FEATURES = 2**20
# --- BATCH PROCESSING MODIFICATION: Define batch size ---
BATCH_SIZE = 100000 # Process 100,000 sequences at a time

# --- Helper Functions ---
def parse_bold_taxonomy_v2(description):
    parsed_ranks = {'genus': None}
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
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if "N" not in kmer.upper():
            yield kmer

def sequence_to_kmer_string(sequence_record, k):
    return " ".join(kmer_generator(str(sequence_record.seq), k))

# --- BATCH PROCESSING MODIFICATION: New batch generator function ---
def process_in_batches(fasta_path, ids_to_process, batch_size):
    """
    Generator function that reads a FASTA file and yields batches of
    k-mer strings for the specified IDs.
    """
    batch = []
    record_iterator = (rec for rec in SeqIO.parse(fasta_path, "fasta") if rec.id in ids_to_process)
    
    for record in record_iterator:
        batch.append(sequence_to_kmer_string(record, KMER_SIZE))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch: # Yield the last, possibly smaller, batch
        yield batch

# =============================================================================
# --- Main Script Execution ---
# =============================================================================

if __name__ == "__main__":
    print(f"--- Running in FULL BATCH mode on: {INPUT_FASTA_PATH.name} ---")

    if not INPUT_FASTA_PATH.exists():
        print(f"[ERROR] Input FASTA file not found at: {INPUT_FASTA_PATH}")
        print("Please ensure you have run the curation script `00_curate_bold_dataset.py` first.")
        exit()

    # --- Step 1 & 2: Parse Taxonomy and Clean Data ---
    print("\n--- Step 1 & 2: Parsing taxonomy and cleaning data... ---")
    labels = []
    record_generator = (record for record in SeqIO.parse(INPUT_FASTA_PATH, "fasta"))
    
    for record in tqdm(record_generator, desc="  - Reading records"):
        genus = parse_bold_taxonomy_v2(record.description)['genus']
        labels.append(genus)

    df = pd.DataFrame({'id': [rec.id for rec in SeqIO.parse(INPUT_FASTA_PATH, "fasta")], 'genus': labels})
    df_cleaned = df.dropna(subset=[TARGET_RANK]).copy()
    class_counts = df_cleaned[TARGET_RANK].value_counts()
    classes_to_keep = class_counts[class_counts >= MIN_CLASS_MEMBERS].index
    df_filtered = df_cleaned[df_cleaned[TARGET_RANK].isin(classes_to_keep)].copy()
    
    final_ids_to_keep = set(df_filtered['id'])
    y = df_filtered[TARGET_RANK].values
    num_final_sequences = len(df_filtered)
    print(f"  - Final dataset has {num_final_sequences} sequences after cleaning.")
    del df, df_cleaned, df_filtered, labels, class_counts, classes_to_keep
    gc.collect()

    # --- Step 3 & 4: Vectorize in Batches ---
    print(f"\n--- Step 3 & 4: Vectorizing {num_final_sequences} sequences in batches of {BATCH_SIZE}... ---")
    
    vectorizer = HashingVectorizer(
        analyzer='char', ngram_range=(KMER_SIZE, KMER_SIZE),
        n_features=HASHING_FEATURES, alternate_sign=False
    )

    # --- BATCH PROCESSING MODIFICATION: Process in a loop ---
    matrix_chunks = []
    batch_generator = process_in_batches(INPUT_FASTA_PATH, final_ids_to_keep, BATCH_SIZE)
    num_batches = (num_final_sequences + BATCH_SIZE - 1) // BATCH_SIZE

    for batch in tqdm(batch_generator, total=num_batches, desc="  - Vectorizing batches"):
        matrix_chunks.append(vectorizer.transform(batch))

    # --- BATCH PROCESSING MODIFICATION: Stack the chunks ---
    print("\n  - Stacking vectorized chunks into final sparse matrix...")
    X = vstack(matrix_chunks)
    
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
    with open(MODELS_DIR / "coi_genus_label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    print("  - All artifacts saved successfully.")
    print("\n--- COI DATA PREPARATION COMPLETE ---")

