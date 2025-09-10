# =============================================================================
# ATLAS - COI PIPELINE - SCRIPT 1: PREPARE DATA (FINAL VERSION)
# =============================================================================
#
# FINAL MODIFICATION:
#   -   Reduced HASHING_FEATURES from 2**20 to 2**18 to prevent GPU VRAM
#       exhaustion during model training. This is the final optimization
#       to ensure the pipeline can run on standard hardware.
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
INPUT_FASTA_PATH = RAW_DATA_DIR / "BOLD_curated_subset_500k.fasta"

# --- Parameters ---
KMER_SIZE = 8
TARGET_RANK = 'genus'
MIN_CLASS_MEMBERS = 3
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
# --- FINAL FIX: Reduce feature space to prevent GPU OOM errors ---
HASHING_FEATURES = 2**18 # (262,144 features)
BATCH_SIZE = 50000

# --- Helper Functions (Same as before) ---
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

def process_in_batches(fasta_path, ids_to_process, batch_size):
    batch = []
    record_iterator = (rec for rec in SeqIO.parse(fasta_path, "fasta") if rec.id in ids_to_process)
    for record in record_iterator:
        batch.append(sequence_to_kmer_string(record, KMER_SIZE))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# =============================================================================
# --- Main Script Execution ---
# =============================================================================

if __name__ == "__main__":
    print(f"--- Running in FULL BATCH mode on: {INPUT_FASTA_PATH.name} ---")

    if not INPUT_FASTA_PATH.exists():
        print(f"[ERROR] Input FASTA file not found at: {INPUT_FASTA_PATH}")
        print("Please ensure you have run the curation script `00_curate_bold_dataset.py` with the 500k setting.")
        exit()

    print("\n--- Step 1 & 2: Parsing taxonomy and cleaning data... ---")
    all_records = list(tqdm(SeqIO.parse(INPUT_FASTA_PATH, "fasta"), desc="  - Reading records into memory"))
    labels = [parse_bold_taxonomy_v2(rec.description)['genus'] for rec in all_records]
    all_ids = [rec.id for rec in all_records]
    df = pd.DataFrame({'id': all_ids, 'genus': labels})
    df_cleaned = df.dropna(subset=[TARGET_RANK]).copy()
    class_counts = df_cleaned[TARGET_RANK].value_counts()
    classes_to_keep = class_counts[class_counts >= MIN_CLASS_MEMBERS].index
    df_filtered = df_cleaned[df_cleaned[TARGET_RANK].isin(classes_to_keep)].copy()
    final_ids_to_keep = set(df_filtered['id'])
    y = df_filtered[TARGET_RANK].values
    num_final_sequences = len(df_filtered)
    print(f"  - Final dataset has {num_final_sequences} sequences after cleaning.")
    del df, df_cleaned, df_filtered, labels, class_counts, classes_to_keep, all_ids
    gc.collect()

    print(f"\n--- Step 3 & 4: Vectorizing {num_final_sequences} sequences in batches of {BATCH_SIZE}... ---")
    vectorizer = HashingVectorizer(
        analyzer='char', ngram_range=(KMER_SIZE, KMER_SIZE),
        n_features=HASHING_FEATURES, alternate_sign=False
    )
    kmer_string_generator = (
        sequence_to_kmer_string(record, KMER_SIZE)
        for record in all_records
        if record.id in final_ids_to_keep
    )
    matrix_chunks = []
    batch_generator = process_in_batches(INPUT_FASTA_PATH, final_ids_to_keep, BATCH_SIZE)
    num_batches = (num_final_sequences + BATCH_SIZE - 1) // BATCH_SIZE
    for batch in tqdm(batch_generator, total=num_batches, desc="  - Vectorizing batches"):
        matrix_chunks.append(vectorizer.transform(batch))
    print("\n  - Stacking vectorized chunks into final sparse matrix...")
    X = vstack(matrix_chunks)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"  - Feature matrix shape: {X.shape}")

    print("\n--- Step 5: Splitting data... ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y_encoded)
    print(f"  - Training set shape: {X_train.shape}")

    print("\n--- Step 6: Saving all COI artifacts to disk... ---")
    save_npz(PROCESSED_DATA_DIR / "X_train_coi.npz", X_train)
    save_npz(PROCESSED_DATA_DIR / "X_test_coi.npz", X_test)
    np.save(PROCESSED_DATA_DIR / "y_train_coi.npy", y_train)
    np.save(PROCESSED_DATA_DIR / "y_test_coi.npy", y_test)
    with open(MODELS_DIR / "coi_genus_label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    print("  - All artifacts saved successfully.")
    print("\n--- COI DATA PREPARATION COMPLETE ---")

