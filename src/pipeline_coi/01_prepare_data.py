# =============================================================================
# ATLAS - COI PIPELINE - SCRIPT 1: PREPARE DATA (SCALABILITY FIX)
# =============================================================================
#
# FINAL MODIFICATION:
#   -   Implements a stratified sampling step on the curated 500k dataset
#       to produce a smaller, more manageable dataset (~5k sequences) for
#       GPU-constrained environments.
#   -   Retains the HashingVectorizer to convert sequences to a sparse
#       matrix, but now on a much smaller dataset.
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
SMALL_SAMPLE_PATH = RAW_DATA_DIR / "BOLD_sample_1pct.fasta"

# --- Parameters ---
KMER_SIZE = 8
TARGET_RANK = 'genus'
MIN_CLASS_MEMBERS = 3
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
# We are now targeting a much smaller feature space
HASHING_FEATURES = 2**16  # (65,536 features)
SAMPLE_FRACTION = 0.01  # Use only 1% of the curated dataset

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

def sequence_to_kmer_string(sequence_record, k):
    def kmer_generator(sequence, k):
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if "N" not in kmer.upper():
                yield kmer
    return " ".join(kmer_generator(str(sequence_record.seq), k))

# =============================================================================
# --- Main Script Execution ---
# =============================================================================

if __name__ == "__main__":
    if not INPUT_FASTA_PATH.exists():
        print(f"[ERROR] Input FASTA file not found at: {INPUT_FASTA_PATH}")
        print("Please ensure you have run the curation script `00_curate_bold_dataset.py` with the 500k setting.")
        exit()

    # --- Step 1: Create a smaller, stratified sample from the 500k dataset ---
    print(f"--- Step 1: Creating a {SAMPLE_FRACTION:.0%} sample from the curated dataset... ---")
    if not SMALL_SAMPLE_PATH.exists():
        metadata = []
        with open(INPUT_FASTA_PATH, "r") as handle:
            for record in tqdm(SeqIO.parse(handle, "fasta"), desc="  - Reading records"):
                tax_info = parse_bold_taxonomy_v2(record.description)
                if tax_info['genus']:
                    metadata.append({'id': record.id, 'genus': tax_info['genus']})
        
        df_meta = pd.DataFrame(metadata)
        
        # Remove singletons for proper stratification
        class_counts = df_meta[TARGET_RANK].value_counts()
        classes_to_keep = class_counts[class_counts >= MIN_CLASS_MEMBERS].index
        df_to_sample = df_meta[df_meta[TARGET_RANK].isin(classes_to_keep)]
        
        # Perform stratified sampling
        df_sampled = df_to_sample.groupby(TARGET_RANK, group_keys=False).apply(lambda x: x.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_STATE))
        
        ids_to_keep = set(df_sampled['id'])
        
        print(f"  - Selected {len(ids_to_keep)} sequences for the 1% subset.")
        
        with open(INPUT_FASTA_PATH, "r") as handle_in, open(SMALL_SAMPLE_PATH, "w") as handle_out:
            for record in tqdm(SeqIO.parse(handle_in, "fasta"), desc="  - Writing sequences"):
                if record.id in ids_to_keep:
                    SeqIO.write(record, handle_out, "fasta")
        
        print(f"  - Sample dataset created successfully at: {SMALL_SAMPLE_PATH}")
        
        del df_meta, df_to_sample, df_sampled
        gc.collect()
    else:
        print(f"  - Sample file already exists. Using: {SMALL_SAMPLE_PATH}")


    # --- Step 2 & 3: Vectorize and process the new, smaller dataset ---
    print("\n--- Step 2: Parsing taxonomy and cleaning data... ---")
    all_records = list(tqdm(SeqIO.parse(SMALL_SAMPLE_PATH, "fasta"), desc="  - Reading records into memory"))
    labels = [parse_bold_taxonomy_v2(rec.description)['genus'] for rec in all_records]
    df = pd.DataFrame({'id': [rec.id for rec in all_records], 'genus': labels})
    df_cleaned = df.dropna(subset=[TARGET_RANK]).copy()
    class_counts = df_cleaned[TARGET_RANK].value_counts()
    classes_to_keep = class_counts[class_counts >= MIN_CLASS_MEMBERS].index
    df_filtered = df_cleaned[df_cleaned[TARGET_RANK].isin(classes_to_keep)].copy()
    y = df_filtered[TARGET_RANK].values
    final_ids_to_keep = set(df_filtered['id'])
    num_final_sequences = len(df_filtered)
    print(f"  - Final dataset has {num_final_sequences} sequences after cleaning.")
    del df, df_cleaned, df_filtered, labels, class_counts, classes_to_keep
    gc.collect()
    
    print(f"\n--- Step 3: Vectorizing {num_final_sequences} sequences... ---")
    vectorizer = HashingVectorizer(
        analyzer='char', ngram_range=(KMER_SIZE, KMER_SIZE),
        n_features=HASHING_FEATURES, alternate_sign=False
    )
    kmer_strings = [
        sequence_to_kmer_string(record, KMER_SIZE)
        for record in all_records
        if record.id in final_ids_to_keep
    ]
    X = vectorizer.transform(kmer_strings)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"  - Feature matrix shape: {X.shape}")

    print("\n--- Step 4: Splitting data... ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y_encoded)
    print(f"  - Training set shape: {X_train.shape}")

    print("\n--- Step 5: Saving all COI artifacts to disk... ---")
    save_npz(PROCESSED_DATA_DIR / "X_train_coi.npz", X_train)
    save_npz(PROCESSED_DATA_DIR / "X_test_coi.npz", X_test)
    np.save(PROCESSED_DATA_DIR / "y_train_coi.npy", y_train)
    np.save(PROCESSED_DATA_DIR / "y_test_coi.npy", y_test)
    with open(MODELS_DIR / "coi_genus_label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    print("  - All artifacts saved successfully.")
    print("\n--- COI DATA PREPARATION COMPLETE ---")
