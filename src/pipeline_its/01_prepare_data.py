# =============================================================================
# ATLAS - ITS PIPELINE - SCRIPT 1: PREPARE DATA
# =============================================================================
# This script converts the raw UNITE FASTA file into clean, numerical,
# and model-ready training and testing sets for the Fungi (ITS) classifier.
#
# WORKFLOW:
# 1.  Reads the source UNITE FASTA file (either a sample or the full dataset
#     extracted from the .tgz archive).
# 2.  Uses a custom parser to handle the `k__Fungi;p__...` taxonomy format.
# 3.  Cleans the data by removing records with missing genus labels and
#     genera with too few members (less than 3).
# 4.  Engineers k-mer features from the DNA sequences (k=7).
# 5.  Vectorizes features and labels into numerical matrices.
# 6.  Splits the data into training and testing sets.
# 7.  Saves all final ITS-specific artifacts (.npz, .npy, .pkl) to disk.
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
# Note: tarfile and io are not needed here as we assume the user has created
# the sample file or will provide a fully extracted fasta for the full run.

# --- Configuration ---
# Set the project root path
try:
    project_root = Path(__file__).parent.parent.parent
except NameError:
    # This fallback is for interactive development in notebooks
    project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()


# Define directories
RAW_DATA_DIR = project_root / "data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "processed"
MODELS_DIR = project_root / "models"

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- SCRIPT BEHAVIOR SWITCH ---
# Set to True to run on the small 10k sample for quick tests.
USE_SAMPLE = True
# USE_SAMPLE = False

# --- Data File Paths ---
# For the full run, the user should provide the extracted FASTA file.
FULL_UNITE_PATH = RAW_DATA_DIR / "sh_general_release_dynamic_19.02.2025.fasta"
SAMPLE_UNITE_PATH = RAW_DATA_DIR / "UNITE_sample_10k.fasta"

# --- Feature and Model Parameters ---
KMER_SIZE = 7
TARGET_RANK = 'genus'
MIN_CLASS_MEMBERS = 3
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

# --- Helper Functions ---

def parse_unite_taxonomy(description):
    """
    Parses the UNITE database header format, correctly targeting the LAST
    pipe-separated element for the taxonomy string.
    """
    parsed_ranks = {
        'kingdom': None, 'phylum': None, 'class': None, 'order': None,
        'family': None, 'genus': None, 'species': None
    }
    try:
        taxonomy_str = description.split('|')[-1]
        ranks = taxonomy_str.split(';')
        for rank_str in ranks:
            parts = rank_str.split('__')
            if len(parts) == 2:
                prefix, name = parts
                if not name: continue
                if   prefix == 'k': parsed_ranks['kingdom'] = name
                elif prefix == 'p': parsed_ranks['phylum'] = name
                elif prefix == 'c': parsed_ranks['class'] = name
                elif prefix == 'o': parsed_ranks['order'] = name
                elif prefix == 'f': parsed_ranks['family'] = name
                elif prefix == 'g': parsed_ranks['genus'] = name
                elif prefix == 's': parsed_ranks['species'] = name
    except IndexError:
        pass
    return parsed_ranks

def get_kmer_counts(sequence, k):
    """Calculates the k-mer counts for a given DNA sequence."""
    counts = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if "N" not in kmer.upper():
            counts[kmer] += 1
    return dict(counts)

# =============================================================================
# --- Main Script Execution ---
# =============================================================================

if __name__ == "__main__":

    # --- Step 1: Select Input File ---
    if USE_SAMPLE:
        input_fasta_path = SAMPLE_UNITE_PATH
        print(f"--- Running in SAMPLE mode on: {input_fasta_path.name} ---")
        if not input_fasta_path.exists():
            print(f"[ERROR] Sample file not found at {input_fasta_path}")
            print("Please run the `07_Refining_ITS_Preparation.ipynb` notebook to create it.")
            exit()
    else:
        input_fasta_path = FULL_UNITE_PATH
        print(f"--- Running in FULL DATASET mode on: {input_fasta_path.name} ---")
        if not input_fasta_path.exists():
            print(f"[ERROR] Full UNITE FASTA file not found at {input_fasta_path}")
            print("Please extract the .fasta file from the .tgz archive into the `data/raw` directory.")
            exit()

    # --- Step 2: Parse Taxonomy ---
    print("\nStep 2: Parsing taxonomy...")
    parsed_data = []
    with open(input_fasta_path, "r") as handle:
        for record in tqdm(SeqIO.parse(handle, "fasta"), desc="Parsing sequences"):
            taxonomy_dict = parse_unite_taxonomy(record.description)
            taxonomy_dict['id'] = record.id
            taxonomy_dict['sequence'] = str(record.seq)
            parsed_data.append(taxonomy_dict)
    df = pd.DataFrame(parsed_data)
    print(f"Created DataFrame with {len(df)} rows.")

    # --- Step 3: Clean and Filter DataFrame ---
    print("\nStep 3: Cleaning and filtering data...")
    df_cleaned = df.dropna(subset=[TARGET_RANK]).copy()
    class_counts = df_cleaned[TARGET_RANK].value_counts()
    classes_to_keep = class_counts[class_counts >= MIN_CLASS_MEMBERS].index
    df_filtered = df_cleaned[df_cleaned[TARGET_RANK].isin(classes_to_keep)].copy()
    print(f"Final DataFrame has {len(df_filtered)} sequences after cleaning.")

    # --- Step 4: Feature Engineering (K-mer Counting) ---
    print(f"\nStep 4: Engineering {KMER_SIZE}-mer features...")
    df_filtered['kmer_counts'] = list(tqdm((get_kmer_counts(seq, KMER_SIZE) for seq in df_filtered['sequence']), total=len(df_filtered), desc="Calculating k-mers"))

    # --- Step 5: Vectorize Features and Labels ---
    print("\nStep 5: Vectorizing data...")
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(df_filtered['kmer_counts'])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_filtered[TARGET_RANK])
    print(f"  - Feature matrix shape: {X.shape}")
    print(f"  - Label vector shape:   {y.shape}")

    # --- Step 6: Split Data ---
    print("\nStep 6: Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"  - Training set shape: {X_train.shape}")
    print(f"  - Testing set shape:  {X_test.shape}")

    # --- Step 7: Save Artifacts ---
    print("\nStep 7: Saving all ITS artifacts to disk...")
    save_npz(PROCESSED_DATA_DIR / "X_train_its.npz", X_train)
    save_npz(PROCESSED_DATA_DIR / "X_test_its.npz", X_test)
    np.save(PROCESSED_DATA_DIR / "y_train_its.npy", y_train)
    np.save(PROCESSED_DATA_DIR / "y_test_its.npy", y_test)
    with open(MODELS_DIR / "its_genus_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(MODELS_DIR / "its_genus_label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    print("  - All artifacts saved successfully.")
    
    # --- Final ASCII Art Confirmation ---
    print("\n" + "="*50)
    print("    ITS DATA PREPARATION SCRIPT COMPLETE")
    print("="*50 + "\n")
