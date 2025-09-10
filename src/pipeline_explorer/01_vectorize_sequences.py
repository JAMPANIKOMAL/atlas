# =============================================================================
# ATLAS - EXPLORER PIPELINE - SCRIPT 1: VECTORIZE SEQUENCES
# =============================================================================
# This script takes a FASTA file of unclassified sequences, trains a Doc2Vec
# model on their k-mer composition, and saves the resulting sequence vectors.
#
# WORKFLOW:
# 1.  Loads an input FASTA file (e.g., sequences unclassified by the Filter).
# 2.  Converts each sequence into a document of its constituent k-mers.
# 3.  Trains a Gensim Doc2Vec model on this corpus.
# 4.  Saves the trained Doc2Vec model for potential future use.
# 5.  Extracts, normalizes, and saves the final numerical vector for each
#     sequence to a .npy file, ready for the clustering script.
# =============================================================================

# --- Imports ---
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize
import argparse

# --- Configuration ---
try:
    project_root = Path(__file__).parent.parent.parent
except NameError:
    project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()

RAW_DATA_DIR = project_root / "data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "processed"
MODELS_DIR = project_root / "models"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Doc2Vec Parameters ---
KMER_SIZE = 6
VECTOR_SIZE = 100

# --- Helper Functions ---
def sequence_to_kmers(sequence_str, k):
    """Converts a DNA sequence string into a list of its k-mers."""
    return [sequence_str[i:i+k] for i in range(len(sequence_str) - k + 1)]

# =============================================================================
# --- Main Script Execution ---
# =============================================================================
if __name__ == "__main__":
    # --- Command Line Argument Parsing ---
    # This allows us to specify the input file when we run the script
    parser = argparse.ArgumentParser(description="Vectorize DNA sequences using Doc2Vec.")
    parser.add_argument(
        '--input_fasta',
        type=Path,
        default=RAW_DATA_DIR / "unclassified_sample_for_explorer.fasta",
        help="Path to the input FASTA file of unclassified sequences."
    )
    args = parser.parse_args()
    INPUT_FASTA = args.input_fasta

    if not INPUT_FASTA.exists():
        print(f"[ERROR] Input file not found at: {INPUT_FASTA}")
        exit()

    # --- 1. Load Sequences and Prepare Corpus ---
    print(f"--- Step 1: Loading sequences from {INPUT_FASTA.name} ---")
    sequences = list(SeqIO.parse(INPUT_FASTA, "fasta"))
    print(f"  - Loaded {len(sequences)} sequences.")

    print("\n--- Step 2: Preparing the Doc2Vec Corpus ---")
    corpus = [
        TaggedDocument(
            words=sequence_to_kmers(str(seq.seq), KMER_SIZE),
            tags=[seq.id]
        )
        for seq in tqdm(sequences, desc="  - Processing sequences")
    ]
    print(f"  - Corpus prepared with {len(corpus)} documents.")

    # --- 2. Define and Train the Doc2Vec Model ---
    print("\n--- Step 3: Training the Doc2Vec Model ---")
    doc2vec_model = Doc2Vec(
        vector_size=VECTOR_SIZE, dm=1, min_count=3, window=8, epochs=40, workers=4
    )
    doc2vec_model.build_vocab(corpus)
    print(f"  - Vocabulary built with {len(doc2vec_model.wv.key_to_index)} unique k-mers.")
    print("  - Starting training (this may take a minute)...")
    doc2vec_model.train(corpus, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    print("  - Training complete.")

    # --- 3. Save the Model ---
    DOC2VEC_MODEL_PATH = MODELS_DIR / "explorer_doc2vec.model"
    print(f"\n--- Step 4: Saving the Model to {DOC2VEC_MODEL_PATH} ---")
    doc2vec_model.save(str(DOC2VEC_MODEL_PATH))
    print("  - Model saved successfully.")

    # --- 4. Extract, Normalize, and Save Vectors ---
    print("\n--- Step 5: Extracting, Normalizing, and Saving Vectors ---")
    sequence_vectors = np.array([doc2vec_model.dv[seq.id] for seq in sequences])
    sequence_vectors = normalize(sequence_vectors)
    
    VECTORS_OUTPUT_PATH = PROCESSED_DATA_DIR / "explorer_sequence_vectors.npy"
    IDS_OUTPUT_PATH = PROCESSED_DATA_DIR / "explorer_sequence_ids.npy"
    
    np.save(VECTORS_OUTPUT_PATH, sequence_vectors)
    # Also save the IDs in the correct order so we can match them up later
    sequence_ids = np.array([seq.id for seq in sequences])
    np.save(IDS_OUTPUT_PATH, sequence_ids)
    
    print(f"  - Final vector matrix of shape {sequence_vectors.shape} saved successfully.")
    print(f"  - Corresponding sequence IDs saved successfully.")

    # --- Final ASCII Art Confirmation ---
    print("\n" + "="*50)
    print("    EXPLORER VECTORIZATION SCRIPT COMPLETE")
    print("="*50 + "\n")
