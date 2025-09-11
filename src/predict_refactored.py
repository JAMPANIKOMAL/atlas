# =============================================================================
# ATLAS - SELF-CONTAINED PREDICTION SCRIPT
# =============================================================================
# This script has been refactored to combine the logic of the main predictor
# and the entire 'Explorer' pipeline into a single, self-contained file.
#
# Changes from the original:
#   - Subprocess calls to the explorer scripts have been replaced with direct
#     function calls to new, internal functions.
#   - Temporary file creation for passing data between explorer steps has
#     been removed. All data is now passed in-memory.
#   - All necessary imports for all parts of the pipeline are now at the top.
#
# This file serves as a single, runnable unit, which is a critical first
# step towards creating a standalone, installable application.
# =============================================================================

# --- Core Imports ---
import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path
import pickle
import sys
from collections import Counter
from tqdm import tqdm
import uuid
import tensorflow as tf
import gc

# --- Imports for the now-internal Explorer pipeline ---
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import hdbscan
from sklearn.preprocessing import normalize
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
import io

# --- Configuration ---
# All file paths are now defined relative to a central models/data directory.
project_root = Path(__file__).parent.parent
MODELS_DIR = project_root / "models"
REPORTS_DIR = project_root / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Explorer Pipeline Parameters (moved here from original scripts) ---
EXPLORER_KMER_SIZE = 6
EXPLORER_VECTOR_SIZE = 100

# --- Helper Function for K-mer Counting ---
def get_kmer_counts(sequence, k):
    """Calculates k-mer counts for a single DNA sequence."""
    counts = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if "N" not in kmer.upper():
            counts[kmer] += 1
    return dict(counts)

# --- Classifier Class ---
class TaxonClassifier:
    """A wrapper class to hold a trained model and its associated artifacts."""
    def __init__(self, marker_name, kmer_size):
        self.name = marker_name
        self.kmer_size = kmer_size
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.is_loaded = self._load_artifacts()

    def _load_artifacts(self):
        """Loads the model, vectorizer, and label encoder from disk."""
        try:
            model_path = MODELS_DIR / f"{self.name}_genus_classifier.keras"
            vectorizer_path = MODELS_DIR / f"{self.name}_genus_vectorizer.pkl"
            encoder_path = MODELS_DIR / f"{self.name}_genus_label_encoder.pkl"
            
            if not all(p.exists() for p in [model_path, vectorizer_path, encoder_path]):
                print(f"Warning: Missing one or more artifacts for {self.name}. Skipping this classifier.")
                return False
            
            # Use 'compile=False' for faster loading on non-training systems
            self.model = tf.keras.models.load_model(model_path, compile=False)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading {self.name} model: {e}")
            return False

    def predict(self, sequence, confidence_threshold=0.8):
        """Predicts the taxon for a single sequence."""
        if not self.is_loaded:
            return None, 0.0
            
        kmer_counts = get_kmer_counts(sequence, self.kmer_size)
        
        if not kmer_counts:
            return None, 0.0

        vectorized_sequence = self.vectorizer.transform([kmer_counts])
        
        prediction_probabilities = self.model.predict(vectorized_sequence, verbose=0)[0]
        
        top_prob = np.max(prediction_probabilities)
        top_class_index = np.argmax(prediction_probabilities)
        
        if top_prob >= confidence_threshold:
            predicted_label = self.label_encoder.inverse_transform([top_class_index])[0]
            return predicted_label, top_prob
        
        return None, top_prob

# --- GPU Check Function ---
def check_gpu_status():
    """Checks and returns the GPU availability status."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return f"GPU is available and configured: {gpus[0].name}"
    else:
        return "GPU not found. Running on CPU."

# =============================================================================
# --- Explorer Pipeline Logic (Moved from separate scripts) ---
# =============================================================================

def explorer_step_1_vectorize(sequences):
    """
    (Refactored from 01_vectorize_sequences.py)
    Vectorizes unclassified sequences using Doc2Vec.
    """
    if not sequences:
        return None, None
        
    doc2vec_model_path = MODELS_DIR / "explorer_doc2vec.model"
    
    if not doc2vec_model_path.exists():
        print("  - [ERROR] Doc2Vec model not found. Cannot run Explorer pipeline.")
        return None, None

    print("  - Step 1.1: Loading Doc2Vec model...")
    try:
        doc2vec_model = Doc2Vec.load(str(doc2vec_model_path))
    except Exception as e:
        print(f"  - [ERROR] Failed to load Doc2Vec model: {e}")
        return None, None
    
    print("  - Step 1.2: Extracting vectors from sequences...")
    # The original script trained a new model every time. This is more efficient.
    sequence_vectors = np.array([
        doc2vec_model.infer_vector(
            get_kmer_counts(str(seq.seq), EXPLORER_KMER_SIZE),
            epochs=20  # Use a fixed number of inference epochs
        )
        for seq in tqdm(sequences, desc="    - Inferring vectors")
    ])
    
    # It's good practice to normalize the vectors for clustering
    sequence_vectors = normalize(sequence_vectors)
    
    sequence_ids = np.array([seq.id for seq in sequences])
    
    return sequence_vectors, sequence_ids

def explorer_step_2_cluster(sequence_vectors, sequence_ids):
    """
    (Refactored from 02_cluster_sequences.py)
    Clusters sequence vectors using HDBSCAN.
    """
    if sequence_vectors is None or sequence_ids is None or len(sequence_vectors) < 5:
        return None
        
    print("  - Step 2: Performing HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(sequence_vectors)
    
    # Create an in-memory DataFrame with the results
    df_results = pd.DataFrame({
        'sequence_id': sequence_ids,
        'cluster_label': cluster_labels
    })
    
    return df_results

def explorer_step_3_interpret(df_clusters, all_sequences, sequence_vectors):
    """
    (Refactored from 03_interpret_clusters.py)
    Interprets clusters by finding representatives and running BLAST.
    """
    if df_clusters is None or not all_sequences or sequence_vectors is None:
        return "No significant clusters were discovered in the input data."

    print("  - Step 3: Interpreting clusters and running BLAST...")
    
    # Create a dictionary for fast sequence lookups
    sequences_dict = {rec.id: str(rec.seq) for rec in all_sequences}
    id_to_vector_index = {seq_id: i for i, seq_id in enumerate(df_clusters['sequence_id'])}
    
    report_lines = []
    unique_cluster_ids = sorted(df_clusters['cluster_label'].unique())
    if -1 in unique_cluster_ids:
        unique_cluster_ids.remove(-1) # Ignore the noise points

    for cluster_id in tqdm(unique_cluster_ids, desc="    - Analyzing clusters"):
        cluster_df = df_clusters[df_clusters['cluster_label'] == cluster_id]
        member_ids = cluster_df['sequence_id'].tolist()
        
        member_indices = [id_to_vector_index[seq_id] for seq_id in member_ids]
        cluster_vectors = sequence_vectors[member_indices]
        
        centroid = np.mean(cluster_vectors, axis=0)
        distances = [np.linalg.norm(vec - centroid) for vec in cluster_vectors]
        rep_index_in_cluster = np.argmin(distances)
        representative_id = member_ids[rep_index_in_cluster]
        representative_sequence = sequences_dict[representative_id]
        
        try:
            # We will perform the BLAST search here directly
            result_handle = NCBIWWW.qblast("blastn", "nt", representative_sequence)
            blast_record = NCBIXML.read(result_handle)
            
            top_hit_title = "No significant similarity found."
            if blast_record.alignments:
                top_hit_title = blast_record.alignments[0].title
                
        except Exception as e:
            top_hit_title = f"BLAST query failed: {e}"

        report_lines.append(f"Cluster ID: {cluster_id}")
        report_lines.append(f"  - Size: {len(member_ids)} sequences")
        report_lines.append(f"  - Representative Sequence ID: {representative_id}")
        report_lines.append(f"  - BLAST Hypothesis: {top_hit_title}\n")
    
    if report_lines:
        return "\n".join(report_lines)
    else:
        return "No significant clusters were discovered in the input data."


# =============================================================================
# --- Main Analysis Function ---
# =============================================================================
def run_analysis(input_fasta_path):
    """
    Main analysis pipeline, now fully self-contained.
    
    Args:
        input_fasta_path (str): Path to the input FASTA file.
        
    Returns:
        dict: A dictionary containing the analysis results.
    """
    # --- 1. Check GPU Status ---
    gpu_status = check_gpu_status()
    print(f"\n--- GPU Status: {gpu_status} ---")

    # --- 2. Load All Filter Models ---
    print("--- Step 1: Loading All 'Filter' AI Models ---")
    potential_classifiers = [
        TaxonClassifier("16s", 6),
        TaxonClassifier("18s", 6),
        TaxonClassifier("coi", 8),
        TaxonClassifier("its", 7)
    ]
    
    classifiers = [clf for clf in potential_classifiers if clf.is_loaded]
    
    if not classifiers:
        return {"error": "No trained models found. Please ensure models are available."}
    
    print(f"  - Successfully loaded {len(classifiers)} models: {[clf.name for clf in classifiers]}")
    if len(classifiers) < len(potential_classifiers):
        missing = [clf.name for clf in potential_classifiers if not clf.is_loaded]
        print(f"  - Warning: Missing models for: {missing}")
    
    # Clear memory just in case, for a clean run
    tf.keras.backend.clear_session()
    gc.collect()

    # --- 3. Process Input FASTA ---
    print(f"\n--- Step 2: Processing Input File: {Path(input_fasta_path).name} ---")
    try:
        input_sequences = list(SeqIO.parse(input_fasta_path, "fasta"))
    except Exception as e:
        return {"error": f"Failed to parse FASTA file: {e}"}
        
    classified_results = Counter()
    unclassified_sequences = []

    for seq_record in tqdm(input_sequences, desc="  - Classifying sequences"):
        sequence_str = str(seq_record.seq)
        prediction_made = False
        for classifier in classifiers:
            label, prob = classifier.predict(sequence_str)
            if label:
                classified_results[label] += 1
                prediction_made = True
                break
        
        if not prediction_made:
            unclassified_sequences.append(seq_record)

    print(f"  - Classification complete.")
    print(f"    - Known organisms identified: {sum(classified_results.values())}")
    print(f"    - Unclassified sequences: {len(unclassified_sequences)}")
    
    # --- 4. Run Explorer Pipeline (if necessary) ---
    explorer_report_content = "No unclassified sequences to explore."
    if unclassified_sequences:
        print("\n--- Step 3: Starting 'Explorer' AI Pipeline ---")
        
        # --- REFACTORED: Call internal functions instead of subprocesses ---
        # The data is passed directly between these function calls in memory.
        sequence_vectors, sequence_ids = explorer_step_1_vectorize(unclassified_sequences)
        df_clusters = explorer_step_2_cluster(sequence_vectors, sequence_ids)
        explorer_report_content = explorer_step_3_interpret(df_clusters, unclassified_sequences, sequence_vectors)

        print("  - Explorer pipeline complete.")

    # --- 5. Generate Final Report ---
    print("\n--- Step 4: Generating Final Biodiversity Report ---")
    report_file_name = f"ATLAS_REPORT_{Path(input_fasta_path).stem}_{uuid.uuid4().hex}.txt"
    final_report_path = REPORTS_DIR / report_file_name
    
    with open(final_report_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("       ATLAS: AI Taxonomic Learning & Analysis System\n")
        f.write("                         FINAL REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"GPU Status: {gpu_status}\n")
        f.write(f"Input File: {Path(input_fasta_path).name}\n")
        f.write(f"Total Sequences Analyzed: {len(input_sequences)}\n\n")

        f.write("-" * 30 + "\n")
        f.write("Part 1: Known Organisms (Filter Results)\n")
        f.write("-" * 30 + "\n\n")
        if classified_results:
            for genus, count in sorted(classified_results.items(), key=lambda item: item[1], reverse=True):
                f.write(f"- {genus}: {count} sequences\n")
        else:
            f.write("No known organisms were identified by the Filter models.\n")
        
        f.write("\n\n" + "-" * 30 + "\n")
        f.write("Part 2: Novel Taxa Discovery (Explorer Results)\n")
        f.write("-" * 30 + "\n\n")
        f.write(explorer_report_content)
    
    print(f"  - Report saved successfully to: {final_report_path}")

    # Return a summary for the UI
    return {
        "status": "success",
        "report_path": str(final_report_path),
        "total_sequences": len(input_sequences),
        "classified": sum(classified_results.values()),
        "unclassified": len(unclassified_sequences),
        "classified_results": {k: v for k, v in classified_results.items()},
        "explorer_report": explorer_report_content
    }

# =============================================================================
# --- Main execution block for command-line use ---
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ATLAS: AI Taxonomic Learning & Analysis System")
    parser.add_argument(
        '--input_fasta', type=Path, required=True,
        help="Path to the input FASTA file for analysis."
    )
    args = parser.parse_args()
    
    run_analysis(args.input_fasta)
    print("\n[SUCCESS] ATLAS analysis complete.")
