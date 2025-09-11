#!/usr/bin/env python3
# =============================================================================
# ATLAS: MASTER PREDICTION SCRIPT
# =============================================================================
# This script serves as the primary engine for the ATLAS system's analysis pipeline.
# It is designed to be a single, callable module that can process a FASTA file from
# start to finish. It integrates the logic of both the "Filter" (classification)
# and "Explorer" (clustering) pipelines in a memory-safe and efficient manner.
#
# USAGE FROM COMMAND LINE:
#
#   To run an analysis, execute the script with the `--input_fasta` argument:
#
#   python src/predict.py --input_fasta "path/to/your/file.fasta"
#
#   The script will print a detailed report to the console.
#
# =============================================================================

# --- Core Imports ---
import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path
import pickle
import argparse
import sys
from collections import Counter
import gc
import io

# --- Machine Learning & Data Processing Imports ---
from tensorflow.keras.models import load_model
from scipy.sparse import csc_matrix
import tensorflow as tf
import hdbscan
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize
from tqdm import tqdm

# --- Configuration ---
# All file paths are relative to the project root for portability.
project_root = Path(__file__).parent.parent
MODELS_DIR = project_root / "models"
REPORTS_DIR = project_root / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Functions ---

def get_kmer_counts(sequence: str, k: int) -> dict:
    """
    Calculates k-mer counts for a single DNA sequence.

    This function iterates through the sequence, extracting all k-mers
    (sub-sequences of length k) and counting their occurrences. It ignores
    any k-mer that contains an 'N', which typically represents an unknown
    nucleotide.

    Args:
        sequence (str): The DNA sequence string.
        k (int): The size of the k-mer.

    Returns:
        dict: A dictionary where keys are k-mers and values are their counts.
    """
    counts = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if "N" not in kmer.upper():
            counts[kmer] += 1
    return dict(counts)

# --- Classifier Class ---

class TaxonClassifier:
    """
    A wrapper class to load and use a trained Keras model and its artifacts.

    This class encapsulates the model, DictVectorizer, and LabelEncoder,
    ensuring they are all loaded correctly for a specific marker (e.g., 16s, coi).
    """
    def __init__(self, marker_name: str, kmer_size: int):
        self.name = marker_name
        self.kmer_size = kmer_size
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        # Check if all necessary artifacts exist and load them
        self.is_loaded = self._load_artifacts()

    def _load_artifacts(self) -> bool:
        """
        Loads the model, vectorizer, and label encoder from disk.

        Returns:
            bool: True if all artifacts were successfully loaded, False otherwise.
        """
        try:
            model_path = MODELS_DIR / f"{self.name}_genus_classifier.keras"
            vectorizer_path = MODELS_DIR / f"{self.name}_genus_vectorizer.pkl"
            encoder_path = MODELS_DIR / f"{self.name}_genus_label_encoder.pkl"
            
            # All three files must exist for the model to be usable
            if not model_path.exists() or not vectorizer_path.exists() or not encoder_path.exists():
                return False
            
            # Load the artifacts
            self.model = load_model(model_path)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading {self.name} model: {e}", file=sys.stderr)
            return False

    def predict(self, sequence: str, confidence_threshold: float = 0.8) -> tuple[str, float]:
        """
        Predicts the taxonomic genus for a single DNA sequence.

        Args:
            sequence (str): The DNA sequence to classify.
            confidence_threshold (float): The minimum probability required to
                                          consider a prediction valid.

        Returns:
            tuple[str, float]: The predicted genus (or None) and the
                               prediction probability.
        """
        if not self.is_loaded:
            return None, 0.0
            
        kmer_counts = get_kmer_counts(sequence, self.kmer_size)
        
        if not kmer_counts:
            return None, 0.0

        # Transform the k-mer counts into the numerical vector the model expects
        vectorized_sequence = self.vectorizer.transform([kmer_counts])
        
        # Predict the probabilities for all classes
        prediction_probabilities = self.model.predict(vectorized_sequence, verbose=0)[0]
        
        # Get the highest probability and its corresponding class index
        top_prob = np.max(prediction_probabilities)
        top_class_index = np.argmax(prediction_probabilities)
        
        # Return the prediction only if it meets the confidence threshold
        if top_prob >= confidence_threshold:
            predicted_label = self.label_encoder.inverse_transform([top_class_index])[0]
            return predicted_label, top_prob
        
        # Otherwise, return None for the label
        return None, top_prob

# --- System & Explorer Functions ---

def check_gpu_status() -> str:
    """Checks and returns the GPU availability status."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return f"GPU is available: {gpus[0].name}"
    else:
        return "GPU not found. Running on CPU."

def explorer_step_1_vectorize(sequences: list) -> np.ndarray:
    """
    Vectorizes unclassified sequences using a pre-trained Doc2Vec model.

    Args:
        sequences (list): A list of Bio.SeqRecord objects.

    Returns:
        np.ndarray: A normalized matrix of sequence vectors.
    """
    KMER_SIZE = 6
    doc2vec_model_path = MODELS_DIR / "explorer_doc2vec.model"
    
    if not doc2vec_model_path.exists():
        return np.array([])
    
    doc2vec_model = Doc2Vec.load(str(doc2vec_model_path))
    
    corpus = [
        TaggedDocument(
            words=[str(s.seq) for s in sequences], tags=[s.id]
        ) for s in sequences
    ]
    
    sequence_vectors = np.array([doc2vec_model.dv[seq.id] for seq in corpus])
    sequence_vectors = normalize(sequence_vectors)
    
    return sequence_vectors

def explorer_step_2_cluster(sequence_vectors: np.ndarray) -> np.ndarray:
    """
    Clusters sequence vectors using HDBSCAN.

    Args:
        sequence_vectors (np.ndarray): The matrix of sequence vectors.

    Returns:
        np.ndarray: An array of cluster labels for each sequence.
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric='euclidean', cluster_selection_method='eom')
    cluster_labels = clusterer.fit_predict(sequence_vectors)
    return cluster_labels

def explorer_step_3_interpret(sequences: list, sequence_vectors: np.ndarray, cluster_labels: np.ndarray) -> str:
    """
    Generates a report for the discovered clusters.

    It finds a representative sequence for each cluster and summarizes the findings.
    Note: This version does not perform BLAST, as that is a slow, online process.

    Args:
        sequences (list): The list of original Bio.SeqRecord objects.
        sequence_vectors (np.ndarray): The sequence vectors.
        cluster_labels (np.ndarray): The cluster assignments.

    Returns:
        str: A human-readable report string.
    """
    report_lines = []
    unique_cluster_ids = sorted(np.unique(cluster_labels))
    if -1 in unique_cluster_ids:
        unique_cluster_ids.remove(-1)

    if not unique_cluster_ids:
        return "No significant clusters were discovered in the unclassified sequences."

    for cluster_id in unique_cluster_ids:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_vectors = sequence_vectors[cluster_indices]
        
        centroid = np.mean(cluster_vectors, axis=0)
        distances = [np.linalg.norm(vec - centroid) for vec in cluster_vectors]
        rep_index_in_cluster = np.argmin(distances)
        
        representative_sequence = sequences[cluster_indices[rep_index_in_cluster]]
        
        report_lines.append(f"Cluster ID: {cluster_id}")
        report_lines.append(f"  - Size: {len(cluster_indices)} sequences")
        report_lines.append(f"  - Representative Sequence ID: {representative_sequence.id}")
        report_lines.append(f"  - Note: BLAST is not run automatically, as it is a slow external step. A manual BLAST search on this representative sequence ID is recommended to hypothesize a taxonomic identity.\n")

    return "\n".join(report_lines)

# =============================================================================
# --- Main Analysis Function ---
# =============================================================================
def run_analysis(input_fasta_path: Path):
    """
    The main analysis pipeline.

    This is the core function of the script. It processes the input FASTA file
    in two stages:
    1.  **Filter**: It attempts to classify each sequence using the four
        pre-trained neural network models (16s, 18s, coi, its).
    2.  **Explorer**: Any sequences that could not be classified are then
        passed to the Explorer pipeline for novel taxa discovery via
        clustering.

    Args:
        input_fasta_path (Path): The Path object to the input FASTA file.
        
    Returns:
        dict: A dictionary containing the analysis results, including a
              formatted report, and structured data for potential further use.
    """
    
    # --- 1. Check GPU Status ---
    gpu_status = check_gpu_status()

    # --- 2. Load and Run Filter Models Sequentially ---
    potential_classifiers = [
        ("16s", 6), ("18s", 6), ("coi", 8), ("its", 7)
    ]
    
    classified_results = Counter()
    unclassified_sequences = []

    try:
        input_sequences = list(SeqIO.parse(input_fasta_path, "fasta"))
    except Exception as e:
        return {"status": "error", "message": f"Failed to parse FASTA file: {e}"}
    
    unclassified_sequences = input_sequences.copy()

    # The sequential loading and unloading of models is critical for memory management.
    for name, kmer_size in potential_classifiers:
        classifier = TaxonClassifier(name, kmer_size)
        if classifier.is_loaded:
            newly_classified = []
            sequences_to_reprocess = []
            
            for seq_record in unclassified_sequences:
                label, prob = classifier.predict(str(seq_record.seq))
                if label:
                    classified_results[label] += 1
                    newly_classified.append(seq_record)
                else:
                    sequences_to_reprocess.append(seq_record)
            
            unclassified_sequences = sequences_to_reprocess
        
        # Explicitly unload the model and free memory
        del classifier
        tf.keras.backend.clear_session()
        gc.collect()

    # --- 3. Run Explorer Pipeline (if necessary) ---
    explorer_report_content = ""
    if unclassified_sequences:
        try:
            sequence_vectors = explorer_step_1_vectorize(unclassified_sequences)
            cluster_labels = explorer_step_2_cluster(sequence_vectors)
            explorer_report_content = explorer_step_3_interpret(unclassified_sequences, sequence_vectors, cluster_labels)
        except Exception as e:
            explorer_report_content = f"Error during explorer pipeline: {e}"
    else:
         explorer_report_content = "All sequences were classified by the Filter models. No need for the Explorer pipeline."
    
    # --- 4. Generate Final Report ---
    sorted_results = sorted(classified_results.items(), key=lambda item: item[1], reverse=True)
    classified_results_str = 'No known organisms were identified.' if not classified_results else '\n'.join([f"- {genus}: {count} sequences" for genus, count in sorted_results])

    final_report_text = f"""
============================================================
          ATLAS: AI Taxonomic Learning & Analysis System
                         FINAL REPORT
============================================================

[  ENVIRONMENT & INPUT  ]
------------------------------------------------------------
GPU Status: {gpu_status}
Input File: {Path(input_fasta_path).name}
Total Sequences Analyzed: {len(input_sequences)}

[  PART 1: FILTER RESULTS  ]
------------------------------------------------------------
(Classification of known organisms)
{classified_results_str}

[  PART 2: EXPLORER RESULTS  ]
------------------------------------------------------------
(Discovery of novel or unclassified taxa)
{explorer_report_content}
"""
    
    return {
        "status": "success",
        "report_content": final_report_text.strip(),
        "classified_results": {k: v for k, v in classified_results.items()},
    }

# =============================================================================
# --- Main Execution Block (for CLI) ---
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ATLAS: AI Taxonomic Learning & Analysis System. Processes a FASTA file through filter and explorer pipelines."
    )
    parser.add_argument(
        '--input_fasta', 
        type=Path, 
        required=True,
        help="Path to the input FASTA file for analysis."
    )
    args = parser.parse_args()
    
    # Run the main analysis function
    result = run_analysis(args.input_fasta)

    # Print the final report to the console
    print(result["report_content"])
    print("\n[SUCCESS] ATLAS analysis complete.")
