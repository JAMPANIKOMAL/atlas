#!/usr/bin/env python3
# =============================================================================
# ATLAS - MASTER PREDICTION SCRIPT
# =============================================================================
# This script is the core prediction engine for the ATLAS system. It provides
# a single, robust function to analyze a FASTA file by running it through
# both the "Filter" and "Explorer" AI pipelines.
#
# The original logic from 'predict_refactored.py' has been used as a base
# as it is more stable and performs all operations in-memory without relying
# on fragile subprocess calls to other scripts.
#
# =============================================================================

# --- Imports ---
import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path
import pickle
import argparse
import sys
from collections import Counter
from tensorflow.keras.models import load_model
from scipy.sparse import csc_matrix
from tqdm import tqdm
import io
import os
import shutil
import tensorflow as tf
from contextlib import redirect_stdout
import uuid
import hdbscan
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize
import gc

# --- Configuration ---
project_root = Path(__file__).parent.parent
MODELS_DIR = project_root / "models"
REPORTS_DIR = project_root / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_EXPLORER_DIR = REPORTS_DIR / "temp"

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
            
            if not model_path.exists() or not vectorizer_path.exists() or not encoder_path.exists():
                return False
            
            self.model = load_model(model_path)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            return True
        except Exception as e:
            # Added for debugging purposes
            print(f"Error loading {self.name} model: {e}", file=sys.stderr)
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
        return f"GPU is available: {gpus[0].name}"
    else:
        return "GPU not found. Running on CPU."

# --- Explorer Pipeline Functions (now in-memory) ---
# NOTE: These functions are copied from the explorer pipeline scripts
# and modified to work in-memory and return values directly.
def explorer_step_1_vectorize(sequences):
    KMER_SIZE = 6
    VECTOR_SIZE = 100
    
    doc2vec_model_path = MODELS_DIR / "explorer_doc2vec.model"
    if doc2vec_model_path.exists():
        doc2vec_model = Doc2Vec.load(str(doc2vec_model_path))
    else:
        # NOTE: A real-world application would need a way to train this model
        # if it's not present. For this CLI, we assume it's pre-trained.
        print("Warning: Explorer model not found. This pipeline will fail.")
        return np.array([]), np.array([])
    
    corpus = [
        TaggedDocument(
            words=[str(s.seq) for s in sequences], tags=[s.id]
        ) for s in sequences
    ]
    
    sequence_vectors = np.array([doc2vec_model.dv[seq.id] for seq in corpus])
    sequence_vectors = normalize(sequence_vectors)
    
    return sequence_vectors

def explorer_step_2_cluster(sequence_vectors):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric='euclidean', cluster_selection_method='eom')
    cluster_labels = clusterer.fit_predict(sequence_vectors)
    return cluster_labels

def explorer_step_3_interpret(sequences, sequence_vectors, cluster_labels):
    report_lines = []
    unique_cluster_ids = sorted(np.unique(cluster_labels))
    if -1 in unique_cluster_ids:
        unique_cluster_ids.remove(-1)

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
        report_lines.append(f"  - Note: BLAST functionality is not implemented in this version of the script.")

    return "\n".join(report_lines)

# =============================================================================
# --- Main Analysis Function ---
# =============================================================================
def run_analysis(input_fasta_path):
    """
    Main analysis pipeline.
    
    Args:
        input_fasta_path (str): Path to the input FASTA file.
        
    Returns:
        dict: A dictionary containing the analysis results.
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

    for name, kmer_size in potential_classifiers:
        # Load one model at a time
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
        
        # Unload the model and clear memory
        del classifier
        tf.keras.backend.clear_session()
        gc.collect()

    # --- 3. Run Explorer Pipeline (if necessary) ---
    explorer_report_content = "No unclassified sequences to explore."
    if unclassified_sequences:
        try:
            sequence_vectors = explorer_step_1_vectorize(unclassified_sequences)
            cluster_labels = explorer_step_2_cluster(sequence_vectors)
            explorer_report_content = explorer_step_3_interpret(unclassified_sequences, sequence_vectors, cluster_labels)
        except Exception as e:
            explorer_report_content = f"Error during explorer pipeline: {e}"
    
    # --- 4. Generate Final Report ---
    sorted_results = sorted(classified_results.items(), key=lambda item: item[1], reverse=True)
    classified_results_str = 'No known organisms were identified.' if not classified_results else '\n'.join([f"- {genus}: {count} sequences" for genus, count in sorted_results])

    final_report_text = f"""
============================================================
          ATLAS: AI Taxonomic Learning & Analysis System
                        FINAL REPORT
============================================================

GPU Status: {gpu_status}
Input File: {Path(input_fasta_path).name}
Total Sequences Analyzed: {len(input_sequences)}

--------------------------------------
Part 1: Known Organisms (Filter Results)
--------------------------------------
{classified_results_str}

--------------------------------------
Part 2: Novel Taxa Discovery (Explorer Results)
--------------------------------------
{explorer_report_content}
"""
    
    return {
        "status": "success",
        "report_content": final_report_text.strip(),
        "classified_results": {k: v for k, v in classified_results.items()},
    }

# --- Main execution block for command-line interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATLAS: AI Taxonomic Learning & Analysis System")
    parser.add_argument(
        '--input_fasta', type=Path, required=True,
        help="Path to the input FASTA file for analysis."
    )
    args = parser.parse_args()
    
    # Use the run_analysis function
    result = run_analysis(args.input_fasta)

    # Print a clean, formatted report to the console
    print(result["report_content"])
    print("\n[SUCCESS] ATLAS analysis complete.")
