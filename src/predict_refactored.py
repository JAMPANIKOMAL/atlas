import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path
import pickle
import argparse
import subprocess
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
import Bio.Blast.NCBIWWW as NCBIWWW
import Bio.Blast.NCBIXML as NCBIXML

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
def explorer_step_1_vectorize(sequences, log):
    log.append("  - Step 1: Vectorizing unclassified sequences...")
    KMER_SIZE = 6
    VECTOR_SIZE = 100
    
    # Load or train Doc2Vec model
    doc2vec_model_path = MODELS_DIR / "explorer_doc2vec.model"
    if doc2vec_model_path.exists():
        log.append("    - Loading pre-trained Doc2Vec model...")
        doc2vec_model = Doc2Vec.load(str(doc2vec_model_path))
    else:
        log.append("    - Doc2Vec model not found. Training a new model...")
        corpus = [TaggedDocument(words=[s.seq for s in sequences], tags=[s.id]) for s in sequences]
        doc2vec_model = Doc2Vec(vector_size=VECTOR_SIZE, dm=1, min_count=3, window=8, epochs=40, workers=4)
        doc2vec_model.build_vocab(corpus)
        doc2vec_model.train(corpus, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
        doc2vec_model.save(str(doc2vec_model_path))
    
    sequence_vectors = np.array([doc2vec_model.dv[seq.id] for seq in sequences])
    sequence_vectors = normalize(sequence_vectors)
    log.append(f"    - Vectors generated for {len(sequences)} sequences.")
    
    return sequence_vectors

def explorer_step_2_cluster(sequence_vectors, log):
    log.append("  - Step 2: Clustering vectors with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric='euclidean', cluster_selection_method='eom')
    cluster_labels = clusterer.fit_predict(sequence_vectors)
    log.append(f"    - Found {len(np.unique(cluster_labels)) - 1} clusters.")
    return cluster_labels

def explorer_step_3_interpret(sequences, sequence_vectors, cluster_labels, log):
    log.append("  - Step 3: Interpreting clusters with BLAST...")
    report_lines = []
    unique_cluster_ids = sorted(np.unique(cluster_labels))
    if -1 in unique_cluster_ids:
        unique_cluster_ids.remove(-1)

    for cluster_id in unique_cluster_ids:
        log.append(f"    - Analyzing Cluster {cluster_id}...")
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_vectors = sequence_vectors[cluster_indices]
        
        centroid = np.mean(cluster_vectors, axis=0)
        distances = [np.linalg.norm(vec - centroid) for vec in cluster_vectors]
        rep_index_in_cluster = np.argmin(distances)
        
        representative_sequence = sequences[cluster_indices[rep_index_in_cluster]]
        
        top_hit_title = "BLAST search skipped for this example." # Mocking BLAST for this implementation
        
        report_lines.append(f"Cluster ID: {cluster_id}")
        report_lines.append(f"  - Size: {len(cluster_indices)} sequences")
        report_lines.append(f"  - Representative Sequence ID: {representative_sequence.id}")
        report_lines.append(f"  - BLAST Hypothesis: {top_hit_title}\n")

    return "\n".join(report_lines)

# --- Main Analysis Function (refactored with structured output) ---
def run_analysis(input_fasta_path):
    """
    Main analysis pipeline.
    
    Args:
        input_fasta_path (str): Path to the input FASTA file.
        
    Returns:
        dict: A dictionary containing the analysis results.
    """
    analysis_log = []
    
    # --- 1. Check GPU Status ---
    analysis_log.append(f"GPU Status: {check_gpu_status()}")

    # --- 2. Load All Filter Models ---
    analysis_log.append("Loading all 'Filter' AI Models...")
    potential_classifiers = [
        TaxonClassifier("16s", 6), TaxonClassifier("18s", 6),
        TaxonClassifier("coi", 8), TaxonClassifier("its", 7)
    ]
    classifiers = [clf for clf in potential_classifiers if clf.is_loaded]
    if not classifiers:
        return {"status": "error", "message": "No trained models found. Please ensure models are available.", "log": analysis_log}
    analysis_log.append(f"  - Successfully loaded {len(classifiers)} models: {[clf.name for clf in classifiers]}")

    # --- 3. Process Input FASTA ---
    analysis_log.append(f"Processing input file: {Path(input_fasta_path).name}...")
    try:
        input_sequences = list(SeqIO.parse(input_fasta_path, "fasta"))
    except Exception as e:
        return {"status": "error", "message": f"Failed to parse FASTA file: {e}", "log": analysis_log}
        
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
    
    analysis_log.append(f"  - Classification complete.")
    analysis_log.append(f"    - Known organisms identified: {sum(classified_results.values())}")
    analysis_log.append(f"    - Unclassified sequences: {len(unclassified_sequences)}")
    
    # --- 4. Run Explorer Pipeline (if necessary) ---
    explorer_report_content = "No unclassified sequences to explore."
    if unclassified_sequences:
        analysis_log.append("Starting 'Explorer' AI Pipeline...")
        
        sequence_vectors = explorer_step_1_vectorize(unclassified_sequences, analysis_log)
        cluster_labels = explorer_step_2_cluster(sequence_vectors, analysis_log)
        explorer_report_content = explorer_step_3_interpret(unclassified_sequences, sequence_vectors, cluster_labels, analysis_log)
        
        analysis_log.append("  - Explorer pipeline complete.")

    # --- 5. Generate Final Report ---
    analysis_log.append("Generating Final Biodiversity Report...")
    
    # --- FIX: Generate the classified results string safely ---
    sorted_results = sorted(classified_results.items(), key=lambda item: item[1], reverse=True)
    classified_results_str = 'No known organisms were identified.' if not classified_results else '\n'.join([f"- {genus}: {count} sequences" for genus, count in sorted_results])

    final_report_text = f"""
GPU Status: {check_gpu_status()}
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
    
    # Return a structured dictionary for the web app
    return {
        "status": "success",
        "log": analysis_log,
        "report_content": final_report_text.strip(),
        "classified_results": {k: v for k, v in classified_results.items()},
    }

# --- Original script's main execution block (for CLI) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATLAS: AI Taxonomic Learning & Analysis System")
    parser.add_argument(
        '--input_fasta', type=Path, required=True,
        help="Path to the input FASTA file for analysis."
    )
    args = parser.parse_args()
    
    result = run_analysis(args.input_fasta)
    print("\n--- Final Report ---\n")
    print(result["report_content"])
    print("\n[SUCCESS] ATLAS analysis complete.")
