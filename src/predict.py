# =============================================================================
# ATLAS - MASTER PREDICTION SCRIPT (REFRACTORED FOR WEB APP)
# =============================================================================
# This is the main entry point for the ATLAS system, now refactored to be
# callable as a function from a web application.
#
# The original logic has been moved into the `run_analysis` function.
#
# =============================================================================

# --- Imports ---
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

# --- Configuration ---
project_root = Path(__file__).parent.parent
MODELS_DIR = project_root / "models"
SRC_DIR = project_root / "src"
REPORTS_DIR = project_root / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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
            
            if not model_path.exists():
                print(f"Warning: Model for {self.name} not found at {model_path}")
                return False
            
            self.model = load_model(model_path)
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
    print(f"\n--- GPU Status: {gpu_status} ---")

    # --- 2. Load All Filter Models ---
    print("--- Step 1: Loading All 'Filter' AI Models ---")
    potential_classifiers = [
        TaxonClassifier("16s", 6),
        TaxonClassifier("18s", 6),
        TaxonClassifier("coi", 8),
        TaxonClassifier("its", 7)
    ]
    
    # Only keep successfully loaded classifiers
    classifiers = [clf for clf in potential_classifiers if clf.is_loaded]
    
    if not classifiers:
        return {"error": "No trained models found. Please ensure models are available."}
    
    print(f"  - Successfully loaded {len(classifiers)} models: {[clf.name for clf in classifiers]}")
    if len(classifiers) < len(potential_classifiers):
        missing = [clf.name for clf in potential_classifiers if not clf.is_loaded]
        print(f"  - Warning: Missing models for: {missing}")

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
        
        temp_fasta_path = REPORTS_DIR / "temp_unclassified.fasta"
        SeqIO.write(unclassified_sequences, temp_fasta_path, "fasta")
        
        explorer_scripts = [
            "01_vectorize_sequences.py",
            "02_cluster_sequences.py",
            "03_interpret_clusters.py"
        ]
        
        for script in explorer_scripts:
            script_path = SRC_DIR / "pipeline_explorer" / script
            print(f"  - Running {script}...")
            subprocess.run(
                [sys.executable, str(script_path), "--input_fasta", str(temp_fasta_path)],
                capture_output=True, text=True, check=True
            )
        
        explorer_report_path = project_root / "explorer_final_report.txt"
        if explorer_report_path.exists():
            with open(explorer_report_path, 'r') as f:
                explorer_report_content = f.read()
            explorer_report_path.unlink() # Clean up explorer report
        
        temp_fasta_path.unlink() # Clean up temp FASTA file
        print("  - Explorer pipeline complete.")

    # --- 5. Generate Final Report ---
    print("\n--- Step 4: Generating Final Biodiversity Report ---")
    final_report_path = REPORTS_DIR / f"ATLAS_REPORT_{Path(input_fasta_path).stem}.txt"
    
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

    # Return a summary for the web app
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
# --- Original script's main execution block (now a simple placeholder) ---
# This block is for running the script directly from the command line.
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATLAS: AI Taxonomic Learning & Analysis System")
    parser.add_argument(
        '--input_fasta', type=Path, required=True,
        help="Path to the input FASTA file for analysis."
    )
    args = parser.parse_args()
    
    run_analysis(args.input_fasta)
    print("\n[SUCCESS] ATLAS analysis complete.")
