# =============================================================================
# ATLAS - MASTER PREDICTION SCRIPT
# =============================================================================
# This is the main entry point for the ATLAS system. It orchestrates the
# entire workflow, combining the "Filter" and "Explorer" pipelines to generate
# a comprehensive biodiversity report from a user-provided FASTA file.
#
# WORKFLOW:
# 1.  Load all four trained "Filter" models (16S, 18S, COI, ITS).
# 2.  Process an input FASTA file sequence by sequence.
# 3.  For each sequence, attempt to classify it with each Filter model.
# 4.  If a model makes a high-confidence prediction, the result is stored.
# 5.  If no model can classify the sequence, it's added to a list of
#     "unclassified" sequences.
# 6.  The unclassified sequences are passed to the "Explorer" pipeline, which
#     runs its three scripts (vectorize, cluster, interpret) in order.
# 7.  The final report combines the classified results from the Filter with the
#     discovery report from the Explorer.
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
# --- FIX: Add the missing import for the progress bar ---
from tqdm import tqdm

# --- Configuration ---
try:
    project_root = Path(__file__).parent.parent
except NameError:
    project_root = Path.cwd()

MODELS_DIR = project_root / "models"
SRC_DIR = project_root / "src"

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
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads the model, vectorizer, and label encoder from disk."""
        try:
            self.model = load_model(MODELS_DIR / f"{self.name}_genus_classifier.keras")
            with open(MODELS_DIR / f"{self.name}_genus_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(MODELS_DIR / f"{self.name}_genus_label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
        except FileNotFoundError:
            print(f"[ERROR] Artifacts for {self.name} model not found. Please ensure all models are trained.")
            sys.exit(1)

    def predict(self, sequence, confidence_threshold=0.8):
        """Predicts the taxon for a single sequence."""
        kmer_counts = get_kmer_counts(sequence, self.kmer_size)
        
        # We need to handle the case where a sequence has no valid k-mers
        if not kmer_counts:
            return None, 0.0

        # The vectorizer expects a list of dictionaries
        vectorized_sequence = self.vectorizer.transform([kmer_counts])
        
        # The model makes a prediction
        prediction_probabilities = self.model.predict(vectorized_sequence, verbose=0)[0]
        
        # Get the highest probability and its corresponding class index
        top_prob = np.max(prediction_probabilities)
        top_class_index = np.argmax(prediction_probabilities)
        
        if top_prob >= confidence_threshold:
            predicted_label = self.label_encoder.inverse_transform([top_class_index])[0]
            return predicted_label, top_prob
        
        return None, top_prob

# =============================================================================
# --- Main Script Execution ---
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATLAS: AI Taxonomic Learning & Analysis System")
    parser.add_argument(
        '--input_fasta', type=Path, required=True,
        help="Path to the input FASTA file for analysis."
    )
    args = parser.parse_args()

    # --- 1. Load All Filter Models ---
    print("--- Step 1: Loading All 'Filter' AI Models ---")
    classifiers = [
        TaxonClassifier("16s", 6),
        TaxonClassifier("18s", 6),
        TaxonClassifier("coi", 8),
        TaxonClassifier("its", 7)
    ]
    print("  - All models loaded successfully.")

    # --- 2. Process Input FASTA ---
    print(f"\n--- Step 2: Processing Input File: {args.input_fasta.name} ---")
    input_sequences = list(SeqIO.parse(args.input_fasta, "fasta"))
    
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
                break # Move to the next sequence once classified
        
        if not prediction_made:
            unclassified_sequences.append(seq_record)

    print(f"  - Classification complete.")
    print(f"    - Known organisms identified: {sum(classified_results.values())}")
    print(f"    - Unclassified sequences: {len(unclassified_sequences)}")

    # --- 3. Run Explorer Pipeline (if necessary) ---
    explorer_report_content = "No unclassified sequences to explore."
    if unclassified_sequences:
        print("\n--- Step 3: Starting 'Explorer' AI Pipeline ---")
        
        # Save unclassified sequences to a temporary file
        temp_fasta_path = project_root / "temp_unclassified.fasta"
        SeqIO.write(unclassified_sequences, temp_fasta_path, "fasta")
        
        # Run the explorer scripts in sequence
        explorer_scripts = [
            "01_vectorize_sequences.py",
            "02_cluster_sequences.py",
            "03_interpret_clusters.py"
        ]
        
        for script in explorer_scripts:
            script_path = SRC_DIR / "pipeline_explorer" / script
            print(f"  - Running {script}...")
            # Use subprocess to run each script and wait for it to complete
            subprocess.run(
                [sys.executable, str(script_path), "--input_fasta", str(temp_fasta_path)],
                capture_output=True, text=True, check=True
            )
        
        # Read the explorer's final report
        explorer_report_path = project_root / "explorer_final_report.txt"
        if explorer_report_path.exists():
            with open(explorer_report_path, 'r') as f:
                explorer_report_content = f.read()
        
        # Clean up temporary files
        temp_fasta_path.unlink()
        explorer_report_path.unlink()
        print("  - Explorer pipeline complete.")

    # --- 4. Generate Final Report ---
    print("\n--- Step 4: Generating Final Biodiversity Report ---")
    # --- FIX: Create and use a dedicated reports directory ---
    REPORTS_DIR = project_root / "reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT_PATH = REPORTS_DIR / f"ATLAS_REPORT_{args.input_fasta.stem}.txt"
    
    with open(FINAL_REPORT_PATH, "w") as f:
        f.write("="*60 + "\n")
        f.write("       ATLAS: AI Taxonomic Learning & Analysis System\n")
        f.write("                         FINAL REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Input File: {args.input_fasta.name}\n")
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

    print(f"  - Report saved successfully to: {FINAL_REPORT_PATH}")
    print("\n[SUCCESS] ATLAS analysis complete.")


