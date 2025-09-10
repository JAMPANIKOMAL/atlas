# =============================================================================
# ATLAS - EXPLORER PIPELINE - SCRIPT 3: INTERPRET CLUSTERS
# =============================================================================
# This is the final script in the Explorer pipeline. It takes the discovered
# clusters, identifies the most representative sequence for each, and
# automatically performs a BLAST search to assign a taxonomic hypothesis.
#
# WORKFLOW:
# 1.  Loads the cluster results CSV, the original FASTA file, and the sequence
#     vectors.
# 2.  For each cluster (ignoring noise label -1):
#     a. Calculates the cluster's geometric center (centroid).
#     b. Finds the sequence closest to the centroid.
#     c. Submits this representative sequence to the NCBI BLAST API.
#     d. Parses the result to get the top hit's scientific name.
# 3.  Generates a final, human-readable report summarizing the findings.
# =============================================================================

# --- Imports ---
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from pathlib import Path
import argparse
import io

# --- Configuration ---
try:
    project_root = Path(__file__).parent.parent.parent
except NameError:
    project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()

PROCESSED_DATA_DIR = project_root / "data" / "processed"
RAW_DATA_DIR = project_root / "data" / "raw"

# =============================================================================
# --- Main Script Execution ---
# =============================================================================
if __name__ == "__main__":
    # --- Command Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Interpret clusters by finding representatives and running BLAST.")
    parser.add_argument(
        '--clusters_path', type=Path,
        default=PROCESSED_DATA_DIR / "explorer_cluster_results.csv",
        help="Path to the cluster results CSV file."
    )
    parser.add_argument(
        '--fasta_path', type=Path,
        default=RAW_DATA_DIR / "unclassified_sample_for_explorer.fasta",
        help="Path to the original FASTA file used for vectorization."
    )
    parser.add_argument(
        '--vectors_path', type=Path,
        default=PROCESSED_DATA_DIR / "explorer_sequence_vectors.npy",
        help="Path to the sequence vectors .npy file."
    )
    args = parser.parse_args()
    
    # --- 1. Load All Necessary Data ---
    print("--- Step 1: Loading All Input Data ---")
    df_clusters = pd.read_csv(args.clusters_path)
    sequence_vectors = np.load(args.vectors_path)
    
    # Create a dictionary for fast sequence lookups
    sequences_dict = {rec.id: str(rec.seq) for rec in SeqIO.parse(args.fasta_path, "fasta")}
    
    # Create a mapping from sequence ID to its vector index
    id_to_vector_index = {seq_id: i for i, seq_id in enumerate(df_clusters['sequence_id'])}
    
    print(f"  - Loaded {len(df_clusters)} cluster assignments.")

    # --- 2. Analyze Each Cluster and Perform BLAST Search ---
    print("\n--- Step 2: Analyzing Clusters and Querying BLAST ---")
    
    report_lines = []
    unique_cluster_ids = sorted(df_clusters['cluster_label'].unique())
    if -1 in unique_cluster_ids:
        unique_cluster_ids.remove(-1) # Ignore the noise points

    for cluster_id in unique_cluster_ids:
        print(f"\n--- Analyzing Cluster {cluster_id} ---")
        
        # Get all sequence IDs for the current cluster
        cluster_df = df_clusters[df_clusters['cluster_label'] == cluster_id]
        member_ids = cluster_df['sequence_id'].tolist()
        
        # Get the vector indices for these members
        member_indices = [id_to_vector_index[seq_id] for seq_id in member_ids]
        cluster_vectors = sequence_vectors[member_indices]
        
        # Find the representative sequence (closest to centroid)
        centroid = np.mean(cluster_vectors, axis=0)
        distances = [np.linalg.norm(vec - centroid) for vec in cluster_vectors]
        rep_index_in_cluster = np.argmin(distances)
        representative_id = member_ids[rep_index_in_cluster]
        representative_sequence = sequences_dict[representative_id]
        
        print(f"  - Cluster Size: {len(member_ids)}")
        print(f"  - Representative ID: {representative_id}")
        print("  - Submitting to NCBI BLAST (this can take a few minutes)...")
        
        # Perform BLAST search
        try:
            result_handle = NCBIWWW.qblast(
                program="blastn", database="nt", sequence=representative_sequence
            )
            blast_record = NCBIXML.read(result_handle)
            
            top_hit_title = "No significant similarity found."
            if blast_record.alignments:
                top_hit_title = blast_record.alignments[0].title
            
            print("  - BLAST search complete.")
            
        except Exception as e:
            print(f"  - [ERROR] BLAST query failed: {e}")
            top_hit_title = "BLAST query failed."

        # Append to report
        report_lines.append(f"Cluster ID: {cluster_id}")
        report_lines.append(f"  - Size: {len(member_ids)} sequences")
        report_lines.append(f"  - Representative Sequence ID: {representative_id}")
        report_lines.append(f"  - BLAST Hypothesis: {top_hit_title}\n")

    # --- 3. Save Final Report ---
    REPORT_OUTPUT_PATH = project_root / "explorer_final_report.txt"
    print(f"\n--- Step 3: Saving Final Report ---")
    with open(REPORT_OUTPUT_PATH, "w") as f:
        f.write("="*50 + "\n")
        f.write("      ATLAS EXPLORER - FINAL REPORT\n")
        f.write("="*50 + "\n\n")
        if report_lines:
            f.write("\n".join(report_lines))
        else:
            f.write("No significant clusters were discovered in the input data.\n")
    print(f"  - Report saved successfully to: {REPORT_OUTPUT_PATH}")

    # --- Final ASCII Art Confirmation ---
    print("\n" + "="*50)
    print("      EXPLORER INTERPRETATION SCRIPT COMPLETE")
    print("="*50 + "\n")
