# =============================================================================
# ATLAS: INTERACTIVE CLI TOOL
# =============================================================================
# This script provides a user-friendly command-line interface for the ATLAS
# analysis pipeline. It is the primary entry point for the application,
# handling argument parsing and user interaction.
#
# USAGE:
#
# 1. To run interactively:
#    python -m cli
#
# 2. To run with a direct file path:
#    python -m cli --input_fasta "path/to/your/file.fasta"
#
# =============================================================================

# --- Imports ---
import sys
import argparse
from pathlib import Path

# Add the 'src' directory to the Python path to allow importing predict.py
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

try:
    from predict import run_analysis
except ImportError:
    print("Error: Could not import the analysis engine.")
    print("Please ensure you are running this script from the project's root directory.")
    sys.exit(1)

# --- ASCII Art Header for "ATLAS" ---
ATLAS_ASCII = r"""
    ___  _____ _      ___  ____  
  / _ \|_   _| |    / _ \/ ___| 
 | |_| | | | | |   | |_| \___ \ 
 |  _  | | | | |___|  _  |___) |
 |_| |_| |_| |_____|_| |_|____/ 
                                          
 A.T.L.A.S. Command-Line Interface
---------------------------------------
"""

def main():
    """
    The main function for the command-line tool.
    
    It parses command-line arguments or prompts the user for a file path,
    then calls the core analysis engine.
    """
    parser = argparse.ArgumentParser(
        description="ATLAS: AI Taxonomic Learning & Analysis System. Processes a FASTA file.",
        usage=f"python -m cli --input_fasta <path_to_file>"
    )
    parser.add_argument(
        '--input_fasta',
        type=Path,
        help="Path to the input FASTA file for analysis."
    )
    
    args = parser.parse_args()
    fasta_path = args.input_fasta

    # If no argument is provided, switch to interactive mode
    if not fasta_path:
        print(ATLAS_ASCII)
        print("Welcome to the ATLAS analysis tool.")
        print("This program will process a FASTA file and generate a taxonomic report.")
        print("To proceed, please enter the path to your FASTA file.")
        print("Example: data/raw/your_file.fasta\n")
        input_path = input("Enter FASTA file path: ")
        fasta_path = Path(input_path.strip())

    if not fasta_path.is_file():
        print(f"\n[ERROR] File not found: {fasta_path}")
        print("Please check the path and try again.")
        sys.exit(1)

    print(f"\nProcessing '{fasta_path}'...")
    print("This may take a few minutes, please wait...")

    # Call the core analysis function from the predict.py module
    result = run_analysis(fasta_path)

    # Print the report from the returned dictionary
    if result["status"] == "success":
        print("\n" + result["report_content"])
    else:
        print(f"\n[ERROR] An error occurred during the analysis:")
        print(f"  > {result['message']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
