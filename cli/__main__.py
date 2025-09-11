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
# 2. To run with a direct file path and options:
#    python -m cli --input_fasta "path/to/your/file.fasta" [--report-name <name>] [--verbose]
#
# =============================================================================

# --- Imports ---
import sys
import argparse
from pathlib import Path
import logging

# Set up logging for verbose output
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Add the 'src' directory to the Python path to allow importing predict.py
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

try:
    from predict import run_analysis
except ImportError:
    print("Error: Could not import the analysis engine.")
    print("Please ensure you are running this script from the project's root directory.")
    sys.exit(1)

# --- ASCII Art Headers ---
ATLAS_ASCII = r"""
    _    _____ _        _    ____  
   / \  |_   _| |      / \  / ___| 
  / _ \   | | | |     / _ \ \___ \ 
 / ___ \  | | | |___ / ___ \ ___) |
/_/   \_\ |_| |_____/_/   \_\____/ 

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
        usage="python -m cli --input_fasta <path_to_file> [--report-name <name>] [--verbose]"
    )
    parser.add_argument(
        '--input_fasta',
        type=Path,
        help="Path to the input FASTA file for analysis."
    )
    parser.add_argument(
        '--report-name',
        type=str,
        help="A custom name for the output report file (e.g., 'my_report')."
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose output to show detailed analysis steps."
    )

    args = parser.parse_args()
    fasta_path = args.input_fasta

    # Set logging level based on the --verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # If no argument is provided, switch to interactive mode
    if not fasta_path:
        print(ATLAS_ASCII)
        logging.warning("Welcome to the ATLAS analysis tool.")
        logging.warning("This program will process a FASTA file and generate a taxonomic report.")
        logging.warning("To proceed, please enter the path to your FASTA file.")
        logging.warning("Example: data/raw/your_file.fasta\n")
        input_path = input("Enter FASTA file path: ")
        fasta_path = Path(input_path.strip())

    if not fasta_path.is_file():
        logging.error(f"\n[ERROR] File not found: {fasta_path}")
        logging.error("Please check the path and try again.")
        sys.exit(1)

    logging.info(f"\nProcessing '{fasta_path}'...")
    logging.info("This may take a few minutes, please wait...")

    # Call the core analysis function from the predict.py module
    # Pass the report_name and verbosity flag
    result = run_analysis(
        input_fasta_path=fasta_path,
        report_name=args.report_name,
        verbose=args.verbose
    )

    # Print the report from the returned dictionary
    if result["status"] == "success":
        print("\n" + result["report_content"])
    else:
        logging.error(f"\n[ERROR] An error occurred during the analysis:")
        logging.error(f"  > {result['message']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
