# =============================================================================
# ATLAS: INTERACTIVE CLI TOOL
# =============================================================================
# This script provides a user-friendly command-line interface for the ATLAS
# analysis pipeline. It guides the user to a FASTA file, runs the analysis,
# and displays the final report.
#
# USAGE:
#
#   To run the tool, navigate to the project root and execute it as a module:
#
#   python -m cli
#
# =============================================================================

# --- Imports ---
import sys
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

# --- ASCII Art Header ---
ATLAS_ASCII = r"""
  ___ _____ _   _ _____ _____
 / _ \_   _| | | |_   _|  ___|
| | | || | | |_| | | | | |__
| | | || | |  _  | | | |  __|
| |_| || | | | | |_| |_| |___
 \___/ \_/ \_| |_/\___/\____/

 A.T.L.A.S. Command-Line Interface
---------------------------------------
"""

def main():
    """
    The main function for the interactive command-line tool.
    """
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
