from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path
import subprocess

# Add the project root to the Python path to import predict.py
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.predict import run_analysis # Assuming a function named run_analysis in predict.py

app = Flask(__name__,
            static_folder=str(Path(__file__).parent / "static"),
            template_folder=str(Path(__file__).parent / "templates"))

@app.route('/')
def home():
    """Serves the main homepage."""
    return render_template('index.html')

@app.route('/analysis')
def analysis_page():
    """Serves the DNA analysis page."""
    return render_template('analysis.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_sequence():
    """
    Receives DNA sequences from the frontend, runs the ATLAS analysis,
    and returns the results.
    """
    data = request.json
    sequences = data.get('sequences')
    analysis_mode = data.get('analysisMode')

    # This is where you would call your main predict.py script
    # The current predict.py uses argparse, so you'll need to adapt this.
    # For now, we'll simulate the call and return a dummy result.
    
    # You would need to write the sequences to a temporary FASTA file
    # and then call your predict.py script with the correct arguments.
    # For example:
    # temp_fasta_path = "temp.fasta"
    # with open(temp_fasta_path, "w") as f:
    #     f.write(sequences)
    #
    # subprocess.run([sys.executable, str(project_root / "src" / "predict.py"), "--input_fasta", temp_fasta_path])
    #
    # Then read the generated report and return it.
    
    # Placeholder for the actual processing logic
    print(f"Received sequences for analysis in {analysis_mode} mode.")
    print("Simulating analysis...")
    
    # Return a dummy response to the frontend for now
    results = {
        "status": "success",
        "report_url": "/path/to/generated_report.txt",
        "summary": {
            "total_sequences": len(sequences.split('>')) - 1,
            "classified": 10,
            "unclassified": 2,
            "novel_clusters": 1,
        }
    }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)