from flask import Flask, render_template, request, jsonify, send_from_directory
import sys
from pathlib import Path
import os
import tempfile
import uuid

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.predict import run_analysis

# Use the full path for the static and template folders
web_app_dir = Path(__file__).parent
app = Flask(__name__,
            static_folder=str(web_app_dir / "static"),
            template_folder=str(web_app_dir / "templates"))

# A new route to handle a specific part of the static file path.
# This helps Flask find files nested in sub-folders of /static.
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve files from the static folder."""
    return send_from_directory(app.static_folder, filename)

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
    
    if not sequences:
        return jsonify({"error": "No sequences provided."}), 400

    # Write the sequences to a temporary FASTA file
    temp_dir = Path(tempfile.gettempdir())
    input_file_name = f"input_{uuid.uuid4().hex}.fasta"
    temp_fasta_path = temp_dir / input_file_name
    
    try:
        with open(temp_fasta_path, "w") as f:
            f.write(sequences)
    except IOError as e:
        return jsonify({"error": f"Failed to write temporary file: {e}"}), 500
        
    try:
        # Run the refactored analysis pipeline
        analysis_results = run_analysis(str(temp_fasta_path))
        return jsonify(analysis_results)
    except Exception as e:
        # Log the error for debugging
        print(f"Analysis failed with an error: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the temporary file
        if temp_fasta_path.exists():
            os.remove(temp_fasta_path)

@app.route('/reports/<path:filename>')
def serve_report(filename):
    """Serves the generated reports for download."""
    reports_dir = project_root / "reports"
    return send_from_directory(reports_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)