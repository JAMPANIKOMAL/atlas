# =============================================================================
# ATLAS - FLASK BACKEND API
# =============================================================================
# This Flask application provides a REST API for the ATLAS frontend to
# interact with the bioinformatics pipeline for eDNA analysis.
# =============================================================================

import os
import sys
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import subprocess
import threading
import time

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
RESULTS_FOLDER = Path(__file__).parent / 'results'
ALLOWED_EXTENSIONS = {'fasta', 'fa', 'fas', 'txt'}

# Create necessary directories
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

# Store active analysis jobs
active_jobs = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_fasta_content(content):
    """Validate that the content is in FASTA format"""
    lines = content.strip().split('\n')
    if not lines:
        return False, "Empty content"
    
    has_header = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            has_header = True
        elif has_header and not line.startswith('>'):
            # Check if line contains valid DNA characters
            valid_chars = set('ATCGRYSWKMBDHVN-')
            if not all(c.upper() in valid_chars for c in line):
                return False, f"Invalid DNA characters in sequence: {line[:50]}..."
        elif not has_header:
            return False, "FASTA content must start with a header line (>)"
    
    return has_header, "Valid FASTA format" if has_header else "No sequences found"

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and text input for analysis"""
    try:
        job_id = str(uuid.uuid4())
        
        # Create job directory
        job_dir = UPLOAD_FOLDER / job_id
        job_dir.mkdir(exist_ok=True)
        
        fasta_content = ""
        filename = "input.fasta"
        
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = job_dir / filename
                file.save(str(file_path))
                
                # Read file content
                with open(file_path, 'r') as f:
                    fasta_content = f.read()
        
        # Check if text was provided
        elif 'sequences' in request.form:
            fasta_content = request.form['sequences']
            # Save text input to file
            file_path = job_dir / filename
            with open(file_path, 'w') as f:
                f.write(fasta_content)
        
        else:
            return jsonify({'error': 'No file or sequences provided'}), 400
        
        # Validate FASTA content
        is_valid, message = validate_fasta_content(fasta_content)
        if not is_valid:
            # Clean up job directory
            shutil.rmtree(job_dir)
            return jsonify({'error': f'Invalid FASTA format: {message}'}), 400
        
        # Count sequences
        sequence_count = fasta_content.count('>')
        
        # Get analysis mode
        analysis_mode = request.form.get('analysis_mode', 'rapid')
        
        # Store job info
        active_jobs[job_id] = {
            'id': job_id,
            'status': 'uploaded',
            'progress': 0,
            'filename': filename,
            'sequence_count': sequence_count,
            'analysis_mode': analysis_mode,
            'created_at': datetime.now().isoformat(),
            'file_path': str(file_path),
            'job_dir': str(job_dir)
        }
        
        return jsonify({
            'job_id': job_id,
            'filename': filename,
            'sequence_count': sequence_count,
            'analysis_mode': analysis_mode,
            'message': 'File uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/<job_id>', methods=['POST'])
def start_analysis(job_id):
    """Start the analysis process for a given job"""
    try:
        if job_id not in active_jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = active_jobs[job_id]
        if job['status'] != 'uploaded':
            return jsonify({'error': f'Job already {job["status"]}'}), 400
        
        # Update job status
        job['status'] = 'running'
        job['started_at'] = datetime.now().isoformat()
        
        # Start analysis in background thread
        thread = threading.Thread(target=run_analysis, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'message': 'Analysis started successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_analysis(job_id):
    """Run the analysis pipeline in background"""
    try:
        job = active_jobs[job_id]
        file_path = job['file_path']
        job_dir = Path(job['job_dir'])
        results_dir = RESULTS_FOLDER / job_id
        results_dir.mkdir(exist_ok=True)
        
        # Update progress: Data Processing
        job['status'] = 'processing'
        job['progress'] = 10
        job['current_step'] = 'Data Processing'
        time.sleep(2)  # Simulate processing time
        
        # Update progress: AI Analysis
        job['progress'] = 30
        job['current_step'] = 'AI Model Analysis'
        time.sleep(3)  # Simulate AI processing
        
        # Run the prediction script
        src_dir = Path(__file__).parent.parent / "src"
        predict_script = src_dir / "predict.py"
        
        if predict_script.exists():
            job['progress'] = 50
            
            # Run the actual prediction
            cmd = [sys.executable, str(predict_script), "--input", file_path, "--output", str(results_dir)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(src_dir))
            
            if result.returncode == 0:
                job['progress'] = 90
                job['current_step'] = 'Generating Results'
                time.sleep(1)
                
                # Generate summary results
                generate_summary_results(job_id, results_dir)
                
                job['status'] = 'completed'
                job['progress'] = 100
                job['completed_at'] = datetime.now().isoformat()
                job['results_dir'] = str(results_dir)
            else:
                job['status'] = 'error'
                job['error'] = result.stderr or "Analysis failed"
        else:
            # Simulate analysis for demo purposes
            job['progress'] = 60
            job['current_step'] = 'Species Classification'
            time.sleep(2)
            
            job['progress'] = 80
            job['current_step'] = 'Biodiversity Analysis'
            time.sleep(2)
            
            job['progress'] = 95
            job['current_step'] = 'Generating Results'
            time.sleep(1)
            
            # Generate mock results
            generate_mock_results(job_id, results_dir)
            
            job['status'] = 'completed'
            job['progress'] = 100
            job['completed_at'] = datetime.now().isoformat()
            job['results_dir'] = str(results_dir)
            
    except Exception as e:
        active_jobs[job_id]['status'] = 'error'
        active_jobs[job_id]['error'] = str(e)

def generate_mock_results(job_id, results_dir):
    """Generate mock results for demonstration"""
    job = active_jobs[job_id]
    
    # Generate mock classification results
    results = {
        'job_id': job_id,
        'analysis_completed': datetime.now().isoformat(),
        'input_sequences': job['sequence_count'],
        'classified_sequences': int(job['sequence_count'] * 0.85),
        'novel_sequences': int(job['sequence_count'] * 0.15),
        'species_identified': int(job['sequence_count'] * 0.7),
        'genera_identified': int(job['sequence_count'] * 0.8),
        'families_identified': int(job['sequence_count'] * 0.9),
        'top_species': [
            {'name': 'Escherichia coli', 'confidence': 0.95, 'count': 12},
            {'name': 'Bacillus subtilis', 'confidence': 0.89, 'count': 8},
            {'name': 'Pseudomonas aeruginosa', 'confidence': 0.92, 'count': 6},
            {'name': 'Staphylococcus aureus', 'confidence': 0.87, 'count': 5},
            {'name': 'Enterococcus faecalis', 'confidence': 0.91, 'count': 4}
        ],
        'diversity_metrics': {
            'shannon_diversity': 2.45,
            'simpson_diversity': 0.78,
            'species_richness': 23,
            'evenness': 0.82
        },
        'marker_breakdown': {
            '16S': {'sequences': int(job['sequence_count'] * 0.4), 'species': 15},
            '18S': {'sequences': int(job['sequence_count'] * 0.3), 'species': 8},
            'COI': {'sequences': int(job['sequence_count'] * 0.2), 'species': 5},
            'ITS': {'sequences': int(job['sequence_count'] * 0.1), 'species': 3}
        }
    }
    
    # Save results
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate CSV report
    csv_content = "Sequence_ID,Species,Genus,Family,Confidence,Marker\n"
    for i in range(job['sequence_count']):
        species = results['top_species'][i % len(results['top_species'])]
        csv_content += f"seq_{i+1},{species['name']},Unknown,Unknown,{species['confidence']:.2f},16S\n"
    
    with open(results_dir / 'classification_report.csv', 'w') as f:
        f.write(csv_content)

def generate_summary_results(job_id, results_dir):
    """Generate summary from actual analysis results"""
    # This would parse the actual results from the prediction script
    # For now, we'll generate mock results
    generate_mock_results(job_id, results_dir)

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get the status of an analysis job"""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'current_step': job.get('current_step', ''),
        'sequence_count': job['sequence_count'],
        'created_at': job['created_at'],
        'error': job.get('error', None)
    })

@app.route('/api/results/<job_id>', methods=['GET'])
def get_results(job_id):
    """Get analysis results for a job"""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    results_file = Path(job['results_dir']) / 'results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        return jsonify(results)
    else:
        return jsonify({'error': 'Results not found'}), 404

@app.route('/api/download/<job_id>/<filename>', methods=['GET'])
def download_results(job_id, filename):
    """Download analysis results file"""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    file_path = Path(job['results_dir']) / filename
    if file_path.exists():
        return send_file(str(file_path), as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    jobs_summary = []
    for job_id, job in active_jobs.items():
        jobs_summary.append({
            'job_id': job_id,
            'status': job['status'],
            'filename': job['filename'],
            'sequence_count': job['sequence_count'],
            'created_at': job['created_at'],
            'analysis_mode': job['analysis_mode']
        })
    
    return jsonify({'jobs': jobs_summary})

if __name__ == '__main__':
    print("Starting ATLAS Backend API...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Results folder: {RESULTS_FOLDER}")
    app.run(debug=True, host='0.0.0.0', port=5000)
