import http.server
import socketserver
import json
import cgi
import os
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))
from src.predict_refactored import run_analysis

# --- Configuration ---
PORT = 8000
ROOT_DIR = Path(__file__).parent
HTML_FILE_PATH = ROOT_DIR / 'index.html'
TEMP_DIR = ROOT_DIR / 'temp'
os.makedirs(TEMP_DIR, exist_ok=True)

# Define the custom handler class that extends the standard HTTP server
class MyHandler(http.server.SimpleHTTPRequestHandler):
    
    def do_GET(self):
        """Handle GET requests to serve the main HTML file."""
        if self.path == '/':
            self.path = '/index.html'
        
        if self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open(HTML_FILE_PATH, 'rb') as f:
                self.wfile.write(f.read())
            return
        
        # This allows the server to correctly serve assets like Chart.js
        return super().do_GET()

    def do_POST(self):
        """Handle POST requests for running analysis."""
        if self.path == '/run_analysis':
            content_type = self.headers.get('Content-Type')
            if 'multipart/form-data' in content_type:
                self.handle_multipart_post()
            else:
                self.send_error(415, "Unsupported Media Type")
        else:
            self.send_error(404, "Not Found")

    def handle_multipart_post(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.headers['Content-Type']}
        )
        
        input_data = form.getvalue('data', '')
        fasta_file_data = form.getvalue('file')
        
        temp_fasta_path = None
        analysis_result = {"status": "error", "message": "No valid input provided."}
        
        try:
            if fasta_file_data and isinstance(fasta_file_data, bytes):
                temp_fasta_path = TEMP_DIR / 'temp_upload.fasta'
                with open(temp_fasta_path, 'wb') as f:
                    f.write(fasta_file_data)
                analysis_result = run_analysis(str(temp_fasta_path))
            
            elif input_data:
                temp_fasta_path = TEMP_DIR / 'temp_input.fasta'
                with open(temp_fasta_path, 'w', encoding='utf-8') as f:
                    f.write(input_data)
                analysis_result = run_analysis(str(temp_fasta_path))

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(analysis_result).encode('utf-8'))
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
        finally:
            if temp_fasta_path and os.path.exists(temp_fasta_path):
                os.remove(temp_fasta_path)

# --- Start the Server ---
if __name__ == "__main__":
    # Correct pathing to handle file serving from the root directory
    os.chdir(ROOT_DIR) 
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"ATLAS server started at http://localhost:{PORT}")
        httpd.serve_forever()
