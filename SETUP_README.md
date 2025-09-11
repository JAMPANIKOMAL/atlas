# ATLAS - Environmental DNA Analysis Platform

ATLAS is a revolutionary environmental DNA analysis platform that maps biodiversity, discovers new species, and monitors ecosystem health with unprecedented precision using AI-driven models.

## 🚀 Quick Start

### Prerequisites
- Anaconda or Miniconda installed
- Conda environments set up (see docs/02_Environment_and_Installation.md)
- Modern web browser

### Installation & Setup

#### Step 1: Create Conda Environment
First, create the appropriate conda environment based on your system:

**For GPU systems (recommended):**
```bash
conda env create -f environment.yml
```

**For CPU-only systems:**
```bash
conda env create -f environment-cpu.yml
```

#### Step 2: Start ATLAS

**Windows:**
1. Double-click `start_atlas.bat`
2. The script will automatically install Flask dependencies and start the system
3. Your browser will open the ATLAS frontend

**Linux/macOS:**
1. Make the script executable: `chmod +x start_atlas.sh`
2. Run: `./start_atlas.sh`
3. Open `frontend/index.html` in your web browser

#### Manual Setup
```bash
# Activate your conda environment
conda activate atlas  # or atlas-cpu

# Install Flask dependencies
pip install -r backend/requirements.txt

# Start the backend server
python backend/app.py

# Open frontend/index.html in your browser
```

## 💻 Usage

### 1. Access the Platform
- Open your web browser and navigate to the ATLAS homepage
- Click "Start Analysis" to go to the analysis page

### 2. Upload Your Data
You can provide data in two ways:
- **File Upload**: Upload FASTA files (.fasta, .fa, .fas, .txt)
- **Text Input**: Paste FASTA sequences directly

### 3. Choose Analysis Mode
- **Rapid Analysis**: Quick species identification
- **Comprehensive**: Detailed biodiversity analysis  
- **Custom**: Advanced parameters

### 4. Start Analysis
- Click "Start Analysis" button
- Monitor real-time progress through the pipeline
- View results when analysis completes

### 5. Download Results
- Download CSV classification report
- Download JSON results file
- View biodiversity metrics and species identification

## 🔬 Analysis Pipeline

### Filter Models (AI Classification)
- **16S rRNA**: Bacterial and archaeal identification
- **18S rRNA**: Eukaryotic microorganism identification
- **COI**: Metazoan species identification
- **ITS**: Fungal species identification

### Explorer Pipeline (Novel Discovery)
For sequences that cannot be classified:
1. **Vectorize**: Convert sequences to numerical representations
2. **Cluster**: Group similar unclassified sequences
3. **Interpret**: Provide insights about potential novel species

## 📊 Results

### Classification Report
- Species, genus, and family assignments
- Confidence scores for each prediction
- Marker gene breakdown (16S, 18S, COI, ITS)

### Biodiversity Metrics
- Shannon diversity index
- Simpson diversity index
- Species richness
- Evenness measures

### Novel Sequences
- Unclassified sequences grouped into clusters
- Potential new species discoveries
- Ecological insights

## 🛠 API Endpoints

The backend provides RESTful API endpoints:

- `GET /api/health` - Health check
- `POST /api/upload` - Upload data for analysis
- `POST /api/analyze/<job_id>` - Start analysis
- `GET /api/status/<job_id>` - Check analysis status
- `GET /api/results/<job_id>` - Get analysis results
- `GET /api/download/<job_id>/<filename>` - Download result files

## 🔧 Configuration

### Backend Configuration
The backend runs on `http://localhost:5000` by default. To change:
1. Edit `backend/app.py`
2. Modify the `app.run()` parameters
3. Update frontend API URLs in `frontend/js/analysis.js`

### Model Paths
Models are expected in the `models/` directory:
- `16s_genus_classifier.keras`
- `18s_genus_classifier.keras`
- `its_genus_classifier.keras`
- Label encoders and vectorizers (`.pkl` files)

## 🔒 Data Security

- Files are processed locally on your machine
- No data is sent to external servers
- Temporary files are cleaned up after processing
- Analysis jobs are stored in memory only

## 🐛 Troubleshooting

### Common Issues

1. **Backend won't start**
   - Check Python installation: `python --version`
   - Install dependencies: `pip install -r backend/requirements.txt`
   - Check port 5000 isn't in use

2. **Analysis fails**
   - Ensure FASTA format is correct
   - Check that model files exist in `models/` directory
   - View console logs for detailed error messages

3. **Frontend can't connect to backend**
   - Ensure backend is running on localhost:5000
   - Check browser console for CORS errors
   - Verify firewall isn't blocking connections

### File Format Requirements
- FASTA files must have proper headers (starting with `>`)
- Sequences should contain valid DNA characters (A, T, C, G, N, etc.)
- Files should use UTF-8 encoding

## 🤝 Contributing

This project is part of the Centre for Marine Living Resources and Ecology (CMLRE) initiative for deep-sea biodiversity assessment.

## 📄 License

This project is developed for the Ministry of Earth Sciences (MoES) - Centre for Marine Living Resources and Ecology (CMLRE).

---

For technical support or questions, please refer to the documentation in the `docs/` directory.
