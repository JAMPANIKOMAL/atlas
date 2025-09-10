# ATLAS: Artificial Taxonomic Learning & Analysis System

This repository contains the official codebase for the ATLAS project, an AI-driven software suite for taxonomic identification and biodiversity assessment from environmental DNA (eDNA).

## Project Mission

The goal of ATLAS is to address a critical challenge in modern biodiversity research: the "database gap." Standard reference databases are often incomplete, especially for organisms from unique biomes like the deep sea. ATLAS is an AI-driven pipeline that minimizes reliance on these databases, reduces computational time, and enables the discovery of novel taxa from raw eDNA reads.

For a detailed overview of the project's scientific background and long-term goals, please see the [Project Overview](docs/01_Project_Overview.md) document.

## Getting Started

To get started with ATLAS, you will need to set up a Conda environment and download the required reference databases.

### 1. Environment Setup

This project supports both GPU-accelerated and CPU-only workflows. Please follow the [Environment and Installation Guide](docs/02_Environment_and_Installation.md) for detailed, step-by-step instructions on setting up the correct environment for your system.

### 2. Data Acquisition

The ATLAS pipelines rely on large, public reference databases that are not included in this repository. You must download them manually. The required files are listed in the [16S Pipeline Workflow](docs/03_Pipeline_16S_Workflow.md). Place all downloaded files in the `data/raw/` directory.

### 3. Running a Pipeline

Once your environment is configured and the data is in place, you can run any of the processing pipelines. Each pipeline consists of a two-step script-based workflow. For example, to run the 16S pipeline:

```bash
# First, run the data preparation script
python src/pipeline_16s/01_prepare_data.py

# Then, run the model training script
python src/pipeline_16s/02_train_model.py
```

For detailed workflow instructions for each pipeline, please refer to the documentation in the [docs](docs/) directory.

## Project Structure

The project is organized into a clean, modular structure:

- **`/data/`**: Holds the raw and processed datasets.
- **`/docs/`**: Contains all project documentation, development logs, and guides.
- **`/models/`**: Stores the final trained models (`.keras`) and data encoders (`.pkl`).
- **`/notebooks/`**: Contains the Jupyter Notebooks used for the interactive development and refinement of each pipeline.
- **`/src/`**: Contains the final, production-ready Python (`.py`) scripts for each pipeline.