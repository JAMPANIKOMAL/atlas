# ATLAS: Artificial Taxonomic Learning & Analysis System

This repository contains the official codebase for the ATLAS project, an AI-driven software suite for taxonomic identification and biodiversity assessment from environmental DNA (eDNA).

## About This Repository
This repository implements an AI-driven pipeline that minimizes reliance on reference databases, reduces computational time, and enables the discovery of novel taxa and ecological insights in deep-sea environments.

## Project Mission

The goal of ATLAS is to address a critical challenge in modern biodiversity research: the "database gap." Standard reference databases are often incomplete, especially for organisms from unique biomes like the deep sea. ATLAS is an AI-driven pipeline that minimizes reliance on these databases, reduces computational time, and enables the discovery of novel taxa from raw eDNA reads.

For a detailed overview of the project's scientific background and long-term goals, please see the [Project Overview](docs/01_Project_Overview.md) document.

## Problem Statement & Context

**Problem Statement ID:** ID25042  
**Problem Statement Title:** Identifying Taxonomy and Assessing Biodiversity from eDNA Datasets

### Description

The deep ocean, encompassing vast and remote ecosystems like abyssal plains, hydrothermal vents, and seamounts, harbors a significant portion of global biodiversity, much of which remains undiscovered due to its inaccessibility. Understanding deep-sea biodiversity is critical for elucidating ecological interactions (e.g., food webs, nutrient cycling), informing conservation strategies for vulnerable marine habitats, and identifying novel eukaryotic species with potential biotechnological or ecological significance.

Environmental DNA (eDNA) has emerged as a powerful, non-invasive tool for studying these ecosystems by capturing genetic traces of organisms from environmental samples, such as seawater or sediment, without the need for physical collection or disturbance of fragile habitats. By targeting marker genes like 18S rRNA or COI, eDNA enables the detection of diverse eukaryotic taxa, including protists, cnidarians, and rare metazoans, offering insights into species richness and community structure.

The Centre for Marine Living Resources and Ecology (CMLRE) will undertake routine voyages to the deep sea and collect sediment and water samples from hotspot regions for biodiversity assessment and ecosystem monitoring. The water and sediment samples will be used to extract eDNA and will be subject to high-throughput sequencing.

However, assigning raw eDNA sequencing reads to eukaryotic taxa or inferring their ecological roles presents significant challenges, primarily due to the poor representation of deep-sea organisms in reference databases like SILVA, PR2, or NCBI. These databases, built primarily from well-studied terrestrial or shallow-water species, lack comprehensive sequences for deep-sea eukaryotes, leading to misclassifications, unassigned reads, or underestimation of biodiversity.

Traditional bioinformatic pipelines for eDNA analysis, such as those implemented in QIIME2, DADA2, or mothur, rely heavily on sequence alignment or mapping to these databases, which is inadequate for novel or divergent deep-sea taxa. This dependency limits the discovery of new species and hinders accurate biodiversity assessments, critical for conservation in rapidly changing deep-sea environments. The computational time required for processing eDNA data exacerbates these challenges, particularly given the limitations of database-dependent methods and the complexity of eDNA datasets.

### Expected Solution

To address the challenges of poor database representation and computational time in deep-sea eDNA analysis, we propose an AI-driven pipeline that uses deep learning and unsupervised learning to identify eukaryotic taxa and assess biodiversity directly from raw eDNA reads. The solution should be able to classify the sequences, annotate and estimate abundance. This solution minimizes reliance on reference databases, reduces computational time through optimized workflows, and enables the discovery of novel taxa and ecological insights in deep-sea ecosystems.

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

## Documentation

The project includes comprehensive documentation in the `docs/` directory:

1. [Project Overview](docs/01_Project_Overview.md) - Scientific background and project goals
2. [Environment and Installation Guide](docs/02_Environment_and_Installation.md) - Setup instructions
3. [16S Pipeline Workflow](docs/03_Pipeline_16S_Workflow.md) - 16S rRNA gene analysis pipeline
4. [18S Pipeline Workflow](docs/04_Pipeline_18S_Workflow.md) - 18S rRNA gene analysis pipeline
5. [COI Pipeline Workflow](docs/05_Pipeline_COI_Workflow.md) - Cytochrome c oxidase subunit I analysis pipeline
6. [Performance Evaluation](docs/06_Performance_Evaluation.md) - Model performance metrics and benchmarks
7. [Troubleshooting and FAQ](docs/07_Troubleshooting_and_FAQ.md) - Common issues and solutions

## Project Structure

The project is organized into a clean, modular structure:

- **`/data/`**: Holds the raw and processed datasets.
- **`/docs/`**: Contains all project documentation, development logs, and guides.
- **`/models/`**: Stores the final trained models (`.keras`) and data encoders (`.pkl`).
- **`/notebooks/`**: Contains the Jupyter Notebooks used for the interactive development and refinement of each pipeline.
- **`/src/`**: Contains the final, production-ready Python (`.py`) scripts for each pipeline.

