#!/usr/bin/env python3
"""
ATLAS Environment Test Script
This script verifies that all required dependencies are available in the conda environment.
"""

import sys
from pathlib import Path

def test_import(module_name, description=""):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✓ {module_name} {description}")
        return True
    except ImportError as e:
        print(f"✗ {module_name} {description} - Error: {e}")
        return False

def test_models_directory():
    """Check if models directory exists and has required files"""
    models_dir = Path(__file__).parent.parent / "models"
    
    if not models_dir.exists():
        print("✗ Models directory not found")
        return False
    
    required_models = [
        "16s_genus_classifier.keras",
        "18s_genus_classifier.keras", 
        "its_genus_classifier.keras",
        "16s_genus_label_encoder.pkl",
        "18s_genus_label_encoder.pkl",
        "its_genus_label_encoder.pkl",
        "16s_genus_vectorizer.pkl",
        "18s_genus_vectorizer.pkl", 
        "its_genus_vectorizer.pkl"
    ]
    
    print("\nChecking model files:")
    all_present = True
    for model_file in required_models:
        if (models_dir / model_file).exists():
            print(f"✓ {model_file}")
        else:
            print(f"✗ {model_file} (missing)")
            all_present = False
    
    return all_present

def main():
    print("ATLAS Environment Test")
    print("=" * 50)
    
    # Test core scientific packages (should be in conda environment)
    core_packages = [
        ("pandas", "- Data manipulation"),
        ("numpy", "- Numerical computing"),
        ("scipy", "- Scientific computing"),
        ("sklearn", "- Machine learning (scikit-learn)"),
        ("Bio", "- Bioinformatics (biopython)"),
        ("tensorflow", "- Deep learning framework"),
        ("tqdm", "- Progress bars")
    ]
    
    print("\nCore Scientific Packages:")
    core_success = all(test_import(pkg, desc) for pkg, desc in core_packages)
    
    # Test Flask packages (installed separately)
    flask_packages = [
        ("flask", "- Web framework"),
        ("flask_cors", "- CORS support")
    ]
    
    print("\nFlask Web Framework:")
    flask_success = all(test_import(pkg, desc) for pkg, desc in flask_packages)
    
    # Test model files
    models_success = test_models_directory()
    
    print("\n" + "=" * 50)
    
    if core_success and flask_success:
        print("✓ All required packages are available!")
        print("✓ ATLAS backend is ready to run")
        
        if not models_success:
            print("\n⚠ Warning: Some model files are missing")
            print("The system will use demo mode for missing models")
        
        return 0
    else:
        print("✗ Some required packages are missing")
        
        if not core_success:
            print("\nTo install core packages, run:")
            print("  conda env create -f environment.yml")
            print("  OR")
            print("  conda env create -f environment-cpu.yml")
        
        if not flask_success:
            print("\nTo install Flask packages, run:")
            print("  conda run -n atlas pip install -r backend/requirements.txt")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
