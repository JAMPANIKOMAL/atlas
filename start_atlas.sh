#!/bin/bash

# ATLAS Startup Script (Conda Version)
echo "Starting ATLAS Environmental DNA Analysis Platform..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if atlas environment exists
if ! conda info --envs | grep -q "atlas "; then
    echo "Error: 'atlas' conda environment not found"
    echo "Please create the environment first with:"
    echo "  conda env create -f environment.yml"
    echo "OR for CPU-only:"
    echo "  conda env create -f environment-cpu.yml"
    exit 1
fi

# Install Flask dependencies if not already installed
echo "Checking Flask dependencies in atlas environment..."
conda run -n atlas pip install -q -r backend/requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install Flask dependencies"
    exit 1
fi

# Start Flask backend using conda environment in background
echo "Starting backend server with conda environment 'atlas'..."
conda run -n atlas python backend/app.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Check if backend is running
if ps -p $BACKEND_PID > /dev/null; then
    echo "Backend server started successfully (PID: $BACKEND_PID)"
else
    echo "Error: Failed to start backend server"
    exit 1
fi

echo ""
echo "🧬 ATLAS is now running!"
echo ""
echo "Frontend: Open 'frontend/index.html' in your web browser"
echo "Backend API: http://localhost:5000"
echo ""
echo "Note: Using conda environment 'atlas'"
echo ""
echo "To stop the backend server, run: kill $BACKEND_PID"
echo "PID saved to: atlas_backend.pid"

# Save PID for easy stopping
echo $BACKEND_PID > atlas_backend.pid

# Keep script running
wait $BACKEND_PID
