#!/bin/bash

# Start the Resume Screening LLM API

echo "Starting Resume Screening LLM API..."

# Get the project root directory
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements-api.txt

# Check if model exists
if [ ! -f "models/resume_llm_latest.pkl" ]; then
    echo "Warning: No trained model found at models/resume_llm_latest.pkl"
    echo "You may need to train a model first using: python src/train.py"
fi

# Start the API
echo "Starting API server on http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"

cd src/api
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload