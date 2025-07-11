#!/bin/bash

# Forensic Dashboard Startup Script
# ================================

echo "Starting Forensic Dashboard..."

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not installed."
    exit 1
fi

# Check if Streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Streamlit not found. Installing requirements..."
    python3 setup.py
fi

# Set environment variables
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Development vs Production settings
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Starting in PRODUCTION mode..."
    export STREAMLIT_SERVER_ENABLE_CORS=false
    export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
else
    echo "Starting in DEVELOPMENT mode..."
    export DEBUG=true
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the dashboard
echo "Dashboard will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Start Streamlit
python3 -m streamlit run main.py \
    --server.port=$STREAMLIT_SERVER_PORT \
    --server.address=$STREAMLIT_SERVER_ADDRESS \
    --server.headless=$STREAMLIT_SERVER_HEADLESS \
    --browser.gatherUsageStats=$STREAMLIT_BROWSER_GATHER_USAGE_STATS