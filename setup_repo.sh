#!/bin/bash

# Setup script for Resume Screening LLM Repository
echo "Setting up Resume Screening LLM Repository..."
echo "============================================="

# Initialize git repository
echo "1. Initializing Git repository..."
git init
git add .
git commit -m "Initial commit: Resume Screening LLM for forensic investigation

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Create initial directories
echo "2. Creating necessary directories..."
mkdir -p models experiments forensic_reports logs

# Generate initial synthetic data
echo "3. Generating initial synthetic data..."
python examples/quick_start.py

echo ""
echo "============================================="
echo "✅ Repository setup complete!"
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Train the model: python -m src.train --evaluate"
echo "3. Try the CLI: python -m src.cli.cli --help"
echo "4. Start the API: python -m src.api.app"
echo "5. Run forensic analysis: python examples/forensic_analysis.py"
echo ""
echo "For forensic investigation:"
echo "- Review src/model/resume_llm.py for model implementation"
echo "- Check examples/forensic_analysis.py for bias detection tools"
echo "- Examine data/synthetic/ for training data patterns"
echo "- Use the API endpoints to test different scenarios"
echo ""
echo "Repository ready for cloning and analysis! 🔍"