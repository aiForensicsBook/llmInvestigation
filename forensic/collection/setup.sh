#!/bin/bash
# Forensic Collection Framework Setup Script
# Run this script to install dependencies and test the framework

set -e

echo "======================================================"
echo "Forensic Collection Framework Setup"
echo "======================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "✓ Python version check passed: $python_version"

# Install system dependencies
echo ""
echo "Installing system dependencies..."

if command -v apt-get &> /dev/null; then
    echo "Detected Debian/Ubuntu system"
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-dev libmagic1 python3-magic
elif command -v yum &> /dev/null; then
    echo "Detected RedHat/CentOS system"
    sudo yum install -y python3-pip python3-devel file-devel
elif command -v brew &> /dev/null; then
    echo "Detected macOS system"
    brew install libmagic
else
    echo "⚠️  Unknown system. Please install libmagic manually."
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."

if [ -f "requirements.txt" ]; then
    pip3 install --user -r requirements.txt
    echo "✓ Python dependencies installed"
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Make scripts executable
echo ""
echo "Setting up scripts..."
chmod +x forensic_collector.py
chmod +x example_usage.py
chmod +x test_framework.py

# Run basic tests
echo ""
echo "Running framework tests..."
python3 test_framework.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Review the README.md for detailed documentation"
    echo "2. Run the example: python3 example_usage.py --demo"
    echo "3. Start collecting evidence: python3 forensic_collector.py --help"
else
    echo ""
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi