#!/bin/bash

echo "================================================"
echo "ANPR System Setup"
echo "================================================"

# Check Python version
echo "Checking Python version..."
python3 --version

# Check if pip is installed
echo "Checking pip..."
python3 -m pip --version

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check Tesseract installation
echo ""
echo "Checking Tesseract OCR installation..."
if command -v tesseract &> /dev/null
then
    echo "✓ Tesseract is installed"
    tesseract --version
else
    echo "✗ Tesseract is NOT installed"
    echo ""
    echo "Please install Tesseract OCR:"
    echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr"
    echo "  macOS: brew install tesseract"
    echo "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
fi

# Test the system
echo ""
read -p "Run system tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Running tests..."
    python3 test_system.py
fi

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To run the ANPR system:"
echo "  python src/main.py"
echo ""
echo "For external USB camera:"
echo "  python src/main.py --camera 1"
echo ""
echo "Controls during operation:"
echo "  'q' - Quit"
echo "  's' - Save current frame"
echo "  'r' - Reset confirmation"
echo ""