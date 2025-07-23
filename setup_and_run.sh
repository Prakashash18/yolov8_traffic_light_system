#!/bin/bash

# Create Python virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
# For Linux/macOS
source venv/bin/activate

# Install required packages
echo "Installing packages..."
pip install flask flask_cors ultralytics pillow opencv-python

# Change into webapp directory
cd webapp || { echo "Failed to cd into webapp"; exit 1; }

# Run the app
echo "Running app..."
python app_for_runpod.py
