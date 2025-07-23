#!/bin/bash

# Define the repo directory name
REPO_DIR="yolov8_traffic_light_system"

# Remove existing folder if it exists
if [ -d "$REPO_DIR" ]; then
    echo "Removing existing folder: $REPO_DIR"
    rm -rf "$REPO_DIR"
fi

# Clone the repository
echo "Cloning repository..."
git clone https://github.com/Prakashash18/yolov8_traffic_light_system.git

# Change to repo directory
cd "$REPO_DIR" || { echo "Failed to cd into $REPO_DIR"; exit 1; }

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
