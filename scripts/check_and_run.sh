#!/bin/zsh

# Check if data directory exists in the project folder
if [ ! -d "../Data" ]; then
  echo "Error: 'data' directory does not exist."
  exit 1
fi

# Check if metadata.csv file exists in the data directory
if [ ! -f "../Data/metadata.csv" ]; then
  echo "Error: 'metadata.csv' file does not exist in the 'data' directory."
  exit 1
fi

# Check if wavs directory exists
if [ ! -d "../Data/wavs" ]; then
  echo "Error: 'data/wavs' directory does not exist."
  exit 1
fi

# If all checks pass, run model.py
echo "All required files and directories are in place. Running model.py..."
python3 project_x_voice/model.py

