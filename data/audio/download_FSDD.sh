#!/bin/bash
# Download Free Spoken Digit Dataset (FSDD) into current dir (data/audio)

# If dataset already exists
if ls recordings/*.wav 1> /dev/null 2>&1; then
    echo "[INFO] FSDD already exists in $(pwd)"
else
    echo "[INFO] Downloading FSDD into $(pwd) ..."
    curl -L -o fsdd.zip https://www.kaggle.com/api/v1/datasets/download/joserzapata/free-spoken-digit-dataset-fsdd
    unzip -o fsdd.zip
    rm -rf fsdd.zip metadata.py pip_requirements.txt .gitignore __init__.py utils acquire_data README.md
    echo "[INFO] Download complete."
fi
