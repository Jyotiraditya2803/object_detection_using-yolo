#!/bin/bash
echo "========================================================"
echo " Launching YOLO Real-Time Detection System (Linux/Mac)"
echo "========================================================"

if [ ! -d "venv" ]; then
    echo "[INFO] Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[INFO] Checking and installing requirements..."
pip install -r requirements.txt

echo "[INFO] Starting main detection pipeline..."
python main.py
