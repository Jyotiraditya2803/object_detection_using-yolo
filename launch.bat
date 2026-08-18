 @echo off
echo ========================================================
echo  Launching YOLO Real-Time Detection System (Windows)
echo ========================================================

IF NOT EXIST "venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Checking and installing requirements...
pip install -r requirements.txt

echo [INFO] Starting main detection pipeline...
python main.py

pause
