#!/bin/bash

# === CONFIGURATION ===
PYTHON_FILE="main_K_R.py"
LOG_FILE="log/SOL_sovler.log"
VENV_DIR=".venv"

# === CLEAN OLD FILES ===
rm -rf __pycache__
rm -f "$LOG_FILE"

# === CHECK VENV ===
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' not found!" | tee -a "$LOG_FILE"
    exit 1
fi

# === ACTIVATE VENV ===
source "$VENV_DIR/bin/activate"

# === CHECK PYTHON ===
if ! command -v python3 &> /dev/null; then
    echo "python3 not found in virtual environment!" | tee -a "$LOG_FILE"
    deactivate
    exit 1
fi

# === CHECK PYTHON FILE ===
if [ ! -f "$PYTHON_FILE" ]; then
    echo "Python file '$PYTHON_FILE' not found!" | tee -a "$LOG_FILE"
    deactivate
    exit 1
fi

# === GET USER INPUT WITH DEFAULTS ===
read -p "Enter method [euler, improved_euler, rk2, rk4] (default: rk4): " METHOD

# === ASSIGN DEFAULT IF EMPTY ===
METHOD=${METHOD:-rk4}

# === START LOGGING ===
{
echo "Running $PYTHON_FILE with method: $METHOD"
echo "----------------------------------------"
} | tee "$LOG_FILE"

# === RUN SCRIPT & TIME EXECUTION ===
start_time=$(date +%s)

python3 "$PYTHON_FILE" --method "$METHOD" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
end_time=$(date +%s)
duration=$((end_time - start_time))

# === EXIT STATUS ===
if [ $EXIT_CODE -ne 0 ]; then
    echo "Script failed (code $EXIT_CODE). See $LOG_FILE for details." | tee -a "$LOG_FILE"
    deactivate
    exit $EXIT_CODE
else
    echo "Execution completed successfully in ${duration}s." | tee -a "$LOG_FILE"
fi

# === DEACTIVATE VENV ===
deactivate