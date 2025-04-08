#!/bin/bash

# === CONFIGURATION ===
PYTHON_FILE="main_PINN.py"
LOG_FILE="log/PINN_solver.log"
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

# === GET USER INPUT ===
read -p "Enter Seed (default: 42): " SEED
read -p "Enter Dropout Rate (default: 0.0001): " DROPOUT
read -p "Enter Number of Neurons (default: 256): " NEURONS
read -p "Enter Learning Rate (default: 0.001): " LR
read -p "Enter Weight Decay (default: 0.0001): " WD
read -p "Enter Epochs (default: 3000): " EPOCHS
read -p "Enter Batch Size (default: 32): " BATCH
read -p "Enter Omega eq (default: 1.0): " O1
read -p "Enter Omega bc (default: 1.0): " O2
read -p "Enter Omega dt (default: 1.0): " O3

# === DEFAULT VALUES IF EMPTY ===
SEED=${SEED:-42}
DROPOUT=${DROPOUT:-0.0001}
NEURONS=${NEURONS:-256}
LR=${LR:-0.001}
WD=${WD:-0.0001}
EPOCHS=${EPOCHS:-3000}
BATCH=${BATCH:-32}
O1=${O1:-1.0}
O2=${O2:-1.0}
O3=${O3:-1.0}

# === START LOGGING ===
echo "Running $PYTHON_FILE with parameters:" | tee "$LOG_FILE"
echo "Seed = $SEED" | tee -a "$LOG_FILE"
echo "Dropout = $DROPOUT" | tee -a "$LOG_FILE"
echo "Neurons = $NEURONS" | tee -a "$LOG_FILE"
echo "Learning Rate = $LR" | tee -a "$LOG_FILE"
echo "Weight Decay = $WD" | tee -a "$LOG_FILE"
echo "Epochs = $EPOCHS" | tee -a "$LOG_FILE"
echo "Batch Size = $BATCH" | tee -a "$LOG_FILE"
echo "omega_eq = $O1, omega_bc = $O2, omega_dt = $O3" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# === RUN SCRIPT & TIME EXECUTION ===
start_time=$(date +%s)

stdbuf -oL python3 "$PYTHON_FILE" \
    --seed "$SEED" \
    --dropout_rate "$DROPOUT" \
    --num_neurons "$NEURONS" \
    --learning_rate "$LR" \
    --weight_decay "$WD" \
    --num_epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    --omega_eq "$O1" \
    --omega_bc "$O2" \
    --omega_dt "$O3" 2>&1 | tee -a "$LOG_FILE"


EXIT_CODE=$?
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
