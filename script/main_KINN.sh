#!/bin/bash

# === CONFIGURATION ===
PYTHON_FILE="main/main_KINN.py"
LOG_FILE="log/KINN_solver2.log"
VENV_DIR=".venv"

# === CLEAN OLD FILES ===
rm -rf __pycache__
rm -f "$LOG_FILE"


# === CHECK VENV ===
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment '$VENV_DIR' not found!" | tee -a "$LOG_FILE"
    exit 1
fi

# === ACTIVATE VENV ===
source "$VENV_DIR/bin/activate"

# === GET USER INPUT ===
read -p "Enter Seed (default: 42): " SEED
read -p "Enter Dropout Rate (default: 0.000): " DROPOUT
read -p "Enter Number of Hidden Layer (default: 6): " HIDDEN
read -p "Enter Activation (default: 'tanh'): " ACTIVATION
read -p "Enter Number of Neurons (default: 64): " NEURONS
read -p "Enter Learning Rate (default: 0.001): " LR
read -p "Enter Weight Decay (default: 0.0001): " WD
read -p "Enter Epochs (default: 10000): " EPOCHS
# read -p "Enter Batch Size (default: 32): " BATCH
read -p "Enter Omega eq (default: 1.0): " O1
read -p "Enter Omega bc (default: 1.0): " O2
read -p "Enter Omega dt (default: 1.0): " O3

# === DEFAULT VALUES IF EMPTY ===
SEED=${SEED:-42}
DROPOUT=${DROPOUT:-0.000}
HIDDEN=${HIDDEN:-6}
ACTIVATION=${ACTIVATION:-'tanh'}
NEURONS=${NEURONS:-64}
LR=${LR:-0.001}
WD=${WD:-0.0001}
EPOCHS=${EPOCHS:-10000}
# BATCH=${BATCH:-32}
O1=${O1:-1.0}
O2=${O2:-1.0}
O3=${O3:-1.0}

# === START LOGGING ===
echo "🚀 Running $PYTHON_FILE with parameters:" | tee "$LOG_FILE"
echo "Seed = $SEED" | tee -a "$LOG_FILE"
echo "Dropout = $DROPOUT" | tee -a "$LOG_FILE"
echo "Hidden Layer = $HIDDEN" | tee -a "$LOG_FILE"
echo "Activation = $ACTIVATION" | tee -a "$LOG_FILE"
echo "Neurons = $NEURONS" | tee -a "$LOG_FILE"
echo "Learning Rate = $LR" | tee -a "$LOG_FILE"
echo "Weight Decay = $WD" | tee -a "$LOG_FILE"
echo "Epochs = $EPOCHS" | tee -a "$LOG_FILE"
# echo "Batch Size = $BATCH" | tee -a "$LOG_FILE"
echo "omega_eq = $O1, omega_bc = $O2, omega_dt = $O3" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# === RUN SCRIPT & TIME EXECUTION ===
start_time=$(date +%s)

python3 "$PYTHON_FILE" \
    --seed "$SEED" \
    --dropout_rate "$DROPOUT" \
    --num_neurons "$NEURONS" \
    --learning_rate "$LR" \
    --num_hidden_layers "$HIDDEN" \
    --activation "$ACTIVATION" \
    --weight_decay "$WD" \
    --num_epochs "$EPOCHS" \
    --omega_eq "$O1" \
    --omega_bc "$O2" \
    --omega_dt "$O3" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

# === EXIT STATUS ===
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Script failed (code $EXIT_CODE). See $LOG_FILE for details." | tee -a "$LOG_FILE"
    deactivate
    exit $EXIT_CODE
else
    echo "✅ Execution completed successfully in ${duration}s." | tee -a "$LOG_FILE"
fi

# === DEACTIVATE VENV ===
deactivate
