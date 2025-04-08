#!/bin/bash

# === CONFIGURATION ===
PYTHON_FILE="main/main_MLP.py"
LOG_FILE="log/MLP_solver2.log"
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
read -p "Enter Seed (default: 42): " SEED
read -p "Enter Dropout Rate (default: 0.000): " DROPOUT
read -p "Enter Number of Hidden Layers (default: 6): " HIDDEN
read -p "Enter Activation (default: 'tanh'): " ACTIVATION
read -p "Enter Number of Neurons (default: 64): " NEURONS
read -p "Enter Learning Rate (default: 0.001): " LR
read -p "Enter Weight Decay (default: 0.0001): " WD
read -p "Enter Epochs (default: 10000): " EPOCHS
read -p "Enter Omega dt (default: 1.0): " O3

# === ASSIGN DEFAULT IF EMPTY (remove whitespaces) ===
SEED=${SEED:-42}
HIDDEN=${HIDDEN:-6}
ACTIVATION=${ACTIVATION:-tanh}
DROPOUT=${DROPOUT:-0.000}
NEURONS=${NEURONS:-64}
LR=${LR:-0.001}
WD=${WD:-0.0001}
EPOCHS=${EPOCHS:-10000}
O3=${O3:-1.0}

# === START LOGGING ===
{
echo "Running $PYTHON_FILE with parameters:"
echo "Seed           = $SEED"
echo "Dropout        = $DROPOUT"
echo "Hidden Layers  = $HIDDEN"
echo "Activation     = $ACTIVATION"
echo "Neurons        = $NEURONS"
echo "Learning Rate  = $LR"
echo "Weight Decay   = $WD"
echo "Epochs         = $EPOCHS"
echo "omega_dt       = $O3"
echo "----------------------------------------"
} | tee "$LOG_FILE"

# === RUN SCRIPT & TIME EXECUTION ===
start_time=$(date +%s)

python3 "$PYTHON_FILE" \
    --seed "$SEED" \
    --num_hidden_layers "$HIDDEN" \
    --activation "$ACTIVATION" \
    --dropout_rate "$DROPOUT" \
    --num_neurons "$NEURONS" \
    --learning_rate "$LR" \
    --weight_decay "$WD" \
    --num_epochs "$EPOCHS" \
    --omega_dt "$O3" 2>&1 | tee -a "$LOG_FILE"

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
