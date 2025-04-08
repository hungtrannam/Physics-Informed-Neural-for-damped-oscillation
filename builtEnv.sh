#!/bin/bash

set -e  # Exit script on any error

echo "Updating the system..."
sudo apt update -y && sudo apt upgrade -y

############## Install necessary system dependencies ##############
echo "Installing system dependencies..."
sudo apt install -y python3-venv python3-pip

############## Create a virtual environment ##############
if [ ! -d ".venv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

############## Activate the virtual environment ##############
echo "Activating the virtual environment..."
source .venv/bin/activate

############## Upgrade pip ##############
echo "Upgrading pip..."
pip install --upgrade pip

############## Install required packages ##############
echo "Installing required packages..."
pip install --no-cache-dir \
    numpy matplotlib pandas \
    ipykernel tqdm seaborn optuna jupyter\
    argparse imageio torch scikit-learn \
    torchdiffeq

############## Export the list of installed packages to requirements.txt ##############
echo "Exporting installed packages to requirements.txt..."
pip freeze > requirements.txt

############## Jupyter kernel configuration ##############
echo "Setting up Jupyter kernel..."
python -m ipykernel install --user --name=.venv --display-name "Python (venv)"

############## Additional Utilities ##############
echo "Adding useful utilities..."
if ! grep -q "alias activate_env=" ~/.bashrc; then
    echo "alias activate_env='source $(pwd)/.venv/bin/activate'" >> ~/.bashrc
fi

echo "Setup complete. Run 'source ~/.bashrc' to apply changes."
