#!/bin/bash

echo "--- Starting Environment Setup Process ---"

# Update package list and install necessary system packages
echo ">>> Updating package list and installing system packages..."
sudo apt-get update
sudo apt-get install -y libopenblas-dev python3.10-venv wget curl

# Check if python3.10-venv installation was successful
if ! dpkg -s python3.10-venv &> /dev/null; then
    echo "ERROR: Failed to install python3.10-venv. Please check and try again."
    exit 1
fi
echo ">>> System packages installed."

# Create and activate Python virtual environment
VENV_DIR=~/my_env
if [ -d "$VENV_DIR" ]; then
    echo ">>> Virtual environment '$VENV_DIR' already exists."
else
    echo ">>> Creating virtual environment at '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment."
        exit 1
    fi
    echo ">>> Virtual environment created."
fi

echo ">>> Activating virtual environment..."
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment. Please try running 'source $VENV_DIR/bin/activate' manually."
    # Not exiting here so user can try manual activation
fi

# Upgrade pip
echo ">>> Upgrading pip..."
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "WARNING: Failed to upgrade pip."
fi

# Install Python libraries
echo ">>> Installing NumPy..."
pip install numpy
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install NumPy."
    exit 1
fi

echo ">>> Installing JAX with TPU support..."
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install JAX[TPU]. Check network connection and URL."
    exit 1
fi

echo ">>> Installing Keras 3, KerasNLP, and Optax..."
pip install -q -U "keras>=3" keras-nlp optax
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Keras 3, KerasNLP, or Optax."
    exit 1
fi
echo ">>> Python libraries installed."

# Set up environment variables (recommended to add to ~/.bashrc or ~/.zshrc)
echo ">>> Setting up basic environment variables (suggestions for your shell config)..."
echo "export PJRT_DEVICE=TPU"
echo "export KERAS_BACKEND=jax"
echo "export XLA_PYTHON_CLIENT_MEM_FRACTION=1.00" # Use full TPU memory for JAX

# Suggest adding to shell config file
SHELL_CONFIG_FILE=""
if [ -n "$BASH_VERSION" ]; then
    SHELL_CONFIG_FILE="$HOME/.bashrc"
elif [ -n "$ZSH_VERSION" ]; then
    SHELL_CONFIG_FILE="$HOME/.zshrc"
fi

if [ -n "$SHELL_CONFIG_FILE" ]; then
    echo ""
    echo "To have these environment variables set automatically every time you log in,"
    echo "you can add the following lines to your '$SHELL_CONFIG_FILE' file:"
    echo ""
    echo "  export PJRT_DEVICE=TPU"
    echo "  export KERAS_BACKEND=jax"
    echo "  export XLA_PYTHON_CLIENT_MEM_FRACTION=1.00"
    echo "  # export CUDA_VISIBLE_DEVICES=\"-1\" # If you want to hide GPUs"
    echo ""
    echo "Then, run 'source $SHELL_CONFIG_FILE' or open a new terminal."
else
    echo "Could not automatically determine your shell configuration file. Please add the environment variables above manually."
fi


echo "--- Environment Setup Process Completed ---"
echo "Please ensure you have activated the virtual environment: source $VENV_DIR/bin/activate"
echo "And that the PJRT_DEVICE, KERAS_BACKEND, XLA_PYTHON_CLIENT_MEM_FRACTION environment variables are set."

