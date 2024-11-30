#!/bin/bash

# Install system dependencies
if [ -f /etc/debian_version ]; then
    # Debian/Ubuntu
    echo "Installing system dependencies for Debian/Ubuntu..."
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        python3-dev \
        gcc \
        pkg-config \
        libfreetype6-dev \
        libpng-dev \
        python3-matplotlib
elif [ -f /etc/fedora-release ]; then
    # Fedora
    echo "Installing system dependencies for Fedora..."
    sudo dnf install -y \
        gcc \
        gcc-c++ \
        python3-devel \
        pkgconfig \
        freetype-devel \
        libpng-devel \
        python3-matplotlib
else
    echo "Unsupported distribution. Please install the required dependencies manually."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv --python python3.10

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt

echo "Setup complete!"