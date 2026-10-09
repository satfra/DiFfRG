#!/bin/zsh
# Rebuild the wheel for the DiFfRG package in a virtual environment

# Clean previous builds
rm -rf build dist DiFfRG.egg-info
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip build
python -m build --wheel
