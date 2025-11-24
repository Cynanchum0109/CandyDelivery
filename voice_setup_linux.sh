#!/bin/bash
# Automated setup script for running voice_candy_chat.py on Linux (Debian/Ubuntu based).
# It checks for prerequisites, creates a virtual environment, and installs Python packages.

set -e

VENV_DIR="venv"
PYTHON_BIN=${PYTHON_BIN:-python3}

info() { echo -e "\033[1;34m[INFO]\033[0m $1"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $1"; }

check_python() {
    if ! command -v "$PYTHON_BIN" >/dev/null; then
        error "Python not found. Install python3 (preferred 3.10/3.11)."
        exit 1
    fi
    info "$PYTHON_BIN found: $($PYTHON_BIN --version)"
}

install_system_packages() {
    info "Installing system dependencies (requires sudo)..."
    sudo apt-get update
    sudo apt-get install -y build-essential ffmpeg portaudio19-dev python3-dev python3-venv
}

create_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtual environment in $VENV_DIR"
        $PYTHON_BIN -m venv "$VENV_DIR"
    else
        warn "Virtual environment already exists at $VENV_DIR"
    fi
}

activate_venv() {
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    info "Virtual environment activated."
}

install_python_packages() {
    info "Installing Python dependencies..."
    pip install --upgrade pip
    pip install \
        openai \
        speechrecognition \
        pyttsx3 \
        pyaudio \
        faster-whisper \
        websockets
}

main() {
    check_python
    install_system_packages
    create_venv
    activate_venv
    install_python_packages
    info "Setup complete. To run voice chat:"
    echo "    source $VENV_DIR/bin/activate"
    echo "    python voice_candy_chat.py"
}

main "$@"
