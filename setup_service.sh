#!/bin/bash

# Setup script for Whiteboard Eraser Car Service
# Run this script with sudo to install the service

set -e

echo "Setting up Whiteboard Eraser Car Service..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run this script with sudo"
    echo "Usage: sudo ./setup_service.sh"
    exit 1
fi

# Get the actual user (not root)
REAL_USER=${SUDO_USER:-$USER}
if [ "$REAL_USER" = "root" ]; then
    echo "Please run with sudo from a regular user account"
    exit 1
fi

echo "Setting up for user: $REAL_USER"

# Determine the installation directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/home/$REAL_USER/whiteboard-eraser"

echo "Script directory: $SCRIPT_DIR"
echo "Install directory: $INSTALL_DIR"

# Create installation directory if it doesn't exist
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Creating installation directory..."
    mkdir -p "$INSTALL_DIR"
    chown "$REAL_USER:$REAL_USER" "$INSTALL_DIR"
fi

# Copy files to installation directory (if not already there)
if [ "$SCRIPT_DIR" != "$INSTALL_DIR" ]; then
    echo "Copying files to installation directory..."

    # Copy Python files
    cp "$SCRIPT_DIR"/*.py "$INSTALL_DIR/"

    # Copy other necessary files
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        cp "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/"
    fi

    if [ -f "$SCRIPT_DIR/README_USAGE.md" ]; then
        cp "$SCRIPT_DIR/README_USAGE.md" "$INSTALL_DIR/"
    fi

    # Set ownership
    chown -R "$REAL_USER:$REAL_USER" "$INSTALL_DIR"

    echo "✓ Files copied to $INSTALL_DIR"
else
    echo "✓ Already in installation directory"
fi

# Install Python dependencies
echo "Installing Python dependencies..."

# Try different pip commands in order of preference
if command -v pip3 >/dev/null 2>&1; then
    echo "Using pip3..."
    sudo -u "$REAL_USER" pip3 install -r "$INSTALL_DIR/requirements.txt"
elif command -v pip >/dev/null 2>&1; then
    echo "Using pip..."
    sudo -u "$REAL_USER" pip install -r "$INSTALL_DIR/requirements.txt"
elif command -v python3 -m pip >/dev/null 2>&1; then
    echo "Using python3 -m pip..."
    sudo -u "$REAL_USER" python3 -m pip install -r "$INSTALL_DIR/requirements.txt"
else
    echo "⚠️  No pip command found. Please install manually:"
    echo "    python3 -m pip install -r $INSTALL_DIR/requirements.txt"
    echo "Or install pip first:"
    echo "    sudo apt update && sudo apt install python3-pip"
fi

# Make Python files executable
chmod +x "$INSTALL_DIR"/*.py

# Copy service file
echo "Installing service file..."
cp "$SCRIPT_DIR/whiteboard-eraser.service" /etc/systemd/system/

# Update service file with correct paths
sed -i "s|/home/pi/whiteboard-eraser|$INSTALL_DIR|g" /etc/systemd/system/whiteboard-eraser.service
sed -i "s|User=pi|User=$REAL_USER|g" /etc/systemd/system/whiteboard-eraser.service
sed -i "s|Group=pi|Group=$REAL_USER|g" /etc/systemd/system/whiteboard-eraser.service

# Add user to gpio group for hardware access
usermod -a -G gpio "$REAL_USER"
usermod -a -G i2c "$REAL_USER"
usermod -a -G video "$REAL_USER"

# Enable I2C and camera if not already enabled
echo "Enabling I2C and camera..."
raspi-config nonint do_i2c 0  # Enable I2C
raspi-config nonint do_camera 0  # Enable camera (legacy)

# Reload systemd and enable service
echo "Reloading systemd..."
systemctl daemon-reload

echo "Enabling whiteboard-eraser service..."
systemctl enable whiteboard-eraser.service
