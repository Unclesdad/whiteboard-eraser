#!/bin/bash
# Install dependencies for Whiteboard Eraser Car
# Run this script on the Raspberry Pi

echo "🔧 Installing dependencies for Whiteboard Eraser Car..."

# Update package list
echo "Updating package list..."
sudo apt update

# Install pip
echo "Installing pip..."
sudo apt install -y python3-pip

# Install system packages for OpenCV and other dependencies
echo "Installing system packages..."
sudo apt install -y python3-opencv python3-numpy python3-rpi.gpio

# Install Python packages via pip
echo "Installing Python packages..."
pip3 install smbus2

# Install picamera2 (Raspberry Pi specific)
echo "Installing picamera2..."
sudo apt install -y python3-picamera2

# Install additional packages that might be needed
echo "Installing additional packages..."
pip3 install dataclasses-json

echo "✅ Dependencies installation complete!"
echo ""
echo "🧪 Testing installation..."

# Test imports
python3 -c "
try:
    import cv2
    print('✅ OpenCV imported successfully')
except ImportError as e:
    print(f'❌ OpenCV import failed: {e}')

try:
    import numpy as np
    print('✅ NumPy imported successfully')
except ImportError as e:
    print(f'❌ NumPy import failed: {e}')

try:
    import RPi.GPIO as GPIO
    print('✅ RPi.GPIO imported successfully')
except ImportError as e:
    print(f'❌ RPi.GPIO import failed: {e}')

try:
    import smbus2
    print('✅ smbus2 imported successfully')
except ImportError as e:
    print(f'❌ smbus2 import failed: {e}')

try:
    from picamera2 import Picamera2
    print('✅ picamera2 imported successfully')
except ImportError as e:
    print(f'❌ picamera2 import failed: {e}')
"

echo ""
echo "📋 Next steps:"
echo "1. Run: python3 test_hardware.py"
echo "2. If hardware test passes, run: python3 test_system.py --all"
echo "3. To install as service: sudo ./setup_service.sh"