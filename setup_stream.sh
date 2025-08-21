#!/bin/bash
# Setup script for Whiteboard Detection Streaming System
# Run this on your Raspberry Pi 5 with Camera Module 3

echo "Whiteboard Detection Streaming Setup"
echo "======================================"

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "Warning: This script is designed for Raspberry Pi"
    echo "   The camera module may not work on other systems"
fi

# Update system packages
echo "Updating system packages..."
sudo apt update

# Install required system packages
echo "Installing system dependencies..."
sudo apt install -y python3-pip python3-flask python3-opencv python3-numpy

# picamera2 should already be installed on Pi OS, but install if missing
if ! python3 -c "import picamera2" 2>/dev/null; then
    echo "Installing picamera2..."
    sudo apt install -y python3-picamera2
else
    echo "picamera2 already installed"
fi

# Install Python packages from requirements
echo "Installing Python dependencies..."
pip3 install -r requirements.txt --user

# Make scripts executable
chmod +x whiteboard_stream.py

# Check camera access
echo "Checking camera access..."
if [ -e /dev/video0 ]; then
    echo "Camera device found at /dev/video0"
else
    echo "Camera device not found. Make sure:"
    echo "   1. Camera module is properly connected"
    echo "   2. Camera is enabled in raspi-config"
    echo "   3. Reboot after enabling camera"
fi

# Get Pi's IP address
PI_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "Setup complete!"
echo ""
echo "Your Pi's IP address: $PI_IP"
echo "Access the stream at: http://$PI_IP:5000"
echo ""
echo "Tips:"
echo "   • Position camera 1-3 feet above whiteboard"
echo "   • Ensure good lighting for best detection"
echo "   • Use contrasting dry-erase markers"
echo ""

# Ask if user wants to start the server now
read -p "Start the streaming server now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting whiteboard detection server..."
    echo "Press Ctrl+C to stop the server"
    echo "Stream will be available at: http://$PI_IP:5000"
    echo ""
    python3 whiteboard_stream.py
else
    echo "To start the server later, run:"
    echo "   python3 whiteboard_stream.py"
    echo ""
    echo "Then access: http://$PI_IP:5000"
fi