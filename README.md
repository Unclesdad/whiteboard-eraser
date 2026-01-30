# Whiteboard Eraser

An autonomous whiteboard eraser robot built on a Raspberry Pi 5. Uses computer vision to detect markings on a whiteboard and navigates to erase them.

![Wiring Diagram](readme_images/wiring_diagram.png)

![3d Model Screenshot](readme_images/3d_model_screenshot.png)

## Project Structure

```
whiteboard-eraser/
├── lib/                    # Hardware abstraction layer
│   ├── motor.py            # N20 motor control with encoders
│   ├── servo.py            # Steering servo control
│   └── gyro.py             # MPU-6050 gyroscope/IMU
├── src/                    # Algorithm and application code
│   ├── whiteboard_eraser_main.py   # Main control loop
│   ├── car_controller.py           # PID control for motion
│   ├── simple_marking_detector.py  # Computer vision detection
│   ├── localization.py             # Position tracking (odometry + gyro)
│   ├── mapping.py                  # Global marking map
│   ├── pathfinder.py               # A* pathfinding
│   └── ...
├── setup/                  # Installation scripts
│   ├── setup_service.sh    # Systemd service installer
│   └── install_dependencies.sh
└── ...
```

### lib/ - Hardware Abstraction

Low-level drivers that interface directly with hardware components. These modules abstract away GPIO pins, I2C communication, and PWM signals into clean Python APIs.

### src/ - Algorithms

High-level application code including computer vision, path planning, localization, and the main state machine. These modules use the hardware abstractions from `lib/` to control the robot.

## Getting Started

See [USAGE.md](USAGE.md) for installation and usage instructions.

