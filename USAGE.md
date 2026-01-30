# Whiteboard Eraser Car - Usage Guide

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the system:**
   ```bash
   python test_system.py --all
   ```

3. **Run the whiteboard eraser:**
   ```bash
   python src/whiteboard_eraser_main.py
   ```

## Auto-Start Setup (Plug and Play)

To set up the car to start automatically when powered on:

1. **Install as a service:**
   ```bash
   sudo ./setup/setup_service.sh
   ```

2. **Reboot the system:**
   ```bash
   sudo reboot
   ```

3. **The car will now automatically:**
   - Wait 5 seconds after boot
   - Calibrate its encoders by moving forward then backward
   - Perform a circular scan to detect markings
   - Begin systematic erasing of detected markings

4. **Monitor the service:**
   ```bash
   sudo journalctl -u whiteboard-eraser -f
   ```

## Testing Individual Components

### Test Marking Detection
```bash
python test_system.py --detection
```
Place some whiteboard images (with markings) in the directory first.

### Test Hardware Integration
```bash
python test_system.py --controller --camera
```

### Test Full System (Safe Mode)
```bash
python test_system.py --integration
```

## Command Line Options

### Main System
```bash
python src/whiteboard_eraser_main.py [options]

Options:
  --debug                 Enable debug mode with camera view
  --camera-width WIDTH    Camera resolution width (default: 640)
  --camera-height HEIGHT  Camera resolution height (default: 480)
  --max-speed SPEED       Maximum speed in mm/s (default: 150)
  --scan-time TIME        Maximum scan time in seconds (default: 60)
```

### Testing
```bash
python test_system.py [options]

Options:
  --all                   Run all tests
  --detection             Test marking detection only
  --localization          Test position tracking only
  --mapping               Test global mapping only
  --pathfinding           Test path planning only
  --controller            Test car control only
  --camera                Test camera integration only
  --integration           Test full system integration
```

## System Architecture

```
whiteboard-eraser/
├── lib/                    # Hardware abstraction layer
│   ├── motor.py
│   ├── servo.py
│   └── gyro.py
├── src/                    # Algorithm and application code
│   ├── whiteboard_eraser_main.py
│   ├── car_controller.py
│   ├── simple_marking_detector.py
│   ├── localization.py
│   ├── mapping.py
│   ├── pathfinder.py
│   └── ...
└── setup/                  # Installation scripts
```

### lib/ - Hardware Abstraction

Low-level drivers that interface directly with hardware. These abstract GPIO, I2C, and PWM into clean APIs.

- **motor.py** - N20 motor control with encoder reading
- **gyro.py** - MPU-6050 gyroscope/accelerometer with sensor fusion
- **servo.py** - Steering servo PWM control

### src/ - Algorithms

High-level application code that uses the hardware abstractions from `lib/`.

- **whiteboard_eraser_main.py** - Main state machine and control loop
- **car_controller.py** - PID control for position and heading
- **simple_marking_detector.py** - Computer vision for detecting markings
- **localization.py** - Position tracking using encoders + gyro fusion
- **mapping.py** - Global map of detected markings with clustering
- **pathfinder.py** - A* pathfinding with Ackermann steering constraints

## Operation States

### Startup Sequence
1. **STARTUP_DELAY** - Wait 5 seconds after power-on
2. **CALIBRATING_ENCODERS** - Move forward/backward to calibrate encoder differences
3. **INITIAL_SCAN** - Drive in circle to detect markings

### Main Operation
4. **SCANNING** - Continue detecting and mapping markings
5. **PLANNING** - Plan path to next marking cluster
6. **NAVIGATING** - Follow planned path to target
7. **ERASING** - Mark area as erased when car drives over it
8. **COMPLETED** - All markings processed

## Key Features

### Computer Vision
- Handles upside-down camera mounting (180° rotation)
- Efficient marking detection optimized for RPi5
- Real-world coordinate conversion from pixels
- Adaptive thresholding for different lighting

### Localization
- Combines wheel encoder odometry with gyroscope
- Drift-free heading using sensor fusion
- Coordinate transformations between reference frames
- Position validation and error detection

### Mapping
- Global marking map with confidence tracking
- Duplicate detection and merging
- Clustering for efficient pathfinding
- Progress tracking and persistence

### Pathfinding
- A* search with Ackermann steering constraints
- Respects minimum turning radius
- Obstacle avoidance
- Path smoothing for natural motion

### Control
- PID control for position and heading
- Smooth velocity commands
- Emergency stop capabilities
- Real-time status monitoring

## Calibration Tips

### Camera Calibration
1. Test detection on sample images first
2. Adjust `marking_threshold` if markings not detected
3. Verify coordinate conversion using known distances

### Motor Calibration
1. Check encoder tick counts per revolution
2. Measure actual wheel radius
3. Calibrate `mm_per_tick` conversion factor

### Steering Calibration
1. Verify servo center position (90°)
2. Measure actual max steering angles
3. Test turning radius calculations

## Troubleshooting

### No Markings Detected
- Check camera focus and exposure
- Verify marking contrast against whiteboard
- Test with debug mode: `--debug`

### Poor Navigation Accuracy
- Calibrate wheel encoder counts
- Check gyroscope calibration
- Verify motor speed consistency

### Path Planning Failures
- Check obstacle map boundaries
- Reduce max speed for tighter turns
- Increase planning time limit

### Hardware Issues
- Run component tests: `python test_system.py --all`
- Check GPIO connections and permissions
- Verify I2C communication for gyroscope

## Performance Optimization

### For Better Detection Speed
- Reduce camera resolution
- Lower detection frequency
- Use smaller blur kernel size

### For Better Navigation
- Increase control loop frequency
- Tune PID parameters
- Use path smoothing

### For Better Coverage
- Increase scan time
- Lower minimum confidence threshold
- Use smaller merge distance

## Safety Notes

- Always test in simulation mode first
- Use emergency stop (Ctrl+C) if needed
- Start with slow speeds for testing
- Ensure adequate clearance around whiteboard
- Monitor system status during operation

## File Output

The system saves several files during operation:

- `whiteboard_map_[timestamp].json` - Periodic map saves
- `final_whiteboard_map.json` - Complete final map
- `shutdown_map.json` - Map saved on emergency shutdown

These can be loaded later for analysis or resuming operation.