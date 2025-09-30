#!/usr/bin/env python3
"""
Whiteboard Eraser Car - Main Control Script
Integrates all systems for autonomous marking detection and erasure
"""

import cv2
import numpy as np
import time
import threading
import signal
import sys
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import our custom modules
from marking_detector import MarkingDetector
from localization import LocalizationSystem
from mapping import GlobalMap
from pathfinder import AckermannPathfinder, ObstacleMap, Path, CarConfig
from car_controller import CarController, CarState

# Import camera
try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    print("Warning: picamera2 not available. Running without camera.")
    CAMERA_AVAILABLE = False

class EraserState(Enum):
    """Main state machine states"""
    INITIALIZING = "initializing"
    STARTUP_DELAY = "startup_delay"
    WHITEBOARD_MAPPING = "whiteboard_mapping"
    INITIAL_SCAN = "initial_scan"
    SCANNING = "scanning"
    PLANNING = "planning"
    NAVIGATING = "navigating"
    ERASING = "erasing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class EraserConfig:
    """Configuration for the eraser system"""
    # Camera settings
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 10

    # Detection settings
    detection_interval: float = 0.2  # seconds between detections
    min_marking_confidence: float = 0.4

    # Navigation settings
    goal_tolerance_mm: float = 25.0
    max_speed_mm_s: float = 150.0
    erase_radius_mm: float = 25.0

    # Safety settings
    max_planning_time: float = 10.0
    max_navigation_time: float = 30.0

    # Performance settings
    max_scan_time: float = 60.0  # seconds to scan before starting erasure
    save_map_interval: float = 30.0  # seconds between map saves

    # Startup calibration settings
    startup_delay: float = 12.0  # seconds to wait for gyro calibration + system init
    gyro_calibration_samples: int = 1000  # gyro calibration samples
    whiteboard_mapping_time: float = 20.0  # seconds to map whiteboard orientation
    circle_scan_time: float = 15.0  # seconds for additional marking scan
    circle_radius_mm: float = 300.0  # radius of scanning circle

class WhiteboardEraserMain:
    """
    Main control system for autonomous whiteboard eraser car
    """

    def __init__(self, config: EraserConfig = None, debug: bool = False):
        """
        Initialize the whiteboard eraser system

        Args:
            config: System configuration
            debug: Enable debug mode with visualization
        """
        self.config = config or EraserConfig()
        self.debug = debug

        # System state
        self.state = EraserState.INITIALIZING
        self.state_lock = threading.Lock()
        self.running = False
        self.shutdown_requested = False

        # Initialize subsystems
        self.marking_detector: Optional[MarkingDetector] = None
        self.localization: Optional[LocalizationSystem] = None
        self.global_map: Optional[GlobalMap] = None
        self.pathfinder: Optional[AckermannPathfinder] = None
        self.car_controller: Optional[CarController] = None
        self.camera: Optional[Picamera2] = None

        # Threading
        self.main_thread: Optional[threading.Thread] = None
        self.camera_thread: Optional[threading.Thread] = None

        # Performance tracking
        self.start_time = time.time()
        self.last_detection_time = 0.0
        self.last_map_save_time = 0.0

        # Current navigation
        self.current_path: Optional[Path] = None
        self.current_target: Optional[Tuple[float, float]] = None
        self.navigation_start_time = 0.0

        # Statistics
        self.total_markings_detected = 0
        self.total_markings_erased = 0
        self.total_distance_traveled = 0.0

        # Encoder calibration data (removed - not needed)
        self.left_encoder_scale = 1.0
        self.right_encoder_scale = 1.0

        # Whiteboard orientation mapping
        self.whiteboard_right_angle = 0.0  # Gyro angle pointing to right edge (start position)
        self.whiteboard_up_angle = 0.0     # Gyro angle pointing up (against gravity)
        self.whiteboard_left_angle = 0.0   # Gyro angle pointing to left edge
        self.whiteboard_down_angle = 0.0   # Gyro angle pointing down (with gravity)
        self.whiteboard_mapped = False
        self.mapping_start_angle = 0.0
        self.mapping_angles = []  # Store angles during mapping

        # State timing
        self.state_start_time = 0.0
        self.gyro_calibration_complete = False

        print("WhiteboardEraserMain initialized")
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, _):
            print(f"\nReceived signal {signum}. Shutting down...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def initialize_systems(self) -> bool:
        """
        Initialize all subsystems

        Returns:
            True if initialization successful
        """
        try:
            print("Initializing subsystems...")

            # Initialize marking detector
            self.marking_detector = MarkingDetector(
                image_width=self.config.camera_width,
                image_height=self.config.camera_height,
                debug=self.debug
            )
            print("✓ Marking detector initialized")

            # Initialize localization
            self.localization = LocalizationSystem()
            # Try to load previous whiteboard orientation
            self._load_whiteboard_orientation()
            print("✓ Localization system initialized")

            # Initialize global map
            self.global_map = GlobalMap(
                min_confidence=self.config.min_marking_confidence,
                erase_confirmation_distance=self.config.erase_radius_mm
            )
            print("✓ Global map initialized")

            # Initialize pathfinder
            obstacle_map = ObstacleMap(width_mm=2000, height_mm=1500)  # Adjust to your whiteboard size
            car_config = CarConfig()
            # Will set whiteboard orientation after mapping is complete
            self.pathfinder = AckermannPathfinder(car_config, obstacle_map)
            print("✓ Pathfinder initialized")

            # Initialize car controller
            self.car_controller = CarController(
                max_speed_mm_s=self.config.max_speed_mm_s,
                update_rate_hz=50.0
            )
            print("✓ Car controller initialized")

            # Initialize camera
            if CAMERA_AVAILABLE:
                self.camera = Picamera2()
                camera_config = self.camera.create_preview_configuration(
                    main={"size": (self.config.camera_width, self.config.camera_height)}
                )
                self.camera.configure(camera_config)
                self.camera.start()
                print("✓ Camera initialized")
            else:
                print("⚠ Camera not available - using simulation mode")

            # Start subsystem threads
            self.car_controller.start_control_loop()

            with self.state_lock:
                self.state = EraserState.STARTUP_DELAY
                self.state_start_time = time.time()

            print("✓ All systems initialized successfully")
            return True

        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            with self.state_lock:
                self.state = EraserState.ERROR
            return False

    def start(self):
        """Start the main control loop"""
        if not self.initialize_systems():
            return

        self.running = True
        self.start_time = time.time()

        # Start main control thread
        self.main_thread = threading.Thread(target=self._main_control_loop, daemon=True)
        self.main_thread.start()

        # Start camera processing thread if available
        if self.camera:
            self.camera_thread = threading.Thread(target=self._camera_processing_loop, daemon=True)
            self.camera_thread.start()

        print("🚗 Whiteboard eraser started!")
        print("Press Ctrl+C to stop")

        # Main thread monitoring
        try:
            while self.running and not self.shutdown_requested:
                self._print_status()
                time.sleep(1.0)

        except KeyboardInterrupt:
            print("\nShutdown requested...")

        finally:
            self.shutdown()

    def _main_control_loop(self):
        """Main state machine control loop"""
        while self.running:
            try:
                with self.state_lock:
                    current_state = self.state

                if current_state == EraserState.STARTUP_DELAY:
                    self._handle_startup_delay_state()

                elif current_state == EraserState.WHITEBOARD_MAPPING:
                    self._handle_whiteboard_mapping_state()

                elif current_state == EraserState.INITIAL_SCAN:
                    self._handle_initial_scan_state()

                elif current_state == EraserState.SCANNING:
                    self._handle_scanning_state()

                elif current_state == EraserState.PLANNING:
                    self._handle_planning_state()

                elif current_state == EraserState.NAVIGATING:
                    self._handle_navigating_state()

                elif current_state == EraserState.ERASING:
                    self._handle_erasing_state()

                elif current_state == EraserState.COMPLETED:
                    self._handle_completed_state()

                elif current_state == EraserState.ERROR:
                    self._handle_error_state()

                # Periodic map saving
                self._periodic_map_save()

                time.sleep(0.1)  # 10Hz main loop

            except Exception as e:
                print(f"Error in main control loop: {e}")
                with self.state_lock:
                    self.state = EraserState.ERROR

    def _camera_processing_loop(self):
        """Camera processing loop for continuous marking detection"""
        while self.running and self.camera:
            try:
                current_time = time.time()

                # Limit detection rate
                if current_time - self.last_detection_time < self.config.detection_interval:
                    time.sleep(0.05)
                    continue

                # Capture image
                image = self.camera.capture_array()

                # Convert from RGB to BGR for OpenCV
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Detect markings in car coordinates
                car_markings = self.marking_detector.detect_and_convert_to_car_coordinates(image)

                if car_markings:
                    # Update localization with calibrated encoder readings
                    self._update_localization_with_calibrated_encoders()

                    # Convert to global coordinates using localization
                    current_pose = self.localization.get_pose()
                    global_markings = []

                    for car_x, car_y, confidence in car_markings:
                        global_x, global_y = self.localization.transform_to_global(car_x, car_y)
                        global_markings.append((global_x, global_y, confidence))

                    # Add to global map
                    if global_markings:
                        self.global_map.add_markings(global_markings)
                        self.total_markings_detected += len(global_markings)

                # Debug visualization
                if self.debug:
                    vis_image = self.marking_detector.visualize_detections(
                        image, self.marking_detector.detect_markings(image)
                    )
                    cv2.imshow("Marking Detection", vis_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.shutdown()

                self.last_detection_time = current_time

            except Exception as e:
                print(f"Error in camera processing: {e}")
                time.sleep(0.1)

    def _handle_startup_delay_state(self):
        """Handle startup delay - wait for gyro calibration and system initialization"""
        elapsed_time = time.time() - self.state_start_time

        # Check if gyro calibration is complete
        if not self.gyro_calibration_complete:
            try:
                if self.car_controller.gyro.is_calibrated:
                    self.gyro_calibration_complete = True
                    print("✓ Gyroscope calibration complete")
                elif elapsed_time > 8.0:  # Force calibration if needed
                    print("🔧 Force-starting gyro calibration...")
                    self.car_controller.gyro.calibrate(samples=self.config.gyro_calibration_samples, show_progress=False)
                    self.gyro_calibration_complete = True
            except:
                # Fallback if gyro not available
                self.gyro_calibration_complete = True

        # Wait for full startup delay
        if elapsed_time >= self.config.startup_delay and self.gyro_calibration_complete:
            print(f"✓ Startup delay complete ({self.config.startup_delay}s)")

            # Reset position
            self.localization.reset_position(0, 0, 0)

            if self.whiteboard_mapped:
                print("✓ Using previously mapped whiteboard orientation")
                print("🔍 Starting marking detection scan...")
                with self.state_lock:
                    self.state = EraserState.INITIAL_SCAN
                    self.state_start_time = time.time()
            else:
                print("🧭 Starting whiteboard orientation mapping...")
                with self.state_lock:
                    self.state = EraserState.WHITEBOARD_MAPPING
                    self.state_start_time = time.time()
        else:
            remaining = max(0, self.config.startup_delay - elapsed_time)
            if int(elapsed_time) % 2 == 0 and elapsed_time > 0:  # Print every 2 seconds
                gyro_status = "✓" if self.gyro_calibration_complete else "⏳"
                print(f"⏱️  Startup: {remaining:.1f}s remaining, Gyro: {gyro_status}")

    def _handle_whiteboard_mapping_state(self):
        """Handle whiteboard orientation mapping - drive in circle to map gyro angles to whiteboard directions"""
        elapsed_time = time.time() - self.state_start_time

        if elapsed_time < 1.0:
            # Initialize mapping
            if elapsed_time < 0.1:
                try:
                    _, _, initial_heading = self.car_controller.gyro.get_orientation()
                    self.mapping_start_angle = np.radians(initial_heading)
                    self.whiteboard_right_angle = self.mapping_start_angle  # Car starts facing right
                    self.mapping_angles = []
                    print(f"🧭 Starting orientation mapping from {initial_heading:.1f}° (right edge)")
                except:
                    self.mapping_start_angle = 0.0
                    self.whiteboard_right_angle = 0.0

        elif elapsed_time < self.config.whiteboard_mapping_time:
            # Drive in a circle to map orientations
            linear_speed = 0.4  # Medium forward speed
            angular_speed = 0.3  # Gentle turn to make a circle

            self.car_controller.set_manual_control(linear_speed, angular_speed)

            # Record gyro readings periodically
            try:
                _, _, current_heading = self.car_controller.gyro.get_orientation()
                current_angle = np.radians(current_heading)

                # Calculate how far we've turned from start
                angle_turned = self._normalize_angle(current_angle - self.mapping_start_angle)

                # Record orientations at 90-degree intervals
                quarter_turn = np.pi / 2
                progress = abs(angle_turned)

                if progress >= quarter_turn and len(self.mapping_angles) == 0:
                    # 90° - should be pointing up (against gravity)
                    self.whiteboard_up_angle = current_angle
                    self.mapping_angles.append(('up', current_angle))
                    print(f"📍 Mapped UP direction: {np.degrees(current_angle):.1f}°")

                elif progress >= 2 * quarter_turn and len(self.mapping_angles) == 1:
                    # 180° - should be pointing left
                    self.whiteboard_left_angle = current_angle
                    self.mapping_angles.append(('left', current_angle))
                    print(f"📍 Mapped LEFT direction: {np.degrees(current_angle):.1f}°")

                elif progress >= 3 * quarter_turn and len(self.mapping_angles) == 2:
                    # 270° - should be pointing down (with gravity)
                    self.whiteboard_down_angle = current_angle
                    self.mapping_angles.append(('down', current_angle))
                    print(f"📍 Mapped DOWN direction: {np.degrees(current_angle):.1f}°")

                elif progress >= 4 * quarter_turn and len(self.mapping_angles) == 3:
                    # 360° - back to right, confirm mapping
                    self.mapping_angles.append(('right_confirm', current_angle))
                    print(f"📍 Completed circle: {np.degrees(current_angle):.1f}° (back to right)")

                # Print progress occasionally
                if int(elapsed_time) % 3 == 0 and elapsed_time > 3:
                    remaining_time = self.config.whiteboard_mapping_time - elapsed_time
                    print(f"🧭 Mapping... {len(self.mapping_angles)}/4 directions, {remaining_time:.1f}s remaining")

            except Exception as e:
                print(f"⚠️ Error reading gyro during mapping: {e}")

        else:
            # Stop and finalize mapping
            self.car_controller.stop_all_motors()

            if len(self.mapping_angles) >= 3:
                self.whiteboard_mapped = True
                print("✓ Whiteboard orientation mapping complete:")
                print(f"  Right (start): {np.degrees(self.whiteboard_right_angle):.1f}°")
                print(f"  Up (↑):        {np.degrees(self.whiteboard_up_angle):.1f}°")
                print(f"  Left (←):      {np.degrees(self.whiteboard_left_angle):.1f}°")
                print(f"  Down (↓):      {np.degrees(self.whiteboard_down_angle):.1f}°")

                # Save orientation mapping for future use
                try:
                    import json
                    orientation_data = {
                        'whiteboard_right_angle': float(self.whiteboard_right_angle),
                        'whiteboard_up_angle': float(self.whiteboard_up_angle),
                        'whiteboard_left_angle': float(self.whiteboard_left_angle),
                        'whiteboard_down_angle': float(self.whiteboard_down_angle),
                        'timestamp': time.time()
                    }
                    with open('whiteboard_orientation.json', 'w') as f:
                        json.dump(orientation_data, f, indent=2)
                    print("💾 Orientation mapping saved to whiteboard_orientation.json")
                except Exception as e:
                    print(f"⚠️ Could not save orientation mapping: {e}")

                # Update pathfinder with whiteboard orientation
                self.pathfinder.whiteboard_up_direction = self.whiteboard_up_angle
                print("✓ Pathfinder updated with gravity-aware navigation")

            else:
                print("⚠️ Incomplete orientation mapping - using defaults")
                # Set default orientations (assuming gyro 0° = right)
                self.whiteboard_right_angle = 0.0
                self.whiteboard_up_angle = np.pi / 2
                self.whiteboard_left_angle = np.pi
                self.whiteboard_down_angle = 3 * np.pi / 2
                self.whiteboard_mapped = True

                # Update pathfinder with default orientation
                self.pathfinder.whiteboard_up_direction = self.whiteboard_up_angle

            print("🔍 Starting marking detection scan...")
            with self.state_lock:
                self.state = EraserState.INITIAL_SCAN
                self.state_start_time = time.time()

    def _handle_initial_scan_state(self):
        """Handle initial circular scan to detect markings"""
        elapsed_time = time.time() - self.state_start_time

        if elapsed_time < self.config.circle_scan_time:
            # Drive in a circle to scan for markings
            # Use constant forward speed with turning
            linear_speed = 0.4  # Medium forward speed
            angular_speed = 0.3  # Gentle turn to make a circle

            self.car_controller.set_manual_control(linear_speed, angular_speed)

            # Print progress occasionally
            if int(elapsed_time) % 3 == 0 and elapsed_time > 0:
                markings = self.global_map.get_active_markings()
                remaining_time = self.config.circle_scan_time - elapsed_time

                # Show movement direction if whiteboard is mapped
                if self.whiteboard_mapped and len(markings) > 0:
                    pose = self.localization.get_pose()
                    last_marking = markings[-1]
                    direction = self.get_movement_direction_type(pose.x, pose.y, last_marking.x, last_marking.y)
                    print(f"🔍 Scanning... {len(markings)} markings found, {remaining_time:.1f}s remaining")
                    print(f"    Last marking direction: {direction}")
                else:
                    print(f"🔍 Scanning... {len(markings)} markings found, {remaining_time:.1f}s remaining")

        else:
            # Stop and evaluate findings
            self.car_controller.stop_all_motors()
            markings = self.global_map.get_active_markings()

            print(f"✓ Initial scan complete - found {len(markings)} markings")

            if len(markings) > 0:
                print("🎯 Beginning systematic erasing...")
                with self.state_lock:
                    self.state = EraserState.PLANNING
            else:
                print("🔍 No markings found, continuing to search...")
                with self.state_lock:
                    self.state = EraserState.SCANNING

    def _handle_scanning_state(self):
        """Handle scanning state - collect markings before starting erasure"""
        elapsed_time = time.time() - self.start_time

        # Check if we should start erasing
        active_markings = self.global_map.get_active_markings()

        if len(active_markings) > 0 and elapsed_time > 10.0:  # Found markings after 10s
            print(f"Scanning complete. Found {len(active_markings)} markings.")
            with self.state_lock:
                self.state = EraserState.PLANNING

        elif elapsed_time > self.config.max_scan_time:  # Max scan time reached
            if len(active_markings) > 0:
                print(f"Max scan time reached. Starting with {len(active_markings)} markings.")
                with self.state_lock:
                    self.state = EraserState.PLANNING
            else:
                print("No markings found during scan. Task completed.")
                with self.state_lock:
                    self.state = EraserState.COMPLETED

    def _handle_planning_state(self):
        """Handle planning state - plan path to next marking"""
        # Get current position
        pose = self.localization.get_pose()

        # Get next target
        targets = self.global_map.get_erase_targets_ordered(pose.x, pose.y)

        if not targets:
            print("No more targets to erase. Task completed!")
            with self.state_lock:
                self.state = EraserState.COMPLETED
            return

        target_x, target_y = targets[0]
        self.current_target = (target_x, target_y)

        print(f"Planning path to ({target_x:.1f}, {target_y:.1f})")

        # Plan path
        self.current_path = self.pathfinder.plan_path(
            pose.x, pose.y, pose.theta,
            target_x, target_y,
            max_planning_time=self.config.max_planning_time
        )

        if self.current_path:
            print(f"Path planned: {len(self.current_path.waypoints)} waypoints, "
                  f"{self.current_path.total_distance:.1f}mm")

            # Start navigation
            self.navigation_start_time = time.time()
            self.car_controller.set_target_position(target_x, target_y)

            with self.state_lock:
                self.state = EraserState.NAVIGATING

        else:
            print("Failed to plan path. Trying simple approach...")
            # Try simple direct path
            simple_path = self.pathfinder.plan_simple_path(
                pose.x, pose.y, pose.theta, target_x, target_y
            )

            if simple_path:
                self.current_path = simple_path
                self.navigation_start_time = time.time()
                self.car_controller.set_target_position(target_x, target_y)

                with self.state_lock:
                    self.state = EraserState.NAVIGATING
            else:
                print("Cannot reach target. Marking as unreachable.")
                # Mark area as erased to skip it
                self.global_map.mark_area_erased(target_x, target_y, 50.0)
                # Try again with next target
                with self.state_lock:
                    self.state = EraserState.PLANNING

    def _handle_navigating_state(self):
        """Handle navigation state - move towards target"""
        if not self.current_target:
            with self.state_lock:
                self.state = EraserState.PLANNING
            return

        pose = self.localization.get_pose()
        target_x, target_y = self.current_target

        # Check if we've reached the target
        distance_to_target = np.sqrt((target_x - pose.x)**2 + (target_y - pose.y)**2)

        if distance_to_target <= self.config.goal_tolerance_mm:
            print(f"Reached target! Distance: {distance_to_target:.1f}mm")
            with self.state_lock:
                self.state = EraserState.ERASING
            return

        # Check for navigation timeout
        elapsed_time = time.time() - self.navigation_start_time
        if elapsed_time > self.config.max_navigation_time:
            print(f"Navigation timeout after {elapsed_time:.1f}s")
            self.car_controller.stop_all_motors()
            with self.state_lock:
                self.state = EraserState.PLANNING  # Try again

        # Check car controller state
        car_status = self.car_controller.get_status()
        if car_status.state == CarState.STOPPED:
            # Car stopped - either reached target or failed
            if distance_to_target <= self.config.goal_tolerance_mm * 2:
                print("Close enough to target for erasing")
                with self.state_lock:
                    self.state = EraserState.ERASING
            else:
                print(f"Car stopped unexpectedly. Distance to target: {distance_to_target:.1f}mm")
                with self.state_lock:
                    self.state = EraserState.PLANNING

    def _handle_erasing_state(self):
        """Handle erasing state - mark area as erased"""
        if not self.current_target:
            with self.state_lock:
                self.state = EraserState.PLANNING
            return

        target_x, target_y = self.current_target
        pose = self.localization.get_pose()

        print(f"Erasing area around ({target_x:.1f}, {target_y:.1f})")

        # Mark area as erased
        erased_ids = self.global_map.mark_area_erased(
            pose.x, pose.y, self.config.erase_radius_mm
        )

        if erased_ids:
            self.total_markings_erased += len(erased_ids)
            print(f"Erased {len(erased_ids)} markings")

        # Small delay to simulate erasing action
        time.sleep(1.0)

        # Move to next target
        self.current_target = None
        self.current_path = None

        with self.state_lock:
            self.state = EraserState.PLANNING

    def _handle_completed_state(self):
        """Handle completion state"""
        print("🎉 Erasing task completed!")
        self._print_final_statistics()
        self.car_controller.stop_all_motors()

        # Save final map
        self.global_map.save_map("final_whiteboard_map.json")

        # Stop main loop
        self.running = False

    def _handle_error_state(self):
        """Handle error state"""
        print("❌ System in error state")
        self.car_controller.emergency_stop()
        time.sleep(1.0)

        # Try to recover
        with self.state_lock:
            self.state = EraserState.SCANNING

    def _periodic_map_save(self):
        """Periodically save the map"""
        current_time = time.time()
        if current_time - self.last_map_save_time > self.config.save_map_interval:
            self.global_map.save_map(f"whiteboard_map_{int(current_time)}.json")
            self.last_map_save_time = current_time

    def _print_status(self):
        """Print current system status"""
        with self.state_lock:
            current_state = self.state

        pose = self.localization.get_pose()
        car_status = self.car_controller.get_status()
        progress = self.global_map.estimate_completion_progress()

        elapsed_time = time.time() - self.start_time

        print(f"\n📊 Status (t={elapsed_time:.0f}s):")
        print(f"  State: {current_state.value}")

        if current_state not in [EraserState.STARTUP_DELAY]:
            print(f"  Position: ({pose.x:.1f}, {pose.y:.1f}), θ={np.degrees(pose.theta):.1f}°")
            print(f"  Car: {car_status.state.value}, speed={car_status.linear_velocity:.1f}mm/s")
            print(f"  Markings: {progress['total_detected']} detected, {progress['total_erased']} erased")
            print(f"  Progress: {progress['progress_percent']:.1f}%")

        if self.whiteboard_mapped:
            print(f"  Whiteboard: Up={np.degrees(self.whiteboard_up_angle):.0f}°, Down={np.degrees(self.whiteboard_down_angle):.0f}°")
        else:
            print(f"  Whiteboard: Not mapped")

        if self.current_target:
            target_x, target_y = self.current_target
            distance = np.sqrt((target_x - pose.x)**2 + (target_y - pose.y)**2)
            print(f"  Target: ({target_x:.1f}, {target_y:.1f}), distance={distance:.1f}mm")

    def _print_final_statistics(self):
        """Print final performance statistics"""
        elapsed_time = time.time() - self.start_time
        progress = self.global_map.estimate_completion_progress()

        print("\n🏁 Final Statistics:")
        print(f"  Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
        print(f"  Total markings detected: {progress['total_detected']}")
        print(f"  Total markings erased: {progress['total_erased']}")
        print(f"  Completion rate: {progress['progress_percent']:.1f}%")
        print(f"  False positives removed: {progress['false_positives_removed']}")

        if self.marking_detector:
            perf_stats = self.marking_detector.get_performance_stats()
            print(f"  Detection performance: {perf_stats['avg_time_ms']:.1f}ms avg ({perf_stats['fps']:.1f} FPS)")

    def shutdown(self):
        """Graceful shutdown of all systems"""
        print("\n🛑 Shutting down whiteboard eraser...")

        self.shutdown_requested = True
        self.running = False

        # Stop car
        if self.car_controller:
            self.car_controller.emergency_stop()
            self.car_controller.cleanup()

        # Stop camera
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except:
                pass

        # Close debug windows
        if self.debug:
            cv2.destroyAllWindows()

        # Save final map
        if self.global_map:
            self.global_map.save_map("shutdown_map.json")

        # Wait for threads
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=2.0)

        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)

        print("✓ Shutdown complete")

    def _load_whiteboard_orientation(self):
        """Load whiteboard orientation from previous run if available"""
        try:
            import json
            with open('whiteboard_orientation.json', 'r') as f:
                orientation_data = json.load(f)

            self.whiteboard_right_angle = orientation_data.get('whiteboard_right_angle', 0.0)
            self.whiteboard_up_angle = orientation_data.get('whiteboard_up_angle', np.pi/2)
            self.whiteboard_left_angle = orientation_data.get('whiteboard_left_angle', np.pi)
            self.whiteboard_down_angle = orientation_data.get('whiteboard_down_angle', 3*np.pi/2)
            self.whiteboard_mapped = True

            print(f"🧭 Loaded whiteboard orientation: Up={np.degrees(self.whiteboard_up_angle):.1f}°")

            # Update pathfinder if it exists
            if hasattr(self, 'pathfinder') and self.pathfinder:
                self.pathfinder.whiteboard_up_direction = self.whiteboard_up_angle

        except FileNotFoundError:
            print("🧭 No previous whiteboard orientation found - will map on startup")
            self.whiteboard_mapped = False
        except Exception as e:
            print(f"⚠️ Error loading whiteboard orientation: {e}")
            self.whiteboard_mapped = False

    def get_movement_direction_type(self, from_x: float, from_y: float,
                                  to_x: float, to_y: float) -> str:
        """
        Determine movement direction relative to whiteboard: up, down, left, right

        Returns:
            'up', 'down', 'left', 'right', or 'diagonal'
        """
        if not self.whiteboard_mapped:
            return 'unknown'

        # Calculate movement vector
        dx = to_x - from_x
        dy = to_y - from_y

        if abs(dx) < 10 and abs(dy) < 10:
            return 'stationary'

        # Calculate movement angle
        movement_angle = np.arctan2(dy, dx)

        # Calculate angles relative to whiteboard directions
        angle_to_up = abs(self._normalize_angle(movement_angle - self.whiteboard_up_angle))
        angle_to_down = abs(self._normalize_angle(movement_angle - self.whiteboard_down_angle))
        angle_to_left = abs(self._normalize_angle(movement_angle - self.whiteboard_left_angle))
        angle_to_right = abs(self._normalize_angle(movement_angle - self.whiteboard_right_angle))

        # Find closest direction (within 45 degrees)
        threshold = np.pi / 4  # 45 degrees

        min_angle = min(angle_to_up, angle_to_down, angle_to_left, angle_to_right)

        if min_angle > threshold:
            return 'diagonal'
        elif min_angle == angle_to_up:
            return 'up'
        elif min_angle == angle_to_down:
            return 'down'
        elif min_angle == angle_to_left:
            return 'left'
        else:
            return 'right'

    def _update_localization_with_calibrated_encoders(self):
        """Update localization using encoder readings"""
        # Get encoder counts
        left_ticks = self.car_controller.left_motor.get_encoder_count()
        right_ticks = self.car_controller.right_motor.get_encoder_count()

        # Update localization
        self.localization.update_with_encoders(left_ticks, right_ticks)

        # Also update with gyro if available
        try:
            _, _, gyro_heading = self.car_controller.gyro.get_orientation()
            self.localization.update_with_gyro(np.radians(gyro_heading))
        except:
            pass  # Gyro not available or failed

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Whiteboard Eraser Car")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera width")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera height")
    parser.add_argument("--max-speed", type=float, default=150.0, help="Max speed (mm/s)")
    parser.add_argument("--scan-time", type=float, default=60.0, help="Max scan time (s)")

    args = parser.parse_args()

    # Create configuration
    config = EraserConfig(
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        max_speed_mm_s=args.max_speed,
        max_scan_time=args.scan_time
    )

    # Create and start eraser system
    eraser = WhiteboardEraserMain(config=config, debug=args.debug)

    try:
        eraser.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        eraser.shutdown()


if __name__ == "__main__":
    main()