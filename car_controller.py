#!/usr/bin/env python3
"""
Car Controller for Whiteboard Eraser Car
Implements PID control for smooth position and heading control
Integrates with existing motor.py, gyro.py, and servo.py hardware APIs
"""

import time
import threading
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

# Import hardware APIs
try:
    from motor import N20Motor, DualMotorController
    from gyro import RCCarGyro
    from servo import ServoController
    import RPi.GPIO as GPIO
except ImportError:
    print("Warning: Hardware modules not available. Running in simulation mode.")

    # Mock classes for testing
    class N20Motor:
        def __init__(self, *args, **kwargs):
            self.current_speed = 0.0
            self.encoder_count = 0
        def set(self, speed): self.current_speed = speed
        def get_encoder_count(self): return self.encoder_count
        def get_revolutions(self): return self.encoder_count / 28.0
        def reset_encoder(self): self.encoder_count = 0
        def stop(self): self.current_speed = 0.0
        def cleanup(self): pass

    class DualMotorController:
        def __init__(self, *args, **kwargs): pass
        def enable(self): pass
        def disable(self): pass
        def cleanup(self): pass

    class RCCarGyro:
        def __init__(self, *args, **kwargs):
            self.angle_z = 0.0
        def calibrate(self): pass
        def start_continuous_update(self): pass
        def stop_continuous_update(self): pass
        def get_orientation(self): return (0, 0, self.angle_z)
        def reset_orientation(self): self.angle_z = 0.0

    class ServoController:
        def __init__(self, *args, **kwargs):
            self.current_angle = 90
        def set_angle(self, angle): self.current_angle = angle
        def cleanup(self): pass

    class GPIO:
        BCM = 1
        @staticmethod
        def setmode(mode): pass
        @staticmethod
        def setwarnings(flag): pass
        @staticmethod
        def cleanup(): pass

@dataclass
class PIDController:
    """PID controller for smooth control"""
    kp: float
    ki: float
    kd: float
    max_output: float = 1.0
    min_output: float = -1.0
    max_integral: float = 1.0

    def __post_init__(self):
        self.reset()

    def reset(self):
        """Reset PID controller state"""
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error: float, dt: Optional[float] = None) -> float:
        """
        Update PID controller with new error

        Args:
            error: Current error value
            dt: Time delta (auto-calculated if None)

        Returns:
            Control output
        """
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_time

        if dt <= 0:
            return 0.0

        # Proportional term
        proportional = self.kp * error

        # Integral term with windup protection
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        integral = self.ki * self.integral

        # Derivative term
        derivative = self.kd * (error - self.previous_error) / dt

        # Calculate output
        output = proportional + integral + derivative
        output = np.clip(output, self.min_output, self.max_output)

        # Update state
        self.previous_error = error
        self.last_time = current_time

        return output

class CarState(Enum):
    """Car operational states"""
    STOPPED = "stopped"
    MANUAL_CONTROL = "manual"
    POSITION_CONTROL = "position"
    PATH_FOLLOWING = "path_following"
    EMERGENCY_STOP = "emergency"

@dataclass
class CarStatus:
    """Current car status"""
    state: CarState
    position_x: float
    position_y: float
    heading: float
    linear_velocity: float
    angular_velocity: float
    left_motor_speed: float
    right_motor_speed: float
    steering_angle: float
    timestamp: float

class CarController:
    """
    High-level car controller with PID control for smooth motion
    """

    def __init__(self,
                 # Motor configuration
                 left_motor_pins: dict = None,
                 right_motor_pins: dict = None,
                 standby_pin: int = 22,

                 # Servo configuration
                 servo_pin: int = 12,

                 # Gyro configuration
                 gyro_sda_pin: int = 2,
                 gyro_scl_pin: int = 3,

                 # Car physical parameters
                 wheelbase_mm: float = 110.0,
                 track_width_mm: float = 110.0,
                 wheel_radius_mm: float = 30.0,
                 max_speed_mm_s: float = 200.0,

                 # Control parameters
                 update_rate_hz: float = 50.0):
        """
        Initialize car controller

        Args:
            left_motor_pins: Dict with keys: pwm_pin, dir1_pin, dir2_pin, enc_a_pin, enc_b_pin
            right_motor_pins: Dict with keys: pwm_pin, dir1_pin, dir2_pin, enc_a_pin, enc_b_pin
            Other args: Hardware pins and car parameters
        """

        # Default motor pin configurations
        if left_motor_pins is None:
            left_motor_pins = {
                'pwm_pin': 18, 'dir1_pin': 23, 'dir2_pin': 24,
                'enc_a_pin': 17, 'enc_b_pin': 27
            }

        if right_motor_pins is None:
            right_motor_pins = {
                'pwm_pin': 19, 'dir1_pin': 25, 'dir2_pin': 8,
                'enc_a_pin': 5, 'enc_b_pin': 6
            }

        # Store parameters
        self.wheelbase_mm = wheelbase_mm
        self.track_width_mm = track_width_mm
        self.wheel_radius_mm = wheel_radius_mm
        self.max_speed_mm_s = max_speed_mm_s
        self.update_rate_hz = update_rate_hz
        self.update_interval = 1.0 / update_rate_hz

        # Initialize GPIO (critical - robot cannot function without GPIO access)
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self.hardware_available = True
            print("GPIO initialized successfully")
        except Exception as e:
            print(f"CRITICAL: GPIO initialization failed: {e}")
            print("Solutions:")
            print("  - Run with: sudo python3 whiteboard_eraser_main.py")
            print("  - Check if another process is using GPIO")
            print("  - Verify you're running on a Raspberry Pi")
            print("\nCannot control hardware without GPIO access - stopping program")
            raise RuntimeError("GPIO initialization failed - robot cannot access hardware")

        # Initialize hardware (will fall back to mock if GPIO failed)
        self._init_motors(left_motor_pins, right_motor_pins, standby_pin)
        self._init_servo(servo_pin)
        self._init_gyro(gyro_sda_pin, gyro_scl_pin)

        # Initialize PID controllers
        self._init_pid_controllers()

        # State management
        self.state = CarState.STOPPED
        self.state_lock = threading.Lock()

        # Position tracking
        self.position_x = 0.0
        self.position_y = 0.0
        self.heading = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_heading = 0.0

        # Velocity tracking
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # Odometry tracking (for delta calculations)
        self.prev_left_revs = 0.0
        self.prev_right_revs = 0.0

        # Control thread
        self.control_thread = None
        self.running = False

        # Safety limits
        self.max_steering_angle = 45.0  # degrees (±45° from center)
        self.emergency_stop_flag = False

        print(f"CarController initialized:")
        print(f"  Wheelbase: {wheelbase_mm}mm, Track: {track_width_mm}mm")
        print(f"  Max speed: {max_speed_mm_s}mm/s")
        print(f"  Update rate: {update_rate_hz}Hz")
        if self.hardware_available:
            print(f"  Hardware: Real GPIO detected")
        else:
            print(f"  Hardware: Using mock/simulation mode")

    def _init_motors(self, left_pins: dict, right_pins: dict, standby_pin: int):
        """Initialize motor controllers"""
        try:
            # Create motor controller
            self.motor_controller = DualMotorController(standby_pin)

            # Create individual motors
            self.left_motor = N20Motor(name="Left Motor", **left_pins)
            self.right_motor = N20Motor(name="Right Motor", **right_pins)

            # Assign to controller
            self.motor_controller.motor_a = self.left_motor
            self.motor_controller.motor_b = self.right_motor

            # Reset encoders
            self.left_motor.reset_encoder()
            self.right_motor.reset_encoder()

            print("Motors initialized successfully")

        except Exception as e:
            print(f"CRITICAL: Motor initialization failed: {e}")

            # Check if this is the specific SOC error that needs detailed diagnosis
            if "soc peripheral base address" in str(e).lower():
                print("\nRunning detailed hardware diagnostics...")

                # Try to create a temporary motor instance to get the detailed diagnostics
                try:
                    temp_motor = N20Motor(
                        pwm_pin=left_pins['pwm_pin'],
                        dir1_pin=left_pins['dir1_pin'],
                        dir2_pin=left_pins['dir2_pin'],
                        enc_a_pin=left_pins['enc_a_pin'],
                        enc_b_pin=left_pins['enc_b_pin'],
                        name="Diagnostic Motor"
                    )
                except Exception as motor_error:
                    # This should trigger the detailed diagnostics in motor.py
                    print(f"(Diagnostic attempt also failed, which provided the above details)")

            else:
                # For other errors, show the generic troubleshooting
                print("This could be due to:")
                print("  - Need to run with 'sudo python3 whiteboard_eraser_main.py'")
                print("  - Motor drivers not connected to GPIO pins")
                print("  - Incorrect wiring (check left_motor_pins and right_motor_pins)")
                print("  - Hardware failure in motor drivers or Pi GPIO")
                print("  - Wrong GPIO pin numbers in configuration")

            print("\nCannot operate robot without motors - stopping program")
            raise RuntimeError("Motor hardware initialization failed - robot cannot function")

    def _init_servo(self, servo_pin: int):
        """initialize steering servo"""
        try:
            self.servo = ServoController(servo_pin)
            self.servo.set_angle(90)  # Center position
            print("Servo initialized successfully")
        except Exception as e:
            print(f"CRITICAL: Servo initialization failed: {e}")
            print("This could be due to:")
            print(f"  - Servo not connected to GPIO pin {servo_pin}")
            print("  - Incorrect servo wiring (check power, ground, signal)")
            print("  - GPIO pin conflict or hardware failure")
            print("\nCannot steer robot without servo - stopping program")
            raise RuntimeError("Servo hardware initialization failed - robot cannot steer")

    def _init_gyro(self, sda_pin: int, scl_pin: int):
        """initialize gyroscope"""
        try:
            self.gyro = RCCarGyro(sda_pin=sda_pin, scl_pin=scl_pin)
            self.gyro.calibrate(samples=500, show_progress=False)
            self.gyro.start_continuous_update(update_rate=50)
            print("Gyroscope initialized and calibrated")
        except Exception as e:
            print(f"WARNING: Gyroscope initialization failed: {e}")
            print("This could be due to:")
            print(f"  - IMU not connected to I2C pins (SDA={sda_pin}, SCL={scl_pin})")
            print("  - I2C not enabled (run 'sudo raspi-config' -> Interface Options -> I2C)")
            print("  - Incorrect IMU wiring or hardware failure")
            print("  - Wrong I2C address or conflicting I2C devices")
            print("\nRobot will operate with reduced accuracy (dead reckoning only)")

            class MockGyro:
                def __init__(self):
                    self.angle_z = 0.0
                    self.is_calibrated = True
                def calibrate(self, samples=500, show_progress=False): pass
                def start_continuous_update(self, update_rate=50): pass
                def stop_continuous_update(self): pass
                def get_orientation(self): return (0, 0, self.angle_z)
                def reset_orientation(self): self.angle_z = 0.0

            self.gyro = MockGyro()

    def _init_pid_controllers(self):
        """Initialize PID controllers"""
        # Position PID (output: linear velocity)
        self.position_pid = PIDController(
            kp=0.8,  # Proportional gain
            ki=0.1,  # Integral gain
            kd=0.2,  # Derivative gain
            max_output=1.0,
            min_output=-1.0,
            max_integral=0.5
        )

        # Heading PID (output: angular velocity)
        self.heading_pid = PIDController(
            kp=1.2,
            ki=0.05,
            kd=0.3,
            max_output=1.0,
            min_output=-1.0,
            max_integral=0.3
        )

        # Speed PID controllers for each motor
        self.left_speed_pid = PIDController(
            kp=0.5,
            ki=0.1,
            kd=0.05,
            max_output=1.0,
            min_output=-1.0
        )

        self.right_speed_pid = PIDController(
            kp=0.5,
            ki=0.1,
            kd=0.05,
            max_output=1.0,
            min_output=-1.0
        )

    def start_control_loop(self):
        """Start the main control loop"""
        if self.running:
            return

        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        print("Control loop started")

    def stop_control_loop(self):
        """Stop the main control loop"""
        if not self.running:
            return

        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)

        self.stop_all_motors()
        print("Control loop stopped")

    def _control_loop(self):
        """Main control loop running at specified frequency"""
        last_time = time.time()

        while self.running:
            loop_start = time.time()
            dt = loop_start - last_time

            try:
                # Update odometry
                self._update_odometry(dt)

                # Execute control based on current state
                with self.state_lock:
                    if self.state == CarState.POSITION_CONTROL:
                        self._position_control_update(dt)
                    elif self.state == CarState.MANUAL_CONTROL:
                        pass  # Manual commands handled separately
                    elif self.state == CarState.EMERGENCY_STOP:
                        self.stop_all_motors()

            except Exception as e:
                print(f"Error in control loop: {e}")
                self.emergency_stop()

            # Maintain loop timing
            last_time = loop_start
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.update_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _update_odometry(self, dt: float):
        """Update position and velocity estimates"""
        if dt <= 0:
            return

        # Get encoder readings (cumulative revolutions)
        left_revs = self.left_motor.get_revolutions()
        right_revs = self.right_motor.get_revolutions()

        # Calculate DELTA revolutions since last update
        delta_left_revs = left_revs - self.prev_left_revs
        delta_right_revs = right_revs - self.prev_right_revs

        # Store for next iteration
        self.prev_left_revs = left_revs
        self.prev_right_revs = right_revs

        # Calculate incremental distances traveled this update
        left_distance = delta_left_revs * 2 * np.pi * self.wheel_radius_mm
        right_distance = delta_right_revs * 2 * np.pi * self.wheel_radius_mm

        # For now, use gyro for heading
        try:
            _, _, gyro_heading = self.gyro.get_orientation()
            self.heading = np.radians(gyro_heading)
        except:
            pass  # Keep previous heading if gyro fails

        # Velocity estimation from incremental distance
        forward_distance = (left_distance + right_distance) / 2.0
        self.linear_velocity = forward_distance / dt if dt > 0 else 0

        # Update position using incremental distance
        self.position_x += forward_distance * np.cos(self.heading)
        self.position_y += forward_distance * np.sin(self.heading)

    def _position_control_update(self, dt: float):
        """Update position control"""
        # Calculate position error
        dx = self.target_x - self.position_x
        dy = self.target_y - self.position_y
        distance_error = np.sqrt(dx*dx + dy*dy)

        # Calculate heading error
        target_heading = np.arctan2(dy, dx)
        heading_error = self._normalize_angle(target_heading - self.heading)

        # PID control
        linear_cmd = self.position_pid.update(distance_error, dt)
        angular_cmd = self.heading_pid.update(heading_error, dt)

        # Convert to motor commands
        self._set_velocity_commands(linear_cmd, angular_cmd)

        # Check if we've reached the target
        if distance_error < 20.0:  # 20mm tolerance
            self.stop_all_motors()
            with self.state_lock:
                self.state = CarState.STOPPED

    def set_target_position(self, x: float, y: float, heading: float = None):
        """Set target position for position control"""
        self.target_x = x
        self.target_y = y
        if heading is not None:
            self.target_heading = heading

        # Reset PID controllers
        self.position_pid.reset()
        self.heading_pid.reset()

        with self.state_lock:
            self.state = CarState.POSITION_CONTROL

        print(f"Target set: ({x:.1f}, {y:.1f})")

    def set_manual_control(self, linear_speed: float, angular_speed: float):
        """Set manual velocity commands"""
        with self.state_lock:
            self.state = CarState.MANUAL_CONTROL

        self._set_velocity_commands(linear_speed, angular_speed)

    def _set_velocity_commands(self, linear: float, angular: float):
        """Convert velocity commands to motor speeds"""
        # Clamp inputs
        linear = np.clip(linear, -1.0, 1.0)
        angular = np.clip(angular, -1.0, 1.0)

        # Differential drive kinematics
        # linear: forward speed (-1 to 1)
        # angular: turning speed (-1 to 1, positive = turn right)

        left_speed = linear - angular * 0.5
        right_speed = linear + angular * 0.5

        # Clamp to motor limits
        left_speed = np.clip(left_speed, -1.0, 1.0)
        right_speed = np.clip(right_speed, -1.0, 1.0)

        # Set motor speeds
        self.left_motor.set(left_speed)
        self.right_motor.set(right_speed)

    def set_steering_angle(self, angle_degrees: float):
        """Set front wheel steering angle"""
        # Clamp to servo limits
        angle_degrees = np.clip(angle_degrees, -self.max_steering_angle, self.max_steering_angle)

        # Convert to servo position (90° = center)
        servo_angle = 90 + angle_degrees
        servo_angle = np.clip(servo_angle, 45, 135)  # ±45° from center

        self.servo.set_angle(int(servo_angle))

    def stop_all_motors(self):
        """Stop all motors immediately"""
        try:
            if hasattr(self, 'left_motor') and self.left_motor is not None:
                self.left_motor.stop()
        except Exception:
            pass

        try:
            if hasattr(self, 'right_motor') and self.right_motor is not None:
                self.right_motor.stop()
        except Exception:
            pass

    def emergency_stop(self):
        """Emergency stop - stop all motion and set emergency state"""
        self.emergency_stop_flag = True

        try:
            self.stop_all_motors()
        except Exception:
            pass

        try:
            if hasattr(self, 'servo') and self.servo is not None:
                self.servo.set_angle(90)  # Center steering
        except Exception:
            pass

        with self.state_lock:
            self.state = CarState.EMERGENCY_STOP

        print("EMERGENCY STOP ACTIVATED")

    def reset_emergency_stop(self):
        """Reset emergency stop state"""
        self.emergency_stop_flag = False
        with self.state_lock:
            self.state = CarState.STOPPED
        print("Emergency stop reset")

    def reset_position(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0):
        """Reset current position estimate"""
        self.position_x = x
        self.position_y = y
        self.heading = heading

        # Reset motor encoders
        self.left_motor.reset_encoder()
        self.right_motor.reset_encoder()

        # Reset gyro orientation
        self.gyro.reset_orientation()

        print(f"Position reset to ({x:.1f}, {y:.1f}, {np.degrees(heading):.1f}°)")

    def get_status(self) -> CarStatus:
        """Get current car status"""
        with self.state_lock:
            state = self.state

        # Get steering angle defensively (handle cases where servo might not be fully initialized)
        steering_angle = 0.0
        try:
            if hasattr(self.servo, 'current_angle'):
                steering_angle = self.servo.current_angle - 90
        except Exception:
            pass

        return CarStatus(
            state=state,
            position_x=self.position_x,
            position_y=self.position_y,
            heading=self.heading,
            linear_velocity=self.linear_velocity,
            angular_velocity=self.angular_velocity,
            left_motor_speed=self.left_motor.get_speed(),
            right_motor_speed=self.right_motor.get_speed(),
            steering_angle=steering_angle,  # Convert back to ±degrees
            timestamp=time.time()
        )

    def cleanup(self):
        """Clean up all hardware resources"""
        print("Cleaning up car controller...")

        self.stop_control_loop()

        try:
            self.motor_controller.cleanup()
        except:
            pass

        try:
            self.servo.cleanup()
        except:
            pass

        try:
            self.gyro.stop_continuous_update()
        except:
            pass

        try:
            GPIO.cleanup()
        except:
            pass

        print("Car controller cleanup complete")

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


# Test function
def test_car_controller():
    """Test the car controller"""
    print("Testing CarController...")

    controller = CarController()

    try:
        # Start control loop
        controller.start_control_loop()

        # Test manual control
        print("\nTesting manual control...")
        controller.set_manual_control(0.3, 0.0)  # Move forward slowly
        time.sleep(2)

        controller.set_manual_control(0.0, 0.5)  # Turn right
        time.sleep(1)

        controller.stop_all_motors()
        time.sleep(1)

        # Test position control
        print("\nTesting position control...")
        controller.reset_position(0, 0, 0)
        controller.set_target_position(200, 100)  # Move 200mm forward, 100mm right

        # Monitor for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            status = controller.get_status()
            print(f"Pos: ({status.position_x:.1f}, {status.position_y:.1f}), "
                  f"Target: ({controller.target_x:.1f}, {controller.target_y:.1f}), "
                  f"State: {status.state.value}")

            if status.state == CarState.STOPPED:
                print("Target reached!")
                break

            time.sleep(0.5)

        # Test steering
        print("\nTesting steering...")
        for angle in [-30, -15, 0, 15, 30, 0]:
            controller.set_steering_angle(angle)
            print(f"Steering angle set to {angle}°")
            time.sleep(0.5)

        print("Test complete!")

    except KeyboardInterrupt:
        print("\nTest interrupted")

    finally:
        controller.cleanup()


if __name__ == "__main__":
    test_car_controller()