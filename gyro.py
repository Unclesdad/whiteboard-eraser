#!/usr/bin/env python3
"""
RC Car Gyroscope with Sensor Fusion
Combines gyroscope and accelerometer data for drift-free orientation tracking
Perfect for RC car applications with continuous motion
"""

import time
import math
import threading
try:
    import smbus2 as smbus
except ImportError:
    raise ImportError("smbus2 not installed. Run: pip install smbus2")

class RCCarGyro:
    def __init__(self, sda_pin=2, scl_pin=3, i2c_address=0x68, i2c_bus=1):
        """
        Initialize RC Car Gyroscope with sensor fusion
        
        Args:
            sda_pin (int): GPIO pin for I2C SDA (default: 2)
            scl_pin (int): GPIO pin for I2C SCL (default: 3)
            i2c_address (int): MPU-6050 I2C address (default: 0x68)
            i2c_bus (int): I2C bus number (default: 1)
        """
        
        # Store pin configuration (for reference)
        self.sda_pin = sda_pin
        self.scl_pin = scl_pin
        self.i2c_address = i2c_address
        
        # Initialize I2C communication
        try:
            self.bus = smbus.SMBus(i2c_bus)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize I2C bus {i2c_bus}: {e}")
        
        # MPU-6050 register addresses
        self.PWR_MGMT_1 = 0x6B
        self.SMPLRT_DIV = 0x19
        self.CONFIG = 0x1A
        self.GYRO_CONFIG = 0x1B
        self.ACCEL_CONFIG = 0x1C
        self.ACCEL_XOUT_H = 0x3B
        self.GYRO_XOUT_H = 0x43
        
        # Sensor fusion parameters
        self.alpha = 0.96  # Complementary filter coefficient (trust gyro 96%, accel 4%)
        self.gyro_sensitivity = 131.0  # For ±250°/s range
        self.accel_sensitivity = 16384.0  # For ±2g range
        
        # Orientation tracking
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0
        
        # Calibration data
        self.gyro_offset_x = 0.0
        self.gyro_offset_y = 0.0
        self.gyro_offset_z = 0.0
        self.is_calibrated = False
        
        # Timing
        self.last_time = time.time()
        
        # Thread management
        self.update_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Initialize sensor
        self._initialize_mpu6050()
        
        print(f"RC Car Gyro initialized on I2C bus {i2c_bus}, address 0x{i2c_address:02X}")
        print(f"GPIO pins: SDA={sda_pin}, SCL={scl_pin}")
        
    def _initialize_mpu6050(self):
        """Configure MPU-6050 for optimal RC car performance"""
        try:
            # Wake up MPU-6050
            self.bus.write_byte_data(self.i2c_address, self.PWR_MGMT_1, 0x00)
            time.sleep(0.1)
            
            # Set sample rate to 200Hz (fast enough for RC car)
            self.bus.write_byte_data(self.i2c_address, self.SMPLRT_DIV, 39)  # 1000Hz/(1+39) = 25Hz
            
            # Configure low pass filter (20Hz cutoff for smooth RC car operation)
            self.bus.write_byte_data(self.i2c_address, self.CONFIG, 0x04)
            
            # Set gyroscope range to ±250°/s (good for RC car turns)
            self.bus.write_byte_data(self.i2c_address, self.GYRO_CONFIG, 0x00)
            
            # Set accelerometer range to ±2g (sufficient for RC car)
            self.bus.write_byte_data(self.i2c_address, self.ACCEL_CONFIG, 0x00)
            
            print("MPU-6050 configured for RC car operation")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MPU-6050: {e}")
    
    def _read_word_2c(self, reg):
        """Read 16-bit signed value from register"""
        try:
            high = self.bus.read_byte_data(self.i2c_address, reg)
            low = self.bus.read_byte_data(self.i2c_address, reg + 1)
            value = (high << 8) + low
            return -((65535 - value) + 1) if value >= 0x8000 else value
        except Exception:
            return 0  # Return 0 on read error to prevent crashes
    
    def _get_gyro_data(self):
        """Get calibrated gyroscope data in degrees/second"""
        gyro_x_raw = self._read_word_2c(self.GYRO_XOUT_H)
        gyro_y_raw = self._read_word_2c(self.GYRO_XOUT_H + 2)
        gyro_z_raw = self._read_word_2c(self.GYRO_XOUT_H + 4)
        
        # Convert to degrees/second and apply calibration
        gyro_x = (gyro_x_raw / self.gyro_sensitivity) - self.gyro_offset_x
        gyro_y = (gyro_y_raw / self.gyro_sensitivity) - self.gyro_offset_y
        gyro_z = (gyro_z_raw / self.gyro_sensitivity) - self.gyro_offset_z
        
        return gyro_x, gyro_y, gyro_z
    
    def _get_accel_data(self):
        """Get accelerometer data in g"""
        accel_x_raw = self._read_word_2c(self.ACCEL_XOUT_H)
        accel_y_raw = self._read_word_2c(self.ACCEL_XOUT_H + 2)
        accel_z_raw = self._read_word_2c(self.ACCEL_XOUT_H + 4)
        
        # Convert to g
        accel_x = accel_x_raw / self.accel_sensitivity
        accel_y = accel_y_raw / self.accel_sensitivity
        accel_z = accel_z_raw / self.accel_sensitivity
        
        return accel_x, accel_y, accel_z
    
    def _get_accel_angles(self):
        """Calculate angles from accelerometer (gravity reference)"""
        accel_x, accel_y, accel_z = self._get_accel_data()
        
        # Calculate tilt angles from gravity vector
        # Only valid when not accelerating (good for RC car at steady state)
        angle_x = math.atan2(accel_y, math.sqrt(accel_x**2 + accel_z**2)) * 180 / math.pi
        angle_y = math.atan2(-accel_x, math.sqrt(accel_y**2 + accel_z**2)) * 180 / math.pi
        # Z angle (yaw) cannot be determined from accelerometer alone
        
        return angle_x, angle_y, 0.0
    
    def calibrate(self, samples=1000, show_progress=True):
        """
        Calibrate gyroscope bias (IMPORTANT: Keep RC car stationary!)
        
        Args:
            samples (int): Number of calibration samples
            show_progress (bool): Show calibration progress
        """
        if show_progress:
            print(f"Calibrating gyroscope... Keep RC car STATIONARY!")
            print(f"Taking {samples} samples...")
        
        sum_x = sum_y = sum_z = 0.0
        
        for i in range(samples):
            gyro_x_raw = self._read_word_2c(self.GYRO_XOUT_H)
            gyro_y_raw = self._read_word_2c(self.GYRO_XOUT_H + 2)
            gyro_z_raw = self._read_word_2c(self.GYRO_XOUT_H + 4)
            
            sum_x += gyro_x_raw / self.gyro_sensitivity
            sum_y += gyro_y_raw / self.gyro_sensitivity
            sum_z += gyro_z_raw / self.gyro_sensitivity
            
            if show_progress and i % 100 == 0:
                print(f"  Progress: {i}/{samples}")
            
            time.sleep(0.005)  # 5ms between samples
        
        self.gyro_offset_x = sum_x / samples
        self.gyro_offset_y = sum_y / samples
        self.gyro_offset_z = sum_z / samples
        self.is_calibrated = True
        
        if show_progress:
            print(f"Calibration complete!")
            print(f"Offsets: X={self.gyro_offset_x:.3f}°/s, Y={self.gyro_offset_y:.3f}°/s, Z={self.gyro_offset_z:.3f}°/s")
    
    def _update_orientation(self):
        """Update orientation using complementary filter (sensor fusion)"""
        if not self.is_calibrated:
            return
        
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Skip if dt is too large (first run or long delay)
        if dt > 0.1:
            self.last_time = current_time
            return
        
        # Get sensor data
        gyro_x, gyro_y, gyro_z = self._get_gyro_data()
        accel_angle_x, accel_angle_y, _ = self._get_accel_angles()
        
        with self.lock:
            # Complementary filter for X and Y (pitch and roll)
            # Trust gyro for short-term, accelerometer for long-term
            gyro_angle_x = self.angle_x + gyro_x * dt
            gyro_angle_y = self.angle_y + gyro_y * dt
            
            self.angle_x = self.alpha * gyro_angle_x + (1 - self.alpha) * accel_angle_x
            self.angle_y = self.alpha * gyro_angle_y + (1 - self.alpha) * accel_angle_y
            
            # Z angle (yaw) - only from gyroscope integration
            # For whiteboard car: rotation around X-axis (perpendicular to surface)
            self.angle_z += gyro_x * dt
            
            # Prevent Z angle from growing too large
            if self.angle_z > 180:
                self.angle_z -= 360
            elif self.angle_z < -180:
                self.angle_z += 360
        
        self.last_time = current_time
    
    def start_continuous_update(self, update_rate=50):
        """
        Start continuous orientation updates in background thread
        
        Args:
            update_rate (int): Updates per second (default: 50Hz, good for RC car)
        """
        if self.running:
            print("Continuous update already running")
            return
        
        if not self.is_calibrated:
            print("Warning: Starting updates without calibration. Call calibrate() first for best results.")
        
        self.running = True
        self.update_interval = 1.0 / update_rate
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        print(f"Started continuous orientation updates at {update_rate}Hz")
    
    def _update_loop(self):
        """Background thread for continuous orientation updates"""
        while self.running:
            start_time = time.time()
            
            self._update_orientation()
            
            # Maintain update rate
            elapsed = time.time() - start_time
            sleep_time = self.update_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def stop_continuous_update(self):
        """Stop continuous orientation updates"""
        if self.running:
            self.running = False
            if self.update_thread:
                self.update_thread.join(timeout=1.0)
            print("Stopped continuous orientation updates")
    
    def get_orientation(self):
        """
        Get current orientation angles
        
        Returns:
            tuple: (x_angle, y_angle, z_angle) in degrees
                   x_angle: pitch (nose up/down)
                   y_angle: roll (left/right tilt)  
                   z_angle: yaw (heading/turning)
        """
        if not self.is_calibrated:
            print("Warning: Gyroscope not calibrated. Call calibrate() first.")
        
        with self.lock:
            return self.angle_x, self.angle_y, self.angle_z
    
    def reset_orientation(self):
        """Reset orientation angles to zero (useful when RC car is level)"""
        with self.lock:
            self.angle_x = 0.0
            self.angle_y = 0.0
            self.angle_z = 0.0
        print("Orientation reset to (0°, 0°, 0°)")
    
    def get_raw_data(self):
        """
        Get raw sensor data for debugging
        
        Returns:
            dict: Raw gyroscope and accelerometer data
        """
        gyro_x, gyro_y, gyro_z = self._get_gyro_data()
        accel_x, accel_y, accel_z = self._get_accel_data()
        
        return {
            'gyro': {'x': gyro_x, 'y': gyro_y, 'z': gyro_z},
            'accel': {'x': accel_x, 'y': accel_y, 'z': accel_z},
            'angles': {'x': self.angle_x, 'y': self.angle_y, 'z': self.angle_z}
        }
    
    def is_moving(self, threshold=0.1):
        """
        Detect if RC car is moving (based on accelerometer)
        Useful for determining when to trust accelerometer readings
        
        Args:
            threshold (float): Movement detection threshold in g
            
        Returns:
            bool: True if significant acceleration detected
        """
        accel_x, accel_y, accel_z = self._get_accel_data()
        
        # Calculate total acceleration magnitude
        total_accel = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # If total acceleration is significantly different from 1g, car is accelerating
        return abs(total_accel - 1.0) > threshold

# Example usage and test functions
def test_rc_car_gyro():
    """Test function demonstrating RC car gyro usage"""
    print("RC Car Gyroscope Test")
    print("=" * 40)
    
    # Initialize gyro (default pins: SDA=2, SCL=3)
    gyro = RCCarGyro()
    
    # Calibrate (keep car stationary!)
    gyro.calibrate()
    
    # Start continuous updates
    gyro.start_continuous_update(update_rate=50)  # 50Hz for RC car
    
    try:
        print("\nOrientation tracking started (Ctrl+C to stop)")
        print("Format: X(pitch) Y(roll) Z(yaw)")
        print("-" * 40)
        
        while True:
            x, y, z = gyro.get_orientation()
            moving = gyro.is_moving()
            status = "MOVING" if moving else "STEADY"
            
            print(f"Pitch: {x:6.1f}° | Roll: {y:6.1f}° | Yaw: {z:6.1f}° | {status}")
            time.sleep(0.2)  # Update display every 200ms
            
    except KeyboardInterrupt:
        print("\nStopping test...")
    
    finally:
        gyro.stop_continuous_update()
        print("Test complete!")

if __name__ == "__main__":
    test_rc_car_gyro()