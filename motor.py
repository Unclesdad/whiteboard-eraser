#!/usr/bin/env python3
"""
N20 Motor Controller with Magnetic Encoder
For CHF-GM12-N20VA-298-6V+ABHL motors with TB6612FNG driver

Motor Specifications:
- 7 PPR (pulses per revolution) = 28 counts per revolution in quadrature
- 6V rated voltage
- 298:1 gear ratio
- Magnetic Hall encoder with A/B phases
"""

from gpiozero import PWMOutputDevice, OutputDevice, InputDevice, Device
from gpiozero.pins.pigpio import PiGPIOFactory
import threading
import time
import os
import re
import atexit
import signal
import sys
import numpy as np

# Use pigpio for better performance and Pi 5 compatibility
try:
    Device.pin_factory = PiGPIOFactory()
    print("✓ Using pigpio pin factory for better performance")
except Exception as e:
    print(f"⚠️  pigpio not available ({e}), using default pin factory")
    # Default pin factory will be used

class N20Motor:
    # Class-level shared resources for high-speed encoder management
    _encoder_thread = None
    _encoder_positions = {}
    _encoder_pins = {}
    _encoder_devices = {}  # Store gpiozero InputDevice objects
    _encoder_last_states = {}
    _running = False
    _lock = threading.Lock()
    _motor_count = 0
    _polling_freq = 7500  # 7.5 kHz polling frequency for reliable encoder reading

    def __init__(self, pwm_pin, dir1_pin, dir2_pin, enc_a_pin, enc_b_pin,
                 pwm_frequency=1000, name="Motor", reverse_encoder=False, use_interrupts=None):
        """
        Initialize N20 motor with encoder

        Args:
            pwm_pin: GPIO pin for PWM speed control
            dir1_pin: GPIO pin for direction control 1
            dir2_pin: GPIO pin for direction control 2
            enc_a_pin: GPIO pin for encoder channel A
            enc_b_pin: GPIO pin for encoder channel B
            pwm_frequency: PWM frequency in Hz (default 1000)
            name: Motor name for identification
            reverse_encoder: If True, reverse encoder direction (backward compatibility)
            use_interrupts: Ignored - always uses high-speed polling (backward compatibility)
        """
        self.pwm_pin = pwm_pin
        self.dir1_pin = dir1_pin
        self.dir2_pin = dir2_pin
        self.enc_a_pin = enc_a_pin
        self.enc_b_pin = enc_b_pin
        self.name = name
        self.reverse_encoder = reverse_encoder

        # Register this motor with shared encoder system
        with N20Motor._lock:
            self.motor_id = N20Motor._motor_count
            N20Motor._motor_count += 1
            N20Motor._encoder_pins[self.motor_id] = (enc_a_pin, enc_b_pin)
            N20Motor._encoder_positions[self.motor_id] = 0

            # Create encoder input devices immediately
            from gpiozero import InputDevice
            N20Motor._encoder_devices[self.motor_id] = {
                'pin_a': InputDevice(enc_a_pin, pull_up=True),
                'pin_b': InputDevice(enc_b_pin, pull_up=True)
            }

        # Motor state
        self.current_speed = 0.0  # -1.0 to 1.0

        # Backward compatibility info
        if use_interrupts is not None:
            print(f"  Note: use_interrupts parameter ignored - using high-speed polling at {N20Motor._polling_freq}Hz")
        if reverse_encoder:
            print(f"  {self.name}: Encoder direction reversed")

        # Initialize gpiozero devices
        self.pwm_device = None
        self.dir1_device = None
        self.dir2_device = None
        self.enc_a_device = None
        self.enc_b_device = None

        # Setup GPIO with diagnostics
        self._setup_gpio(pwm_frequency)

        # Start shared encoder thread if this is the first motor (moved after device creation)
        with N20Motor._lock:
            if not N20Motor._running:
                N20Motor._start_encoder_thread()

        # Setup cleanup handlers
        self._setup_cleanup_handlers()
        
    def _setup_gpio(self, pwm_frequency):
        """Setup GPIO pins for motor control using gpiozero"""
        print(f"🔧 Setting up gpiozero devices for {self.name}...")

        try:
            # Setup motor control devices
            print(f"  Setting up motor pins: PWM={self.pwm_pin}, DIR1={self.dir1_pin}, DIR2={self.dir2_pin}")

            # Create PWM device for speed control
            self.pwm_device = PWMOutputDevice(self.pwm_pin, frequency=pwm_frequency)

            # Create output devices for direction control
            self.dir1_device = OutputDevice(self.dir1_pin)
            self.dir2_device = OutputDevice(self.dir2_pin)

            # Initialize to stopped state
            self.pwm_device.value = 0
            self.dir1_device.off()
            self.dir2_device.off()

            print(f"  PWM device created at {pwm_frequency}Hz on pin {self.pwm_pin}")

            # Note: Encoder pins will be setup by shared encoder thread
            print(f"  Encoder pins A={self.enc_a_pin}, B={self.enc_b_pin} registered for high-speed polling")

            print(f"✓ {self.name} gpiozero setup complete")

        except Exception as e:
            print(f"❌ gpiozero setup failed for {self.name}: {e}")
            print("🔧 This could be due to:")
            print("   - gpiozero not installed: pip3 install gpiozero")
            print("   - GPIO pins already in use")
            print("   - Hardware connection issues")
            print("   - Insufficient permissions (try with sudo)")

            # Add hardware diagnostics
            self._print_hardware_info()

            raise
        
    @classmethod
    def _start_encoder_thread(cls):
        """Start the shared encoder polling thread at 7500Hz."""
        cls._running = True
        cls._encoder_thread = threading.Thread(target=cls._encoder_loop, daemon=True)
        cls._encoder_thread.start()
        print(f"✓ High-speed encoder polling started at {cls._polling_freq}Hz")

    @classmethod
    def _encoder_loop(cls):
        """
        High-speed encoder polling loop running at 7500Hz using gpiozero.
        Monitors all motor encoders simultaneously using quadrature decoding.
        Uses pre-created InputDevice objects for maximum speed.
        """
        print(f"✓ Encoder polling thread started for gpiozero devices")

        polling_interval = 1.0 / cls._polling_freq
        next_poll = time.perf_counter()

        while cls._running:
            # Ultra-fast encoder reading for all motors using pre-created gpiozero devices
            for motor_id, encoder_devices in cls._encoder_devices.items():
                a_state = encoder_devices['pin_a'].value
                b_state = encoder_devices['pin_b'].value

                # Initialize last state if this is a new motor
                if motor_id not in cls._encoder_last_states:
                    cls._encoder_last_states[motor_id] = a_state
                    continue

                last_a = cls._encoder_last_states[motor_id]

                # Process encoder changes using optimized quadrature decoding (exact same logic as working version)
                if a_state != last_a:
                    if a_state == b_state:
                        cls._encoder_positions[motor_id] += 1
                    else:
                        cls._encoder_positions[motor_id] -= 1
                    cls._encoder_last_states[motor_id] = a_state

            # Precise timing control (exact same as working version)
            next_poll += polling_interval
            sleep_time = next_poll - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)

    def _setup_cleanup_handlers(self):
        """Setup cleanup handlers for graceful shutdown"""
        def cleanup_handler(signum, frame):
            N20Motor._running = False
            if N20Motor._encoder_thread:
                N20Motor._encoder_thread.join(timeout=1.0)
            # gpiozero devices will be cleaned up by their own close() methods
            sys.exit(0)

        signal.signal(signal.SIGINT, cleanup_handler)
        atexit.register(lambda: setattr(N20Motor, '_running', False))
    
    def set(self, power):
        """
        Set motor power and direction using gpiozero devices.

        Args:
            power: Motor power from -1.0 to 1.0 (negative = reverse)
        """
        # Clamp power to valid range
        power = min(max(power, -1), 1)
        self.current_speed = power

        # Determine direction using numpy sign
        sign = np.sign(power)

        if sign == 1:
            # Forward direction
            self.dir1_device.on()
            self.dir2_device.off()
        elif sign == -1:
            # Reverse direction
            self.dir1_device.off()
            self.dir2_device.on()
        else:
            # Stop (brake)
            self.dir1_device.off()
            self.dir2_device.off()

        # Set PWM value (0.0 to 1.0)
        self.pwm_device.value = abs(power)
    
    def get_encoder_count(self):
        """
        Get current encoder position from shared high-speed polling system.

        Returns:
            int: Current encoder count (positive = forward, negative = reverse)
        """
        with N20Motor._lock:
            count = N20Motor._encoder_positions.get(self.motor_id, 0)
            return -count if self.reverse_encoder else count

    def reset_encoder(self):
        """Reset encoder position to zero."""
        with N20Motor._lock:
            N20Motor._encoder_positions[self.motor_id] = 0

    def get_position(self):
        """Alias for get_encoder_count for compatibility with old interface"""
        return self.get_encoder_count()

    def reset_position(self):
        """Alias for reset_encoder for compatibility with old interface"""
        return self.reset_encoder()
    
    def get_revolutions(self):
        """
        Get motor shaft revolutions based on encoder count

        Returns:
            float: Number of revolutions (28 counts = 1 revolution for 7 PPR quadrature)
        """
        return self.get_encoder_count() / 28.0

    def get_speed_cps(self, window_time=0.5):
        """
        Calculate motor speed by measuring position change over time.

        Args:
            window_time: Time window in seconds for speed calculation

        Returns:
            float: Speed in encoder counts per second
        """
        start_pos = self.get_encoder_count()
        time.sleep(window_time)
        end_pos = self.get_encoder_count()
        return (end_pos - start_pos) / window_time
    
    def get_speed(self):
        """
        Get current speed setting
        
        Returns:
            float: Current speed from -1.0 to 1.0
        """
        return self.current_speed
    
    def get_encoder_states(self):
        """
        Get current encoder pin states for debugging (gpiozero version)

        Returns:
            tuple: (A_state, B_state, last_A_state, encoder_position)
        """
        # Create temporary input devices to read current states
        try:
            temp_a = InputDevice(self.enc_a_pin, pull_up=True)
            temp_b = InputDevice(self.enc_b_pin, pull_up=True)

            current_a = temp_a.value
            current_b = temp_b.value

            temp_a.close()
            temp_b.close()
        except:
            # If we can't read the pins, return placeholder values
            current_a = 0
            current_b = 0

        with N20Motor._lock:
            last_a = N20Motor._encoder_last_states.get(self.motor_id, 0)
            position = N20Motor._encoder_positions.get(self.motor_id, 0)

        return (current_a, current_b, last_a, position)
    
    def get_status(self):
        """
        Get comprehensive motor status for debugging (updated for shared system)

        Returns:
            dict: Motor status information
        """
        a_state, b_state, last_a, position = self.get_encoder_states()
        return {
            'name': self.name,
            'motor_id': self.motor_id,
            'speed': self.current_speed,
            'encoder_count': self.get_encoder_count(),
            'revolutions': self.get_revolutions(),
            'encoder_a_current': a_state,
            'encoder_b_current': b_state,
            'encoder_a_last': last_a,
            'position_from_shared': position,
            'polling_frequency': N20Motor._polling_freq,
            # Backward compatibility fields for forward_then_backward.py
            'encoder_method': 'high-speed-polling',
            'active': N20Motor._running,
            'reverse_encoder': self.reverse_encoder
        }
    
    def stop(self):
        """Stop the motor"""
        try:
            # Check if devices are still valid before trying to stop
            if (hasattr(self, 'pwm_device') and self.pwm_device is not None and
                hasattr(self, 'dir1_device') and self.dir1_device is not None and
                hasattr(self, 'dir2_device') and self.dir2_device is not None):
                self.set(0)
        except Exception:
            # Ignore errors during stop - we're likely in cleanup
            pass
    
    def _print_hardware_info(self):
        """Print Raspberry Pi hardware information for diagnostics"""
        print("🔍 Raspberry Pi Hardware Detection:")

        # Try to read Pi model from device tree
        pi_model = self._get_pi_model()
        print(f"  Pi Model: {pi_model}")

        # Check RPi.GPIO version
        try:
            import RPi.GPIO as GPIO_check
            gpio_version = getattr(GPIO_check, 'VERSION', 'Unknown')
            print(f"  RPi.GPIO Version: {gpio_version}")
        except:
            print(f"  RPi.GPIO Version: Detection failed")

        # Check kernel version
        try:
            with open('/proc/version', 'r') as f:
                kernel = f.read().split()[2]
                print(f"  Kernel: {kernel}")
        except:
            print(f"  Kernel: Detection failed")

    def _get_pi_model(self):
        """Detect Raspberry Pi model"""
        try:
            # First try device tree
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    return f.read().strip('\x00')
        except:
            pass

        try:
            # Fallback to /proc/cpuinfo
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('Model'):
                        return line.split(':', 1)[1].strip()
        except:
            pass

        return "Unknown Pi Model"

    def _diagnose_gpio_failure(self, error):
        """Provide specific diagnosis for GPIO failures"""
        error_str = str(error).lower()

        if "soc peripheral base address" in error_str:
            print("  ⚠️  SOC peripheral base address error detected")
            print("  🔧 This means RPi.GPIO can't detect your Pi hardware")
            print("  💡 Possible solutions:")
            print("     - Update RPi.GPIO: pip3 install --upgrade RPi.GPIO")
            print("     - Check Pi model compatibility with RPi.GPIO version")
            print("     - Try alternative: pip3 install gpiozero")
            print("     - Verify /proc/cpuinfo and /proc/device-tree/model exist")

            # Show what hardware was detected
            pi_model = self._get_pi_model()
            print(f"     - Detected model: {pi_model}")

            # Check if it's a known incompatible combination
            if "pi 5" in pi_model.lower():
                print("     ⚠️  Raspberry Pi 5 requires RPi.GPIO >= 0.7.1 or gpiozero")
            elif "pi zero 2" in pi_model.lower():
                print("     ⚠️  Pi Zero 2 W requires recent RPi.GPIO version")

        elif "permission denied" in error_str:
            print("  ⚠️  Permission denied - need sudo access")
            print("  🔧 Run with: sudo python3 your_script.py")

        elif "device or resource busy" in error_str:
            print("  ⚠️  GPIO pins already in use by another process")
            print("  🔧 Check for other running programs using GPIO")

        else:
            print(f"  ❓ Unrecognized error pattern: {error}")
            print("  🔧 Try checking wiring and pin numbers")

    def _try_fallback_gpio_init(self):
        """Try alternative GPIO initialization methods"""
        print("🔄 Attempting fallback GPIO initialization...")

        # Method 1: Try importing gpiozero as alternative
        try:
            print("  Trying gpiozero library...")
            import gpiozero
            print("  ✓ gpiozero available - consider switching to gpiozero for better compatibility")
            # Don't actually switch here, just report availability
            return False  # Still want to try other RPi.GPIO fixes
        except ImportError:
            print("  ⚠️  gpiozero not available")

        # Method 2: Try manual Pi detection and GPIO base setup
        pi_model = self._get_pi_model()
        print(f"  Attempting manual GPIO setup for: {pi_model}")

        if "pi 5" in pi_model.lower():
            print("  ⚠️  Raspberry Pi 5 detected - RPi.GPIO may not be fully supported")
            print("  💡 Consider using gpiozero: pip3 install gpiozero")
            return False

        # Method 3: Check if we can read basic Pi info that RPi.GPIO needs
        try:
            # Check /proc/cpuinfo for hardware info RPi.GPIO uses
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'Hardware' in cpuinfo:
                    hardware_line = [line for line in cpuinfo.split('\n') if line.startswith('Hardware')]
                    if hardware_line:
                        hardware = hardware_line[0].split(':')[1].strip()
                        print(f"  Hardware identifier: {hardware}")

                        # Common hardware identifiers
                        if hardware in ['BCM2835', 'BCM2836', 'BCM2837', 'BCM2711']:
                            print(f"  ✓ Recognized hardware: {hardware}")
                            return True  # This should work with RPi.GPIO
                        else:
                            print(f"  ⚠️  Unrecognized hardware: {hardware}")

        except Exception as e:
            print(f"  ❌ Failed to read hardware info: {e}")

        print("  ❌ No successful fallback method found")
        return False

    def cleanup(self):
        """Clean up gpiozero devices"""
        try:
            self.stop()
            time.sleep(0.1)  # Give time for devices to stop
        except Exception:
            pass

        # Close gpiozero devices
        try:
            if hasattr(self, 'pwm_device') and self.pwm_device is not None:
                self.pwm_device.close()
                self.pwm_device = None
        except Exception:
            pass

        try:
            if hasattr(self, 'dir1_device') and self.dir1_device is not None:
                self.dir1_device.close()
                self.dir1_device = None
        except Exception:
            pass

        try:
            if hasattr(self, 'dir2_device') and self.dir2_device is not None:
                self.dir2_device.close()
                self.dir2_device = None
        except Exception:
            pass

        # Close encoder devices
        try:
            if self.motor_id in N20Motor._encoder_devices:
                encoder_devs = N20Motor._encoder_devices[self.motor_id]
                encoder_devs['pin_a'].close()
                encoder_devs['pin_b'].close()
                del N20Motor._encoder_devices[self.motor_id]
        except:
            pass

        # Encoder cleanup is handled by shared thread


class DualMotorController:
    """Controller for two N20 motors with standby control"""
    
    def __init__(self, standby_pin):
        """
        Initialize dual motor controller using gpiozero

        Args:
            standby_pin: GPIO pin for TB6612FNG standby control
        """
        self.standby_pin = standby_pin

        # Setup standby pin with gpiozero
        try:
            self.standby_device = OutputDevice(standby_pin)
            self.enable()
            print(f"✓ Motor controller standby pin {standby_pin} configured with gpiozero")
        except Exception as e:
            print(f"❌ CRITICAL: Motor controller standby pin setup failed: {e}")
            print("🔧 This could be due to:")
            print("   - gpiozero not installed: pip3 install gpiozero")
            print("   - GPIO pin already in use")
            print("   - Insufficient permissions")
            print("\n🛑 Cannot operate motors without standby pin control")
            raise RuntimeError("Motor controller gpiozero initialization failed")

        # Motor instances (to be set by user)
        self.motor_a = None
        self.motor_b = None
    
    def enable(self):
        """Enable motor driver (standby HIGH)"""
        self.standby_device.on()

    def disable(self):
        """Disable motor driver (standby LOW)"""
        self.standby_device.off()
    
    def cleanup(self):
        """Clean up all motors and gpiozero devices"""
        try:
            if self.motor_a:
                self.motor_a.cleanup()
            if self.motor_b:
                self.motor_b.cleanup()
            time.sleep(0.1)
            self.disable()

            # Close standby device
            if hasattr(self, 'standby_device'):
                self.standby_device.close()
        except Exception as e:
            print(f"Warning during cleanup: {e}")
            pass


class EncoderManager:
    """
    Backward compatibility class for old motor code that expects EncoderManager.
    The new implementation integrates encoder management into N20Motor class-level methods.
    """

    @staticmethod
    def get_polling_frequency():
        """Get the current encoder polling frequency"""
        return N20Motor._polling_freq

    @staticmethod
    def is_running():
        """Check if encoder polling thread is running"""
        return N20Motor._running

    @staticmethod
    def get_motor_count():
        """Get number of registered motors"""
        with N20Motor._lock:
            return N20Motor._motor_count

    @staticmethod
    def get_all_positions():
        """Get positions of all registered motors"""
        with N20Motor._lock:
            return N20Motor._encoder_positions.copy()

    def __init__(self):
        """Initialize encoder manager (for backward compatibility)"""
        print("⚠️  EncoderManager is deprecated - encoder management is now built into N20Motor")
        print(f"✓ High-speed encoder polling active at {N20Motor._polling_freq}Hz")


# Example usage based on your wiring configuration
if __name__ == "__main__":
    # gpiozero handles GPIO initialization automatically
    
    try:
        # Create dual motor controller with standby control
        controller = DualMotorController(standby_pin=22)
        
        # Create Motor A (your motor 1)
        motor_a = N20Motor(
            pwm_pin=18,    # PWMA
            dir1_pin=23,   # AIN1
            dir2_pin=24,   # AIN2
            enc_a_pin=17,  # Motor 1 Encoder A
            enc_b_pin=27,  # Motor 1 Encoder B
            name="Motor A"
        )
        
        # Create Motor B (your motor 2)
        motor_b = N20Motor(
            pwm_pin=19,    # PWMB
            dir1_pin=25,   # BIN1
            dir2_pin=8,    # BIN2
            enc_a_pin=5,   # Motor 2 Encoder A
            enc_b_pin=6,   # Motor 2 Encoder B
            name="Motor B"
        )
        
        controller.motor_a = motor_a
        controller.motor_b = motor_b
        
        print("Motor test starting...")
        
        # Reset encoders before testing
        motor_a.reset_encoder()
        motor_b.reset_encoder()
        
        try:
            # Check initial encoder states
            print(f"Initial Motor A encoder states: {motor_a.get_encoder_states()}")
            print(f"Initial Motor B encoder states: {motor_b.get_encoder_states()}")
            
            print("\nTesting Motor A forward at 50%")
            motor_a.set(0.5)
            
            # Monitor encoder counts during movement
            start_time = time.time()
            while time.time() - start_time < 3:  # Extended test time
                time.sleep(0.5)
                count = motor_a.get_encoder_count()
                states = motor_a.get_encoder_states()
                print(f"  Time: {time.time() - start_time:.1f}s, Count: {count}, States: A={states[0]}, B={states[1]}")
            
            motor_a.stop()
            final_count_a = motor_a.get_encoder_count()
            print(f"Motor A final count: {final_count_a}, revolutions: {motor_a.get_revolutions():.2f}")
            
            time.sleep(1)
            
            print("\nTesting Motor B forward at 50%")
            motor_b.reset_encoder()
            motor_b.set(0.5)
            
            # Monitor encoder counts during movement
            start_time = time.time()
            while time.time() - start_time < 3:
                time.sleep(0.5)
                count = motor_b.get_encoder_count()
                states = motor_b.get_encoder_states()
                print(f"  Time: {time.time() - start_time:.1f}s, Count: {count}, States: A={states[0]}, B={states[1]}")
            
            motor_b.stop()
            final_count_b = motor_b.get_encoder_count()
            print(f"Motor B final count: {final_count_b}, revolutions: {motor_b.get_revolutions():.2f}")
            
            time.sleep(1)
            
            # Test direction reversal
            print("\nTesting Motor A reverse at 50%")
            motor_a.reset_encoder()
            motor_a.set(-0.5)
            time.sleep(2)
            motor_a.stop()
            reverse_count_a = motor_a.get_encoder_count()
            print(f"Motor A reverse count: {reverse_count_a}, revolutions: {motor_a.get_revolutions():.2f}")
            
            # Final status check
            print("\nFinal Status:")
            print(f"Motor A: {motor_a.get_status()}")
            print(f"Motor B: {motor_b.get_status()}")
            
        except Exception as test_error:
            print(f"Error during test: {test_error}")
            motor_a.stop()
            motor_b.stop()
    
    except KeyboardInterrupt:
        print("\nTest interrupted")
    
    finally:
        # Clean up
        print("Cleaning up...")
        controller.cleanup()
        print("Done")