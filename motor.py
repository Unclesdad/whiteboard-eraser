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

import RPi.GPIO as GPIO
import threading
import time
import os
import re

class N20Motor:
    def __init__(self, pwm_pin, dir1_pin, dir2_pin, enc_a_pin, enc_b_pin, 
                 pwm_frequency=1000, name="Motor"):
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
        """
        self.pwm_pin = pwm_pin
        self.dir1_pin = dir1_pin
        self.dir2_pin = dir2_pin
        self.enc_a_pin = enc_a_pin
        self.enc_b_pin = enc_b_pin
        self.name = name
        
        # Encoder state
        self.encoder_count = 0
        self.last_a_state = 0
        self.last_b_state = 0
        self._encoder_lock = threading.Lock()
        
        # Motor state
        self.current_speed = 0.0  # -1.0 to 1.0
        
        # Setup GPIO with diagnostics
        self._setup_gpio(pwm_frequency)
        
        # Setup encoder interrupts
        self._setup_encoder_interrupts()
        
    def _setup_gpio(self, pwm_frequency):
        """Setup GPIO pins for motor control with hardware diagnostics"""
        print(f"🔧 Setting up GPIO for {self.name}...")

        # Add hardware diagnostics
        self._print_hardware_info()

        try:
            # Setup motor control pins
            print(f"  Setting up motor pins: PWM={self.pwm_pin}, DIR1={self.dir1_pin}, DIR2={self.dir2_pin}")
            GPIO.setup(self.pwm_pin, GPIO.OUT)
            GPIO.setup(self.dir1_pin, GPIO.OUT)
            GPIO.setup(self.dir2_pin, GPIO.OUT)

            # Setup PWM
            print(f"  Starting PWM at {pwm_frequency}Hz on pin {self.pwm_pin}")
            self.pwm = GPIO.PWM(self.pwm_pin, pwm_frequency)
            self.pwm.start(0)

            # Setup encoder pins
            print(f"  Setting up encoder pins: A={self.enc_a_pin}, B={self.enc_b_pin}")
            GPIO.setup(self.enc_a_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(self.enc_b_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            # Read initial encoder states
            self.last_a_state = GPIO.input(self.enc_a_pin)
            self.last_b_state = GPIO.input(self.enc_b_pin)
            print(f"  Initial encoder states: A={self.last_a_state}, B={self.last_b_state}")

            print(f"✓ {self.name} GPIO setup complete")

        except Exception as e:
            print(f"❌ GPIO setup failed for {self.name}: {e}")
            print("🔍 Hardware diagnosis:")
            self._diagnose_gpio_failure(e)

            # Try fallback methods
            if "soc peripheral base address" in str(e).lower():
                fallback_success = self._try_fallback_gpio_init()
                if not fallback_success:
                    print("\n🛑 All GPIO initialization methods failed")
                    print("🔧 Manual steps to try:")
                    print("   1. pip3 install --upgrade RPi.GPIO")
                    print("   2. pip3 install gpiozero (modern alternative)")
                    print("   3. Check Pi model compatibility")
                    print("   4. Verify hardware connections")

            raise
        
    def _setup_encoder_interrupts(self):
        """Setup interrupt handlers for encoder"""
        # Remove any existing event detection first
        try:
            GPIO.remove_event_detect(self.enc_a_pin)
            GPIO.remove_event_detect(self.enc_b_pin)
        except:
            pass
        
        # Add event detection with minimal bouncetime for better responsiveness
        GPIO.add_event_detect(self.enc_a_pin, GPIO.BOTH, 
                            callback=self._encoder_callback, bouncetime=1)
        GPIO.add_event_detect(self.enc_b_pin, GPIO.BOTH, 
                            callback=self._encoder_callback, bouncetime=1)
    
    def _encoder_callback(self, channel):
        """
        Encoder interrupt callback for quadrature decoding
        Uses standard quadrature decoding logic with improved state tracking
        """
        with self._encoder_lock:
            a_state = GPIO.input(self.enc_a_pin)
            b_state = GPIO.input(self.enc_b_pin)
            
            # Only process if this is actually a state change
            if channel == self.enc_a_pin and a_state != self.last_a_state:
                # A channel changed
                if (self.last_a_state == 0 and a_state == 1):
                    # Rising edge on A
                    if b_state == 0:
                        self.encoder_count += 1  # Forward
                    else:
                        self.encoder_count -= 1  # Reverse
                elif (self.last_a_state == 1 and a_state == 0):
                    # Falling edge on A
                    if b_state == 1:
                        self.encoder_count += 1  # Forward
                    else:
                        self.encoder_count -= 1  # Reverse
                self.last_a_state = a_state
                
            elif channel == self.enc_b_pin and b_state != self.last_b_state:
                # B channel changed
                if (self.last_b_state == 0 and b_state == 1):
                    # Rising edge on B
                    if a_state == 1:
                        self.encoder_count += 1  # Forward
                    else:
                        self.encoder_count -= 1  # Reverse
                elif (self.last_b_state == 1 and b_state == 0):
                    # Falling edge on B
                    if a_state == 0:
                        self.encoder_count += 1  # Forward
                    else:
                        self.encoder_count -= 1  # Reverse
                self.last_b_state = b_state
    
    def set(self, speed):
        """
        Set motor speed and direction
        
        Args:
            speed: Float from -1.0 to 1.0
                  -1.0 = full speed reverse
                   0.0 = stop
                   1.0 = full speed forward
        """
        # Clamp speed to valid range
        speed = max(-1.0, min(1.0, speed))
        self.current_speed = speed
        
        # Calculate PWM duty cycle (0-100%)
        duty_cycle = abs(speed) * 100
        
        if speed > 0:
            # Forward direction
            GPIO.output(self.dir1_pin, GPIO.HIGH)
            GPIO.output(self.dir2_pin, GPIO.LOW)
        elif speed < 0:
            # Reverse direction
            GPIO.output(self.dir1_pin, GPIO.LOW)
            GPIO.output(self.dir2_pin, GPIO.HIGH)
        else:
            # Stop (brake)
            GPIO.output(self.dir1_pin, GPIO.LOW)
            GPIO.output(self.dir2_pin, GPIO.LOW)
        
        # Set PWM duty cycle
        self.pwm.ChangeDutyCycle(duty_cycle)
    
    def get_encoder_count(self):
        """
        Get current encoder count
        
        Returns:
            int: Current encoder count (can be negative)
        """
        with self._encoder_lock:
            return self.encoder_count
    
    def reset_encoder(self):
        """Reset encoder count to zero"""
        with self._encoder_lock:
            self.encoder_count = 0
    
    def get_revolutions(self):
        """
        Get motor shaft revolutions based on encoder count
        
        Returns:
            float: Number of revolutions (28 counts = 1 revolution for 7 PPR quadrature)
        """
        with self._encoder_lock:
            return self.encoder_count / 28.0
    
    def get_speed(self):
        """
        Get current speed setting
        
        Returns:
            float: Current speed from -1.0 to 1.0
        """
        return self.current_speed
    
    def get_encoder_states(self):
        """
        Get current encoder pin states for debugging
        
        Returns:
            tuple: (A_state, B_state, last_A_state, last_B_state)
        """
        with self._encoder_lock:
            current_a = GPIO.input(self.enc_a_pin)
            current_b = GPIO.input(self.enc_b_pin)
            return (current_a, current_b, self.last_a_state, self.last_b_state)
    
    def get_status(self):
        """
        Get comprehensive motor status for debugging
        
        Returns:
            dict: Motor status information
        """
        a_state, b_state, last_a, last_b = self.get_encoder_states()
        return {
            'name': self.name,
            'speed': self.current_speed,
            'encoder_count': self.get_encoder_count(),
            'revolutions': self.get_revolutions(),
            'encoder_a_current': a_state,
            'encoder_b_current': b_state,
            'encoder_a_last': last_a,
            'encoder_b_last': last_b
        }
    
    def stop(self):
        """Stop the motor"""
        self.set(0)
    
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
        """Clean up GPIO and PWM"""
        try:
            self.stop()
            time.sleep(0.1)  # Give time for PWM to stop
            self.pwm.stop()
        except:
            pass

        try:
            GPIO.remove_event_detect(self.enc_a_pin)
        except:
            pass
        try:
            GPIO.remove_event_detect(self.enc_b_pin)
        except:
            pass


class DualMotorController:
    """Controller for two N20 motors with standby control"""
    
    def __init__(self, standby_pin):
        """
        Initialize dual motor controller
        
        Args:
            standby_pin: GPIO pin for TB6612FNG standby control
        """
        self.standby_pin = standby_pin
        
        # Setup standby pin
        GPIO.setup(self.standby_pin, GPIO.OUT)
        self.enable()
        
        # Motor instances (to be set by user)
        self.motor_a = None
        self.motor_b = None
    
    def enable(self):
        """Enable motor driver (standby HIGH)"""
        GPIO.output(self.standby_pin, GPIO.HIGH)
    
    def disable(self):
        """Disable motor driver (standby LOW)"""
        GPIO.output(self.standby_pin, GPIO.LOW)
    
    def cleanup(self):
        """Clean up all motors and GPIO"""
        try:
            if self.motor_a:
                self.motor_a.cleanup()
            if self.motor_b:
                self.motor_b.cleanup()
            time.sleep(0.1)
            self.disable()
        except Exception as e:
            print(f"Warning during cleanup: {e}")
            pass


# Example usage based on your wiring configuration
if __name__ == "__main__":
    # Initialize GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
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
        GPIO.cleanup()
        print("Done")