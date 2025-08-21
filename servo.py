#!/usr/bin/env python3

import atexit
import RPi.GPIO as GPIO
import time
import signal
import sys

class ServoController:
    def __init__(self, pwm_pin:int):
        """
        Initialize servo controller.
        
        Args:
            pwm_pin: GPIO pin for PWM control (default: 12)
            
        Wiring:
            Brown wire → Pi GND (Pin 39)
            Red wire → Pi 5V (Pin 2)  
            Yellow wire → GPIO pin for PWM control
        """
        try:
            GPIO.cleanup()
        except:
            pass
            
        self.PWM_PIN = pwm_pin
        self.setup_gpio()
        
        # Standard servo: 50Hz PWM, 1-2ms pulse width for 0-180 degrees
        self.pwm = GPIO.PWM(self.PWM_PIN, 50)  # 50Hz frequency
        self.pwm.start(0)
        
        # Set to neutral position (90 degrees)
        self.set_angle(90)
        
        # Setup cleanup handlers
        def cleanup_handler(signum, frame):
            self.cleanup()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, cleanup_handler)
        atexit.register(self.cleanup)
    
    def setup_gpio(self):
        """Configure GPIO pins for servo control."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.PWM_PIN, GPIO.OUT)
    
    def set_angle(self, angle):
        """
        Set servo angle.
        
        Args:
            angle: Servo angle from 0 to 180 degrees
        """
        # Clamp angle to valid range
        angle = max(0, min(180, angle))
        
        # Convert angle to duty cycle
        # 0 degrees = 1ms pulse = 2% duty cycle at 50Hz
        # 180 degrees = 2ms pulse = 10% duty cycle at 50Hz
        # Linear interpolation: duty = 2 + (angle/180) * 8
        duty_cycle = 2 + (angle / 180) * 8
        
        self.pwm.ChangeDutyCycle(duty_cycle)
        
        # Brief delay to allow servo to move
        time.sleep(0.1)
        
    def set_pulse_width(self, pulse_ms):
        """
        Set servo position using pulse width in milliseconds.
        
        Args:
            pulse_ms: Pulse width from 1.0 to 2.0 ms
        """
        # Clamp pulse width to safe range
        pulse_ms = max(1.0, min(2.0, pulse_ms))
        
        # Convert pulse width to duty cycle at 50Hz
        # duty_cycle = (pulse_ms / 20ms) * 100%
        duty_cycle = (pulse_ms / 20.0) * 100
        
        self.pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.1)
    
    def get_angle_from_pulse(self, pulse_ms):
        """
        Convert pulse width to angle.
        
        Args:
            pulse_ms: Pulse width in milliseconds
            
        Returns:
            float: Corresponding angle in degrees
        """
        return (pulse_ms - 1.0) * 180.0
    
    def sweep(self, start_angle=0, end_angle=180, step=10, delay=0.5):
        """
        Sweep servo between two angles.
        
        Args:
            start_angle: Starting angle in degrees
            end_angle: Ending angle in degrees  
            step: Step size in degrees
            delay: Delay between steps in seconds
        """
        angles = list(range(start_angle, end_angle + 1, step))
        
        for angle in angles:
            print(f"Moving to {angle} degrees")
            self.set_angle(angle)
            time.sleep(delay)
    
    def center(self):
        """Move servo to center position (90 degrees)."""
        self.set_angle(90)
        
    def cleanup(self):
        """Clean up GPIO resources."""
        try:
            self.pwm.stop()
            GPIO.cleanup()
        except:
            pass

def test_servo_controller():
    """Test servo controller functionality."""
    print("Initializing servo controller...")
    servo = ServoController(12)  # GPIO 12 (Pin 32)
    
    print("Moving to center position...")
    servo.center()
    time.sleep(1)
    
    print("Testing angle positions...")
    test_angles = [0, 45, 90, 135, 180, 90]
    
    for angle in test_angles:
        print(f"Setting angle to {angle} degrees")
        servo.set_angle(angle)
        time.sleep(1)
    
    print("Performing sweep test...")
    servo.sweep(start_angle=0, end_angle=180, step=20, delay=0.3)
    
    print("Sweep in reverse...")
    servo.sweep(start_angle=180, end_angle=0, step=-20, delay=0.3)
    
    print("Testing pulse width control...")
    pulse_widths = [1.0, 1.25, 1.5, 1.75, 2.0, 1.5]
    
    for pulse in pulse_widths:
        angle = servo.get_angle_from_pulse(pulse)
        print(f"Setting pulse width {pulse}ms (≈{angle:.1f}°)")
        servo.set_pulse_width(pulse)
        time.sleep(0.8)
    
    print("Returning to center...")
    servo.center()
    
    print("Servo test complete!")

if __name__ == '__main__':
    test_servo_controller()