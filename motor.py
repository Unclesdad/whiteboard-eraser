#!/usr/bin/env python3

import atexit
import RPi.GPIO as GPIO
import time
import threading
import signal
import sys
import numpy as np

class MotorController:
    # Class-level shared resources for encoder management
    _encoder_thread = None
    _encoder_positions = {}
    _encoder_pins = {}
    _encoder_last_states = {}
    _running = False
    _lock = threading.Lock()
    _motor_count = 0
    _polling_freq = 7500  # 7.5 kHz polling frequency
    
    def __init__(self, pwm_pin:int, in1_pin:int, in2_pin:int, stby_pin:int, enc_a_pin:int, enc_b_pin:int):
        """
        Initialize motor controller with encoder support.
        
        Args:
            pwm_pin: GPIO pin for PWM speed control
            in1_pin: GPIO pin for motor direction control 1
            in2_pin: GPIO pin for motor direction control 2  
            stby_pin: GPIO pin for motor standby control
            enc_a_pin: GPIO pin for encoder channel A
            enc_b_pin: GPIO pin for encoder channel B
        """
        GPIO.cleanup()

        self.PWM = pwm_pin
        self.IN1 = in1_pin
        self.IN2 = in2_pin
        self.STBY = stby_pin
        
        # Register this motor's encoder
        with MotorController._lock:
            self.motor_id = MotorController._motor_count
            MotorController._motor_count += 1
            MotorController._encoder_pins[self.motor_id] = (enc_a_pin, enc_b_pin)
            MotorController._encoder_positions[self.motor_id] = 0

        self.setup_gpio()

        self.pwm = GPIO.PWM(self.PWM, 1000)
        self.pwm.start(0)
        
        # Start encoder thread if first motor
        with MotorController._lock:
            if not MotorController._running:
                MotorController._start_encoder_thread()

        # make it exit cleanly
        def cleanup_handler(signum, frame):
            MotorController._running = False
            if MotorController._encoder_thread:
                MotorController._encoder_thread.join(timeout=1.0)
            GPIO.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, cleanup_handler)

        atexit.register(lambda: (setattr(MotorController, '_running', False), GPIO.cleanup()))

    def setup_gpio(self):
        """Configure GPIO pins for motor control."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        GPIO.setup(self.PWM, GPIO.OUT)
        GPIO.setup(self.IN1, GPIO.OUT)
        GPIO.setup(self.IN2, GPIO.OUT)

        GPIO.setup(self.STBY, GPIO.OUT)
        
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        
        GPIO.output(self.STBY, GPIO.HIGH)

    def set(self, power):
        """
        Set motor power and direction.
        
        Args:
            power: Motor power from -1.0 to 1.0 (negative = reverse)
        """
        power = min(max(power, -1), 1)

        sign = np.sign(power)

        if sign == 1:
            GPIO.output(self.IN1, GPIO.HIGH)
            GPIO.output(self.IN2, GPIO.LOW)
        elif sign == -1:
            GPIO.output(self.IN1, GPIO.LOW)
            GPIO.output(self.IN2, GPIO.HIGH)
        else:
            GPIO.output(self.IN1, GPIO.LOW)
            GPIO.output(self.IN2, GPIO.LOW)

        self.pwm.ChangeDutyCycle(abs(power * 100))

    @classmethod
    def _start_encoder_thread(cls):
        """Start the shared encoder polling thread at 7500Hz."""
        cls._running = True
        cls._encoder_thread = threading.Thread(target=cls._encoder_loop, daemon=True)
        cls._encoder_thread.start()
        print(f"Encoder polling started at {cls._polling_freq} Hz")
    
    @classmethod
    def _encoder_loop(cls):
        """
        High-speed encoder polling loop running at 7500Hz.
        Monitors all motor encoders simultaneously using quadrature decoding.
        """
        # Setup GPIO mode for encoder thread
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup all encoder pins
        for motor_id, (pin_a, pin_b) in cls._encoder_pins.items():
            GPIO.setup(pin_a, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(pin_b, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Initialize last states for edge detection
        for motor_id, (pin_a, pin_b) in cls._encoder_pins.items():
            cls._encoder_last_states[motor_id] = GPIO.input(pin_a)
        
        polling_interval = 1.0 / cls._polling_freq
        next_poll = time.perf_counter()
        
        while cls._running:
            # Ultra-fast encoder reading for all motors
            for motor_id, (pin_a, pin_b) in cls._encoder_pins.items():
                a_state = GPIO.input(pin_a)
                b_state = GPIO.input(pin_b)
                last_a = cls._encoder_last_states[motor_id]
                
                # Process encoder changes using our optimized method
                if a_state != last_a:
                    if a_state == b_state:
                        cls._encoder_positions[motor_id] += 1
                    else:
                        cls._encoder_positions[motor_id] -= 1
                    cls._encoder_last_states[motor_id] = a_state
            
            # Precise timing control
            next_poll += polling_interval
            sleep_time = next_poll - time.perf_counter()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_position(self):
        """
        Get current encoder position in counts.
        
        Returns:
            int: Current encoder count (positive = forward, negative = reverse)
        """
        with MotorController._lock:
            return MotorController._encoder_positions.get(self.motor_id, 0)
    
    def reset_position(self):
        """Reset encoder position to zero."""
        with MotorController._lock:
            MotorController._encoder_positions[self.motor_id] = 0
    
    def get_speed(self, window_time=0.5):
        """
        Calculate motor speed by measuring position change over time.
        
        Args:
            window_time: Time window in seconds for speed calculation
            
        Returns:
            float: Speed in encoder counts per second
        """
        start_pos = self.get_position()
        time.sleep(window_time)
        end_pos = self.get_position()
        return (end_pos - start_pos) / window_time

def test_motor_controller():
    # Updated pin assignments to match our testing
    motor = MotorController(18, 23, 24, 22, 17, 27)  # Using pins 17, 27 for encoder

    def set_then_wait(set_power:float, sleeptime:float):
        start_pos = motor.get_position()
        print(f'Setting power: {set_power:.1f}, start position: {start_pos}')
        motor.set(set_power)
        time.sleep(sleeptime)  
        end_pos = motor.get_position()
        counts = end_pos - start_pos
        rate = abs(counts) / 2.0  # counts per second
        print(f'  End position: {end_pos}, counts: {counts}, rate: {rate:.1f} cps')

    sample_powers = [0.3, 0.4, 0.5, 0.6, 0.8, 0, -0.4, -0.6, -0.8, 0]

    print("Starting motor test with optimized 7.5kHz encoder polling...")
    time.sleep(1)  # Let encoder thread stabilize
    
    motor.reset_position()  # Start from zero
    
    for power in sample_powers:
        set_then_wait(power,2)
        
    print(f'\nTest complete. Final position: {motor.get_position()}')
    print("Encoder polling will continue in background for CV integration")

    set_then_wait(0,10)

if __name__ == '__main__':
    test_motor_controller()