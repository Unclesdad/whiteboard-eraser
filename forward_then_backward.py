#!/usr/bin/env python3
"""
Forward Then Backward Motor Test
Tests two N20 motors moving forward together, then backward together.
One motor is inverted to simulate differential drive setup.
"""

import time
import sys
import os

# Try to import N20Motor with better error handling
try:
    from motor import N20Motor, DualMotorController, EncoderManager
    print("Successfully imported N20Motor classes")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure motor.py contains the N20Motor classes")
    sys.exit(1)

# Try to import ServoController
try:
    from servo import ServoController
    SERVO_AVAILABLE = True
    print("Successfully imported ServoController")
except ImportError as e:
    SERVO_AVAILABLE = False
    print(f"WARNING: Servo not available: {e}")

def main():
    print("=== Forward Then Backward N20 Motor Test ===")
    print("Setting up dual motor controller...")

    # gpiozero handles GPIO setup automatically

    try:
        # Initialize servo to angle 0 if available
        servo = None
        if SERVO_AVAILABLE:
            servo = ServoController(pwm_pin=12)
            servo.set_angle(90)
            print("Servo set to angle 0")

        # Create dual motor controller with standby pin
        controller = DualMotorController(standby_pin=22)
        
        # Motor 1 - Left side (inverted direction)
        motor1 = N20Motor(
            pwm_pin=18, dir1_pin=23, dir2_pin=24,
            enc_a_pin=17, enc_b_pin=27,
            reverse_encoder=True,  # Inverted encoder
            use_interrupts=False,  # Use reliable polling method
            name="Motor1 (Left)"
        )
        
        # Motor 2 - Right side (inverted for differential drive)
        motor2 = N20Motor(
            pwm_pin=19, dir1_pin=25, dir2_pin=8,
            enc_a_pin=5, enc_b_pin=6,
            reverse_encoder=True,  # Inverted encoder
            use_interrupts=False,  # Use reliable polling method
            name="Motor2 (Right)"
        )
        
        # Register motors with controller
        controller.motor_a = motor1
        controller.motor_b = motor2
        
        print("Motors initialized!")
        print("  Motor1 (Left): Inverted encoder")  
        print("  Motor2 (Right): Inverted encoder for differential drive")
        print()
        
        # Give encoder thread time to stabilize
        time.sleep(1)
        
        # Reset both encoder positions
        motor1.reset_encoder()
        motor2.reset_encoder()
        
        def print_positions(label):
            """Print current positions and status of both motors"""
            count1 = motor1.get_encoder_count()
            count2 = motor2.get_encoder_count()
            rev1 = motor1.get_revolutions()
            rev2 = motor2.get_revolutions()
            print(f"{label}:")
            print(f"  Motor1: {count1:6d} counts ({rev1:5.2f} rev)")
            print(f"  Motor2: {count2:6d} counts ({rev2:5.2f} rev)")
        
        def set_both_motors(power1, power2, duration, description):
            """Set both motors and track their movement"""
            print(f"\n--- {description} ---")
            print_positions("Start")
            
            start_count1 = motor1.get_encoder_count()
            start_count2 = motor2.get_encoder_count()
            
            # Set motor powers
            motor1.set(power1)
            motor2.set(power2)
            
            print(f"Powers set - Motor1: {power1:+.1f}, Motor2: {power2:+.1f}")
            
            # Run for specified duration with progress updates
            for i in range(int(duration)):
                time.sleep(1)
                current1 = motor1.get_encoder_count()
                current2 = motor2.get_encoder_count()
                rate1 = current1 - start_count1
                rate2 = current2 - start_count2
                print(f"  {i+1}s: M1={rate1:4d} M2={rate2:4d}")
            
            # Stop motors
            motor1.stop()
            motor2.stop()
            
            # Calculate final movement
            end_count1 = motor1.get_encoder_count()
            end_count2 = motor2.get_encoder_count()
            
            movement1 = end_count1 - start_count1
            movement2 = end_count2 - start_count2
            
            print_positions("End  ")
            print(f"Total movement - Motor1: {movement1:+6d}, Motor2: {movement2:+6d}")
            print(f"Avg rates - Motor1: {abs(movement1)/duration:5.1f} cps, Motor2: {abs(movement2)/duration:5.1f} cps")
        
        print("Starting simple forward/backward test...")
        print("=" * 60)
        
        # Test 1: Both motors forward at 100% power
        set_both_motors(1.0, 1.0, 3.0, "Forward Motion - Both motors at 100% power (3 seconds)")

        # Pause between tests
        time.sleep(2)

        # Test 2: Both motors backward at 100% power
        set_both_motors(-1.0, -1.0, 3.0, "Backward Motion - Both motors at 100% power (3 seconds)")
        
        print("\n" + "=" * 60)
        print("Test sequence complete!")
        print_positions("Final")
        
        # Show motor status
        print("\nMotor Status:")
        status1 = motor1.get_status()
        status2 = motor2.get_status()
        
        for status in [status1, status2]:
            print(f"  {status['name']}:")
            print(f"    Encoder method: {status['encoder_method']}")
            print(f"    Total counts: {status['encoder_count']}")
            print(f"    Total revolutions: {status['revolutions']:.2f}")
            print(f"    Current speed: {status['speed']}")
            print(f"    Active: {status['active']}")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure motors are stopped and cleanup
        print("\nStopping motors and cleaning up...")
        try:
            if 'motor1' in locals():
                motor1.stop()
            if 'motor2' in locals():
                motor2.stop()
            if 'controller' in locals():
                controller.cleanup()
            if 'servo' in locals() and servo is not None:
                servo.cleanup()
            time.sleep(0.5)
            # gpiozero handles cleanup automatically
        except Exception as e:
            print(f"Cleanup error: {e}")

        print("Motors stopped and GPIO cleaned up.")

if __name__ == '__main__':
    main()