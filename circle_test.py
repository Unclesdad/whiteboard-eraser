#!/usr/bin/env python3
"""
Circle Test - Drive in circles both directions
Tests Ackermann steering by driving in full circles with servo at max angles.
"""

import time
import sys

# Try to import N20Motor with better error handling
try:
    from motor import N20Motor, DualMotorController
    print("✓ Successfully imported N20Motor classes")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure motor.py contains the N20Motor classes")
    sys.exit(1)

# Try to import ServoController
try:
    from servo import ServoController
    SERVO_AVAILABLE = True
    print("✓ Successfully imported ServoController")
except ImportError as e:
    SERVO_AVAILABLE = False
    print(f"⚠ Servo not available: {e}")
    sys.exit(1)

def main():
    print("=== Circle Test - Drive in Full Circles ===")
    print("This test will drive the car in two complete circles:")
    print("  1. Left turn circle (servo at 45°)")
    print("  2. Right turn circle (servo at 135°)")
    print()

    try:
        # Initialize servo
        servo = ServoController(pwm_pin=12)
        servo.set_angle(90)  # Start at center
        print("✓ Servo initialized at center (90°)")
        time.sleep(1)

        # Create dual motor controller with standby pin
        controller = DualMotorController(standby_pin=22)

        # Motor 1 - Left side
        motor1 = N20Motor(
            pwm_pin=18, dir1_pin=23, dir2_pin=24,
            enc_a_pin=17, enc_b_pin=27,
            reverse_encoder=True,
            use_interrupts=False,
            name="Motor1 (Left)"
        )

        # Motor 2 - Right side
        motor2 = N20Motor(
            pwm_pin=19, dir1_pin=25, dir2_pin=8,
            enc_a_pin=5, enc_b_pin=6,
            reverse_encoder=True,
            use_interrupts=False,
            name="Motor2 (Right)"
        )

        # Register motors with controller
        controller.motor_a = motor1
        controller.motor_b = motor2

        print("✓ Motors initialized!")
        print()

        # Give encoder thread time to stabilize
        time.sleep(1)

        # Reset both encoder positions
        motor1.reset_encoder()
        motor2.reset_encoder()

        def drive_circle(servo_angle, direction_name, duration):
            """Drive in a circle at specified servo angle"""
            print(f"\n{'='*60}")
            print(f"--- {direction_name} Circle (servo at {servo_angle}°) ---")
            print(f"Duration: {duration} seconds")
            print(f"{'='*60}\n")

            # Set servo angle
            servo.set_angle(servo_angle)
            print(f"✓ Servo set to {servo_angle}°")
            time.sleep(0.5)  # Let servo reach position

            # Start motors at full speed
            power = 1.0  # 100% power
            motor1.set(power)
            motor2.set(power)
            print(f"✓ Motors running at {power*100:.0f}% power")

            start_count1 = motor1.get_encoder_count()
            start_count2 = motor2.get_encoder_count()

            # Run for specified duration with progress updates
            print(f"\nDriving in circle...")
            for i in range(int(duration)):
                time.sleep(1)
                current1 = motor1.get_encoder_count()
                current2 = motor2.get_encoder_count()
                delta1 = current1 - start_count1
                delta2 = current2 - start_count2
                print(f"  {i+1}s: Motor1={delta1:5d} counts, Motor2={delta2:5d} counts")

            # Stop motors
            motor1.stop()
            motor2.stop()

            # Report final movement
            end_count1 = motor1.get_encoder_count()
            end_count2 = motor2.get_encoder_count()
            total1 = end_count1 - start_count1
            total2 = end_count2 - start_count2

            print(f"\n✓ Circle complete!")
            print(f"  Motor1 total: {total1:+6d} counts ({abs(total1)/duration:.1f} cps)")
            print(f"  Motor2 total: {total2:+6d} counts ({abs(total2)/duration:.1f} cps)")

            # Return servo to center
            servo.set_angle(90)
            print(f"✓ Servo returned to center")
            time.sleep(1)

        print("\nStarting circle tests...")

        # Test 1: Left turn circle (servo at 45°)
        drive_circle(45, "LEFT TURN", duration=8)

        # Pause between circles
        print("\n⏸  Pausing 2 seconds before next circle...")
        time.sleep(2)

        # Test 2: Right turn circle (servo at 135°)
        drive_circle(135, "RIGHT TURN", duration=8)

        # Final report
        print(f"\n{'='*60}")
        print("✓ Circle test sequence complete!")
        print(f"{'='*60}")

        # Show final motor status
        print("\nFinal Motor Status:")
        status1 = motor1.get_status()
        status2 = motor2.get_status()

        for status in [status1, status2]:
            print(f"  {status['name']}:")
            print(f"    Total counts: {status['encoder_count']}")
            print(f"    Total revolutions: {status['revolutions']:.2f}")

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
            if 'servo' in locals():
                servo.set_angle(90)  # Center servo before cleanup
                time.sleep(0.2)
                servo.cleanup()
            time.sleep(0.5)
        except Exception as e:
            print(f"Cleanup error: {e}")

        print("✓ Motors stopped and GPIO cleaned up.")

if __name__ == '__main__':
    main()
