#!/usr/bin/env python3
"""
Circle Test - Drive in circles both directions
Tests Ackermann steering by driving in full circles with servo at max angles.
Uses gyroscope to ensure precise 180° turns.
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

# Try to import Gyro
try:
    from gyro import RCCarGyro
    GYRO_AVAILABLE = True
    print("✓ Successfully imported RCCarGyro")
except ImportError as e:
    GYRO_AVAILABLE = False
    print(f"⚠ Gyro not available: {e}")
    sys.exit(1)

def main():
    print("=== Circle Test - Drive in Full Circles ===")
    print("This test will drive the car in two 180° semicircles:")
    print("  1. Left turn semicircle (servo at 45°)")
    print("  2. Right turn semicircle (servo at 135°)")
    print("Using gyroscope for precise 180° turns")
    print()

    try:
        # Initialize gyro
        print("Initializing gyroscope...")
        gyro = RCCarGyro()
        print("✓ Gyroscope initialized")

        # Calibrate gyro (MUST be stationary!)
        print("\n⚠️  KEEP CAR STATIONARY for calibration...")
        time.sleep(2)
        gyro.calibrate(samples=500, show_progress=True)

        # Start continuous gyro updates
        gyro.start_continuous_update(update_rate=50)
        print("✓ Gyro continuous updates started at 50Hz")
        time.sleep(0.5)

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

        def normalize_angle_change(start, current):
            """Calculate angle change handling wraparound at ±180°"""
            delta = current - start
            # Normalize to [-180, 180]
            while delta > 180:
                delta -= 360
            while delta < -180:
                delta += 360
            return delta

        def drive_circle(servo_angle, direction_name):
            """Drive in a 180° semicircle at specified servo angle"""
            print(f"\n{'='*60}")
            print(f"--- {direction_name} Semicircle (servo at {servo_angle}°) ---")
            print(f"Target: 180° turn")
            print(f"{'='*60}\n")

            # Set servo angle
            servo.set_angle(servo_angle)
            print(f"✓ Servo set to {servo_angle}°")
            time.sleep(0.5)  # Let servo reach position

            # Record starting heading
            _, _, start_heading = gyro.get_orientation()
            print(f"Starting heading: {start_heading:.1f}°")

            # Start motors at full speed
            power = 1.0  # 100% power
            motor1.set(power)
            motor2.set(power)
            print(f"✓ Motors running at {power*100:.0f}% power")

            start_count1 = motor1.get_encoder_count()
            start_count2 = motor2.get_encoder_count()
            start_time = time.time()

            # Drive until 180° turn complete
            print(f"\nDriving semicircle (target: 180°)...")
            loop_count = 0
            last_heading = start_heading
            cumulative_turn = 0.0

            while True:
                time.sleep(0.1)  # 10Hz polling
                loop_count += 1

                # Get current heading
                _, _, current_heading = gyro.get_orientation()

                # Calculate incremental change from last reading (handles wraparound)
                incremental_change = normalize_angle_change(last_heading, current_heading)
                cumulative_turn += incremental_change
                last_heading = current_heading

                heading_change = normalize_angle_change(start_heading, current_heading)

                # Print progress every second
                if loop_count % 10 == 0:
                    elapsed = time.time() - start_time
                    current1 = motor1.get_encoder_count()
                    current2 = motor2.get_encoder_count()
                    delta1 = current1 - start_count1
                    delta2 = current2 - start_count2
                    print(f"  {elapsed:.1f}s: Heading={heading_change:+6.1f}° | Cumulative={cumulative_turn:+6.1f}° | M1={delta1:5d} M2={delta2:5d}")

                # Check if we've completed 180° turn (use cumulative to handle wraparound)
                if abs(cumulative_turn) >= 180.0:
                    break

            # Stop motors
            motor1.stop()
            motor2.stop()
            elapsed_time = time.time() - start_time

            # Report final movement
            end_count1 = motor1.get_encoder_count()
            end_count2 = motor2.get_encoder_count()
            total1 = end_count1 - start_count1
            total2 = end_count2 - start_count2
            _, _, final_heading = gyro.get_orientation()
            actual_turn = normalize_angle_change(start_heading, final_heading)

            print(f"\n✓ Semicircle complete!")
            print(f"  Time: {elapsed_time:.1f} seconds")
            print(f"  Cumulative turn: {cumulative_turn:+.1f}° (target: ±180°)")
            print(f"  Final heading delta: {actual_turn:+.1f}°")
            print(f"  Motor1 total: {total1:+6d} counts ({abs(total1)/elapsed_time:.1f} cps)")
            print(f"  Motor2 total: {total2:+6d} counts ({abs(total2)/elapsed_time:.1f} cps)")

            # Return servo to center
            servo.set_angle(90)
            print(f"✓ Servo returned to center")
            time.sleep(1)

        print("\nStarting semicircle tests...")

        # Test 1: Left turn semicircle (servo at 45°)
        drive_circle(45, "LEFT TURN")

        # Pause between semicircles
        print("\n⏸  Pausing 2 seconds before next semicircle...")
        time.sleep(2)

        # Test 2: Right turn semicircle (servo at 135°)
        drive_circle(135, "RIGHT TURN")

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
            if 'gyro' in locals():
                gyro.stop_continuous_update()
            time.sleep(0.5)
        except Exception as e:
            print(f"Cleanup error: {e}")

        print("✓ Motors stopped and GPIO cleaned up.")

if __name__ == '__main__':
    main()
