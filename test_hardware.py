#!/usr/bin/env python3
"""
Simple hardware test for Raspberry Pi
Tests GPIO, I2C, and basic functionality
"""

import time
import os

def test_gpio_access():
    """Test basic GPIO access"""
    print("🔌 Testing GPIO Access...")

    try:
        import RPi.GPIO as GPIO
        print("  ✓ RPi.GPIO import successful")

        # Try to set mode
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        print("  ✓ GPIO mode set to BCM")

        # Test a simple pin setup (LED pin 18)
        test_pin = 18
        GPIO.setup(test_pin, GPIO.OUT)
        print(f"  ✓ Pin {test_pin} configured as output")

        # Test pin control
        GPIO.output(test_pin, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(test_pin, GPIO.LOW)
        print(f"  ✓ Pin {test_pin} control test passed")

        GPIO.cleanup()
        print("  ✓ GPIO cleanup successful")
        return True

    except Exception as e:
        print(f"  ❌ GPIO test failed: {e}")
        print("  💡 Try: sudo usermod -a -G gpio $USER")
        print("  💡 Then logout/login or reboot")
        return False

def test_i2c_access():
    """Test I2C bus access"""
    print("\n🔗 Testing I2C Access...")

    try:
        # Check if I2C devices exist
        i2c_devices = ["/dev/i2c-0", "/dev/i2c-1", "/dev/i2c-11"]
        found_devices = [dev for dev in i2c_devices if os.path.exists(dev)]

        if found_devices:
            print(f"  ✓ I2C devices found: {found_devices}")
        else:
            print("  ⚠️ No I2C devices found")
            print("  💡 Try: sudo raspi-config -> Interface Options -> I2C -> Enable")
            return False

        # Try to import smbus
        try:
            import smbus2 as smbus
            print("  ✓ smbus2 import successful")
        except ImportError:
            print("  ❌ smbus2 not available")
            print("  💡 Install with: pip3 install smbus2")
            return False

        # Try to access I2C bus
        try:
            bus = smbus.SMBus(1)
            print("  ✓ I2C bus 1 accessible")
            bus.close()
        except Exception as e:
            print(f"  ❌ I2C bus access failed: {e}")
            print("  💡 Try: sudo usermod -a -G i2c $USER")
            return False

        return True

    except Exception as e:
        print(f"  ❌ I2C test failed: {e}")
        return False

def test_camera_access():
    """Test camera access"""
    print("\n📷 Testing Camera Access...")

    try:
        # Check for camera device
        camera_devices = ["/dev/video0", "/dev/video10", "/dev/video11"]
        found_cameras = [dev for dev in camera_devices if os.path.exists(dev)]

        if found_cameras:
            print(f"  ✓ Camera devices found: {found_cameras}")
        else:
            print("  ⚠️ No camera devices found")
            print("  💡 Try: sudo raspi-config -> Interface Options -> Camera -> Enable")
            return False

        # Try to import picamera2
        try:
            from picamera2 import Picamera2
            print("  ✓ picamera2 import successful")
        except ImportError:
            print("  ❌ picamera2 not available")
            print("  💡 Install with: sudo apt install python3-picamera2")
            return False

        # Try to create camera instance (don't start it)
        try:
            camera = Picamera2()
            print("  ✓ Camera instance created")
            camera.close()
            return True
        except Exception as e:
            print(f"  ❌ Camera initialization failed: {e}")
            print("  💡 Try: sudo usermod -a -G video $USER")
            return False

    except Exception as e:
        print(f"  ❌ Camera test failed: {e}")
        return False

def test_system_info():
    """Display system information"""
    print("\n🖥️  System Information:")

    try:
        # Read Pi model
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
        print(f"  Model: {model}")
    except:
        print("  Model: Unknown")

    # Check groups
    import subprocess
    try:
        result = subprocess.run(['groups'], capture_output=True, text=True)
        groups = result.stdout.strip()
        print(f"  Groups: {groups}")

        # Check for required groups
        required_groups = ['gpio', 'i2c', 'video']
        missing_groups = [g for g in required_groups if g not in groups]
        if missing_groups:
            print(f"  ⚠️ Missing groups: {missing_groups}")
            print(f"  💡 Add with: sudo usermod -a -G {','.join(missing_groups)} $USER")
    except:
        print("  Groups: Unable to check")

def test_python_packages():
    """Test required Python packages"""
    print("\n🐍 Testing Python Packages...")

    packages = [
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
        ('RPi.GPIO', 'RPi.GPIO'),
        ('smbus2', 'smbus2'),
        ('picamera2', 'picamera2')
    ]

    success_count = 0
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
            success_count += 1
        except ImportError:
            print(f"  ❌ {package_name} - not installed")
            if package_name == 'smbus2':
                print(f"    💡 Install with: pip3 install {package_name}")
            elif package_name == 'picamera2':
                print(f"    💡 Install with: sudo apt install python3-{package_name}")
            else:
                print(f"    💡 Install with: sudo apt install python3-{import_name.replace('.', '-').lower()}")

    return success_count == len(packages)

def main():
    """Run all hardware tests"""
    print("🔧 Raspberry Pi Hardware Test")
    print("=" * 40)

    test_system_info()

    tests = [
        ("Python Packages", test_python_packages),
        ("GPIO Access", test_gpio_access),
        ("I2C Access", test_i2c_access),
        ("Camera Access", test_camera_access),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"💥 {test_name} - ERROR: {e}")

    print("\n" + "=" * 40)
    print(f"🏁 Hardware Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All hardware tests passed! Ready for car testing.")
    elif passed >= total - 1:
        print("⚠️  Almost ready. Fix the failing test above.")
    else:
        print("🔧 Multiple issues detected. Please fix hardware setup.")
        print("\n💡 Quick fixes to try:")
        print("  1. sudo raspi-config -> Enable I2C, Camera, SPI")
        print("  2. sudo usermod -a -G gpio,i2c,video $USER")
        print("  3. sudo apt install python3-opencv python3-numpy python3-rpi.gpio")
        print("  4. pip3 install smbus2")
        print("  5. sudo apt install python3-picamera2")
        print("  6. Reboot after making changes")

if __name__ == "__main__":
    main()