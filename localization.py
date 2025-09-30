#!/usr/bin/env python3
"""
Localization System for Whiteboard Eraser Car
Combines motor encoder data with gyroscope readings for accurate position tracking
"""

import numpy as np
import time
import threading
from typing import Tuple, Optional
from dataclasses import dataclass
from collections import deque

@dataclass
class Pose:
    """Represents car position and orientation"""
    x: float  # mm from start position
    y: float  # mm from start position
    theta: float  # radians (0 = facing positive X)
    timestamp: float
    confidence: float = 1.0

@dataclass
class Velocity:
    """Represents car velocity"""
    linear: float  # mm/s
    angular: float  # rad/s
    timestamp: float

class LocalizationSystem:
    """
    Combines wheel encoder odometry with gyroscope for accurate localization
    """

    def __init__(self,
                 wheel_base_mm: float = 110.0,  # Distance between front and back wheels
                 wheel_radius_mm: float = 30.0,  # Wheel radius
                 ticks_per_revolution: int = 4445,  # Encoder ticks per wheel revolution (calibrated)
                 track_width_mm: float = 110.0,  # Distance between left and right wheels
                 gyro_weight: float = 0.7,  # How much to trust gyro vs encoder for heading
                 max_position_history: int = 100):
        """
        Initialize localization system

        Args:
            wheel_base_mm: Distance between front and back axles
            wheel_radius_mm: Radius of drive wheels
            ticks_per_revolution: Encoder ticks per wheel revolution
            track_width_mm: Distance between left and right wheels
            gyro_weight: Weight for gyro vs encoder heading fusion (0.0-1.0)
            max_position_history: Maximum number of poses to keep in history
        """
        self.wheel_base_mm = wheel_base_mm
        self.wheel_radius_mm = wheel_radius_mm
        self.ticks_per_revolution = ticks_per_revolution
        self.track_width_mm = track_width_mm
        self.gyro_weight = gyro_weight

        # Calculate distance per encoder tick
        wheel_circumference = 2 * np.pi * wheel_radius_mm
        self.mm_per_tick = wheel_circumference / ticks_per_revolution

        # Current pose
        self.current_pose = Pose(0.0, 0.0, 0.0, time.time())
        self.pose_lock = threading.Lock()

        # Position history for smoothing and validation
        self.pose_history = deque(maxlen=max_position_history)

        # Previous encoder readings
        self.prev_left_ticks = 0
        self.prev_right_ticks = 0
        self.prev_time = time.time()

        # Velocity tracking
        self.current_velocity = Velocity(0.0, 0.0, time.time())
        self.velocity_history = deque(maxlen=10)

        # Gyro integration
        self.gyro_heading_offset = 0.0  # Offset to align gyro with encoder heading
        self.gyro_initialized = False

        # Error tracking for diagnostics
        self.odometry_errors = deque(maxlen=50)

        print(f"LocalizationSystem initialized:")
        print(f"  Wheel: radius={wheel_radius_mm}mm, {ticks_per_revolution} ticks/rev")
        print(f"  Distance per tick: {self.mm_per_tick:.3f}mm")
        print(f"  Geometry: wheelbase={wheel_base_mm}mm, track={track_width_mm}mm")

    def reset_position(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """Reset current position and orientation"""
        with self.pose_lock:
            self.current_pose = Pose(x, y, theta, time.time())
            self.pose_history.clear()
            self.pose_history.append(self.current_pose)

        # Reset encoder reference
        self.prev_time = time.time()
        print(f"Position reset to ({x:.1f}, {y:.1f}, {np.degrees(theta):.1f}°)")

    def update_with_encoders(self, left_ticks: int, right_ticks: int) -> Pose:
        """
        Update position using encoder readings

        Args:
            left_ticks: Current left motor encoder count
            right_ticks: Current right motor encoder count

        Returns:
            Updated pose
        """
        current_time = time.time()
        dt = current_time - self.prev_time

        if dt <= 0:
            return self.current_pose

        # Calculate tick differences
        left_delta = left_ticks - self.prev_left_ticks
        right_delta = right_ticks - self.prev_right_ticks

        # Invert right encoder (motor2 encoder is inverted in hardware)
        right_delta = -right_delta

        # Convert to distances
        left_distance = left_delta * self.mm_per_tick
        right_distance = right_delta * self.mm_per_tick

        # Calculate motion
        forward_distance = (left_distance + right_distance) / 2.0
        heading_change = (right_distance - left_distance) / self.track_width_mm

        # Calculate velocities
        linear_velocity = forward_distance / dt
        angular_velocity = heading_change / dt

        with self.pose_lock:
            # Update pose using differential drive kinematics with arc integration
            theta = self.current_pose.theta
            new_theta = theta + heading_change
            new_theta = self._normalize_angle(new_theta)

            # Use arc-based position update for turning, straight-line for driving straight
            if abs(heading_change) > 0.001:  # Threshold to avoid division by zero
                # Calculate radius of curvature and update using circular arc
                R = forward_distance / heading_change
                new_x = self.current_pose.x + R * (np.sin(new_theta) - np.sin(theta))
                new_y = self.current_pose.y - R * (np.cos(new_theta) - np.cos(theta))
            else:
                # Straight-line motion (limit case as heading_change -> 0)
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                new_x = self.current_pose.x + forward_distance * cos_theta
                new_y = self.current_pose.y + forward_distance * sin_theta

            # Create new pose
            new_pose = Pose(new_x, new_y, new_theta, current_time)
            self.current_pose = new_pose
            self.pose_history.append(new_pose)

        # Update velocity
        self.current_velocity = Velocity(linear_velocity, angular_velocity, current_time)
        self.velocity_history.append(self.current_velocity)

        # Update references
        self.prev_left_ticks = left_ticks
        self.prev_right_ticks = right_ticks
        self.prev_time = current_time

        return new_pose

    def update_with_gyro(self, gyro_heading_rad: float) -> Pose:
        """
        Fuse gyroscope data with encoder-based heading

        Args:
            gyro_heading_rad: Current heading from gyroscope (radians)

        Returns:
            Updated pose with fused heading
        """
        if not self.gyro_initialized:
            # Initialize gyro offset to align with current encoder heading
            self.gyro_heading_offset = self.current_pose.theta - gyro_heading_rad
            self.gyro_initialized = True
            return self.current_pose

        # Apply offset to gyro reading
        corrected_gyro_heading = gyro_heading_rad + self.gyro_heading_offset
        corrected_gyro_heading = self._normalize_angle(corrected_gyro_heading)

        with self.pose_lock:
            # Fuse encoder and gyro headings
            encoder_heading = self.current_pose.theta

            # Use complementary filter for sensor fusion
            fused_heading = (self.gyro_weight * corrected_gyro_heading +
                           (1 - self.gyro_weight) * encoder_heading)
            fused_heading = self._normalize_angle(fused_heading)

            # Update pose with fused heading
            self.current_pose.theta = fused_heading
            self.current_pose.timestamp = time.time()

        return self.current_pose

    def get_pose(self) -> Pose:
        """Get current pose (thread-safe)"""
        with self.pose_lock:
            return Pose(
                self.current_pose.x,
                self.current_pose.y,
                self.current_pose.theta,
                self.current_pose.timestamp,
                self.current_pose.confidence
            )

    def get_velocity(self) -> Velocity:
        """Get current velocity"""
        return Velocity(
            self.current_velocity.linear,
            self.current_velocity.angular,
            self.current_velocity.timestamp
        )

    def transform_to_global(self, local_x: float, local_y: float) -> Tuple[float, float]:
        """
        Transform local coordinates (relative to car) to global coordinates

        Args:
            local_x: X coordinate relative to car center (right positive)
            local_y: Y coordinate relative to car center (forward positive)

        Returns:
            (global_x, global_y) in global coordinate frame
        """
        pose = self.get_pose()

        # Rotate local coordinates by car heading
        cos_theta = np.cos(pose.theta)
        sin_theta = np.sin(pose.theta)

        global_x = pose.x + local_x * cos_theta - local_y * sin_theta
        global_y = pose.y + local_x * sin_theta + local_y * cos_theta

        return global_x, global_y

    def transform_to_local(self, global_x: float, global_y: float) -> Tuple[float, float]:
        """
        Transform global coordinates to local coordinates (relative to car)

        Args:
            global_x: X coordinate in global frame
            global_y: Y coordinate in global frame

        Returns:
            (local_x, local_y) relative to car center
        """
        pose = self.get_pose()

        # Translate to car-relative coordinates
        rel_x = global_x - pose.x
        rel_y = global_y - pose.y

        # Rotate by negative car heading
        cos_theta = np.cos(-pose.theta)
        sin_theta = np.sin(-pose.theta)

        local_x = rel_x * cos_theta - rel_y * sin_theta
        local_y = rel_x * sin_theta + rel_y * cos_theta

        return local_x, local_y

    def get_distance_to_point(self, target_x: float, target_y: float) -> float:
        """Get distance from current position to target point"""
        pose = self.get_pose()
        dx = target_x - pose.x
        dy = target_y - pose.y
        return np.sqrt(dx*dx + dy*dy)

    def get_heading_to_point(self, target_x: float, target_y: float) -> float:
        """Get heading angle to target point (in global frame)"""
        pose = self.get_pose()
        dx = target_x - pose.x
        dy = target_y - pose.y
        return np.arctan2(dy, dx)

    def get_relative_bearing_to_point(self, target_x: float, target_y: float) -> float:
        """Get relative bearing to target point (relative to current heading)"""
        target_heading = self.get_heading_to_point(target_x, target_y)
        pose = self.get_pose()
        relative_bearing = target_heading - pose.theta
        return self._normalize_angle(relative_bearing)

    def predict_future_pose(self, time_ahead: float) -> Pose:
        """
        Predict future pose based on current velocity

        Args:
            time_ahead: Time in seconds to predict ahead

        Returns:
            Predicted pose
        """
        pose = self.get_pose()
        velocity = self.get_velocity()

        # Predict using current velocity
        predicted_x = pose.x + velocity.linear * np.cos(pose.theta) * time_ahead
        predicted_y = pose.y + velocity.linear * np.sin(pose.theta) * time_ahead
        predicted_theta = pose.theta + velocity.angular * time_ahead
        predicted_theta = self._normalize_angle(predicted_theta)

        return Pose(predicted_x, predicted_y, predicted_theta,
                   pose.timestamp + time_ahead, confidence=0.8)

    def get_smoothed_pose(self, window_size: int = 5) -> Optional[Pose]:
        """Get pose smoothed over recent history"""
        if len(self.pose_history) < window_size:
            return self.get_pose()

        # Get recent poses
        recent_poses = list(self.pose_history)[-window_size:]

        # Average positions
        avg_x = np.mean([p.x for p in recent_poses])
        avg_y = np.mean([p.y for p in recent_poses])

        # Average angles (handling wraparound)
        angles = [p.theta for p in recent_poses]
        avg_theta = self._average_angles(angles)

        latest_time = recent_poses[-1].timestamp

        return Pose(avg_x, avg_y, avg_theta, latest_time, confidence=0.9)

    def validate_pose_update(self, new_pose: Pose, max_velocity: float = 500.0) -> bool:
        """
        Validate if a pose update is reasonable

        Args:
            new_pose: Proposed new pose
            max_velocity: Maximum expected velocity (mm/s)

        Returns:
            True if pose update is valid
        """
        if not self.pose_history:
            return True

        last_pose = self.pose_history[-1]
        dt = new_pose.timestamp - last_pose.timestamp

        if dt <= 0:
            return False

        # Check distance traveled
        dx = new_pose.x - last_pose.x
        dy = new_pose.y - last_pose.y
        distance = np.sqrt(dx*dx + dy*dy)
        velocity = distance / dt

        # Check angular change
        angle_change = abs(self._normalize_angle(new_pose.theta - last_pose.theta))
        angular_velocity = angle_change / dt

        # Validate limits
        max_angular_velocity = np.radians(180)  # 180 deg/s max

        is_valid = (velocity <= max_velocity and
                   angular_velocity <= max_angular_velocity)

        if not is_valid:
            error_info = {
                'velocity': velocity,
                'angular_velocity': np.degrees(angular_velocity),
                'distance': distance,
                'dt': dt
            }
            self.odometry_errors.append(error_info)

        return is_valid

    def get_diagnostics(self) -> dict:
        """Get diagnostic information about localization performance"""
        pose = self.get_pose()
        velocity = self.get_velocity()

        diagnostics = {
            'current_pose': {
                'x_mm': pose.x,
                'y_mm': pose.y,
                'theta_deg': np.degrees(pose.theta),
                'timestamp': pose.timestamp
            },
            'current_velocity': {
                'linear_mm_s': velocity.linear,
                'angular_deg_s': np.degrees(velocity.angular)
            },
            'history_size': len(self.pose_history),
            'gyro_initialized': self.gyro_initialized,
            'recent_errors': len(self.odometry_errors),
            'mm_per_tick': self.mm_per_tick
        }

        # Add recent error info if available
        if self.odometry_errors:
            recent_error = self.odometry_errors[-1]
            diagnostics['last_error'] = recent_error

        return diagnostics

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _average_angles(self, angles: list) -> float:
        """Average a list of angles handling wraparound"""
        x = np.mean([np.cos(a) for a in angles])
        y = np.mean([np.sin(a) for a in angles])
        return np.arctan2(y, x)


# Test function
def test_localization_system():
    """Test the localization system with simulated encoder data"""
    print("Testing LocalizationSystem...")

    localization = LocalizationSystem(debug=True)

    # Simulate moving forward
    print("\nSimulating forward movement...")
    left_ticks = 0
    right_ticks = 0

    for i in range(10):
        # Simulate 10 ticks on each wheel (forward motion)
        left_ticks += 10
        right_ticks += 10

        pose = localization.update_with_encoders(left_ticks, right_ticks)
        velocity = localization.get_velocity()

        print(f"Step {i+1}: pos=({pose.x:.1f}, {pose.y:.1f}), "
              f"heading={np.degrees(pose.theta):.1f}°, "
              f"vel={velocity.linear:.1f}mm/s")

        time.sleep(0.1)

    # Simulate turning right
    print("\nSimulating right turn...")
    for i in range(5):
        # More ticks on left wheel than right (turn right)
        left_ticks += 10
        right_ticks += 5

        pose = localization.update_with_encoders(left_ticks, right_ticks)
        print(f"Turn {i+1}: pos=({pose.x:.1f}, {pose.y:.1f}), "
              f"heading={np.degrees(pose.theta):.1f}°")

        time.sleep(0.1)

    # Test coordinate transformations
    print("\nTesting coordinate transformations...")
    local_point = (100, 50)  # 100mm right, 50mm forward of car
    global_point = localization.transform_to_global(*local_point)
    back_to_local = localization.transform_to_local(*global_point)

    print(f"Local point: {local_point}")
    print(f"Global point: ({global_point[0]:.1f}, {global_point[1]:.1f})")
    print(f"Back to local: ({back_to_local[0]:.1f}, {back_to_local[1]:.1f})")

    # Print diagnostics
    diag = localization.get_diagnostics()
    print(f"\nDiagnostics: {diag}")


if __name__ == "__main__":
    test_localization_system()