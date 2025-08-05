#!/usr/bin/env python3
"""
RC Car Controller for Whiteboard Marking Erasure
Integrates with the whiteboard tracking system to navigate and erase markings
"""

import numpy as np
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
from collections import deque
import heapq
from enum import Enum

# Import from the whiteboard tracker (assuming it's in whiteboard_tracker.py)
from whiteboard_tracker import WhiteboardTracker, WhiteboardConfig, Marking, CameraPosition, ObjectPosition

@dataclass
class RCCarConfig:
    """Configuration for RC car dimensions and constraints"""
    # Dimensions (all in mm)
    length: float = 200.0  # Total length of car
    width: float = 150.0   # Total width of car
    back_wheel_dist: float = 50.0   # Distance from center to back wheel axle
    front_wheel_dist: float = 50.0  # Distance from center to front wheel axle
    wheel_base: float = 100.0  # Distance between front and back axles
    
    # Camera mounting
    camera_mount_forward: float = 30.0  # Camera mounted 30mm forward of center
    camera_height: float = 50.0  # Camera height above whiteboard
    
    # Constraints
    front_axle_max_turning: float = math.radians(30)  # Max steering angle
    max_speed: float = 100.0  # mm/s
    min_speed: float = 10.0   # mm/s
    max_acceleration: float = 50.0  # mm/s²
    
    # Safety margins
    edge_margin: float = 30.0  # Distance to keep from whiteboard edges
    
    # Encoder specifications
    encoder_ticks_per_rev: int = 1024
    wheel_radius: float = 30.0  # mm

class CarState(Enum):
    """States for the car's behavior"""
    SEARCHING = "searching"
    PATHFINDING = "pathfinding"
    FOLLOWING_PATH = "following_path"
    ERASING = "erasing"
    AVOIDING_EDGE = "avoiding_edge"

@dataclass
class Pose:
    """Represents car position and orientation"""
    x: float  # mm
    y: float  # mm
    theta: float  # radians (0 = facing right)
    speed: float = 0.0  # Associated speed for path planning
    steering: float = 0.0  # Associated steering angle
    
    def copy(self):
        return Pose(self.x, self.y, self.theta, self.speed, self.steering)

class PathNode:
    """Node for path planning"""
    def __init__(self, x: float, y: float, theta: float, g: float = 0, h: float = 0, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.g = g  # Cost from start
        self.h = h  # Heuristic to goal
        self.f = g + h  # Total cost
        self.parent = parent
        self.speed = 0.0  # Store movement parameters
        self.steering = 0.0
        
    def __lt__(self, other):
        return self.f < other.f

class RCCarController:
    def __init__(self, car_config: RCCarConfig, whiteboard_config: WhiteboardConfig):
        self.car_config = car_config
        self.whiteboard_config = whiteboard_config
        
        # Initialize whiteboard tracker
        self.tracker = WhiteboardTracker(whiteboard_config)
        self.tracker.initialize_camera()
        
        # Current state
        self.pose = Pose(
            whiteboard_config.camera_x_mm - car_config.camera_mount_forward,
            whiteboard_config.camera_y_mm,
            0.0
        )
        self.velocity = 0.0  # Current speed mm/s
        self.steering_angle = 0.0  # Current front wheel angle
        
        # Control state
        self.state = CarState.SEARCHING
        self.target_marking = None
        self.planned_path = []
        self.path_index = 0
        
        # Position tracking
        self.vision_available = False
        self.last_vision_update = time.time()
        
        # Encoder data
        self.left_encoder_ticks = 0
        self.right_encoder_ticks = 0
        self.last_encoder_time = time.time()
        
        # Grid for pathfinding (5mm resolution)
        self.grid_resolution = 5.0
        self.initialize_occupancy_grid()
        
    def initialize_occupancy_grid(self):
        """Create occupancy grid for pathfinding"""
        self.grid_width = int(self.whiteboard_config.width_mm / self.grid_resolution)
        self.grid_height = int(self.whiteboard_config.height_mm / self.grid_resolution)
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        # Mark edges as occupied
        margin_cells = int((self.car_config.edge_margin + self.car_config.width/2) / self.grid_resolution)
        self.occupancy_grid[:margin_cells, :] = True  # Top
        self.occupancy_grid[-margin_cells:, :] = True  # Bottom
        self.occupancy_grid[:, :margin_cells] = True  # Left
        self.occupancy_grid[:, -margin_cells:] = True  # Right
        
    def update_pose_from_odometry(self, left_ticks: int, right_ticks: int):
        """Update pose based on encoder readings"""
        # Validate inputs
        if left_ticks is None or right_ticks is None:
            return
            
        current_time = time.time()
        dt = current_time - self.last_encoder_time
        
        if dt <= 0:
            return
            
        # Calculate wheel movements
        left_delta = (left_ticks - self.left_encoder_ticks) / self.car_config.encoder_ticks_per_rev * 2 * math.pi * self.car_config.wheel_radius
        right_delta = (right_ticks - self.right_encoder_ticks) / self.car_config.encoder_ticks_per_rev * 2 * math.pi * self.car_config.wheel_radius
        
        # Sanity check on wheel movements
        max_possible_delta = self.car_config.max_speed * dt * 2  # Allow some margin
        if abs(left_delta) > max_possible_delta or abs(right_delta) > max_possible_delta:
            print(f"Warning: Unrealistic encoder delta detected, ignoring")
            return
        
        # Update stored values
        self.left_encoder_ticks = left_ticks
        self.right_encoder_ticks = right_ticks
        self.last_encoder_time = current_time
        
        # Calculate linear and angular displacement (using track width)
        linear_delta = (left_delta + right_delta) / 2.0
        angular_delta = (right_delta - left_delta) / self.car_config.width
        
        # Update pose
        if abs(angular_delta) < 0.001:
            # Straight line motion
            self.pose.x += linear_delta * math.cos(self.pose.theta)
            self.pose.y += linear_delta * math.sin(self.pose.theta)
        else:
            # Arc motion
            radius = linear_delta / angular_delta
            self.pose.x += radius * (math.sin(self.pose.theta + angular_delta) - math.sin(self.pose.theta))
            self.pose.y += radius * (math.cos(self.pose.theta) - math.cos(self.pose.theta + angular_delta))
            self.pose.theta += angular_delta
            
        # Normalize theta to [-pi, pi]
        self.pose.theta = math.atan2(math.sin(self.pose.theta), math.cos(self.pose.theta))
        
        # Update velocity estimate
        self.velocity = linear_delta / dt if dt > 0 else 0
        
    def update_pose_from_camera(self, camera_pos: CameraPosition):
        """Update pose estimate using camera data when confidence is high"""
        if camera_pos.confidence > 0.3:  # Lower threshold for primary use
            # Camera is mounted forward of car center
            old_x = self.pose.x
            old_y = self.pose.y
            
            self.pose.x = camera_pos.x_mm - self.car_config.camera_mount_forward * math.cos(self.pose.theta)
            self.pose.y = camera_pos.y_mm - self.car_config.camera_mount_forward * math.sin(self.pose.theta)
            
            # If confidence is very high, trust it completely
            if camera_pos.confidence > 0.7:
                self.last_vision_update = time.time()
                self.vision_available = True
            
            # Calculate implied velocity from position change (for control purposes)
            dt = time.time() - self.last_encoder_time
            if dt > 0:
                dx = self.pose.x - old_x
                dy = self.pose.y - old_y
                self.velocity = math.sqrt(dx**2 + dy**2) / dt
            
    def calculate_turning_radius(self, steering_angle: float) -> float:
        """Calculate turning radius for given steering angle"""
        if abs(steering_angle) < 0.001:
            return float('inf')
        return self.car_config.wheel_base / math.tan(abs(steering_angle))
        
    def is_pose_valid(self, pose: Pose) -> bool:
        """Check if a pose is valid (not colliding with edges)"""
        # Check car corners
        half_length = self.car_config.length / 2
        half_width = self.car_config.width / 2
        
        # Car corners in local coordinates
        corners = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        
        # Transform to world coordinates
        cos_theta = math.cos(pose.theta)
        sin_theta = math.sin(pose.theta)
        
        for local_x, local_y in corners:
            world_x = pose.x + cos_theta * local_x - sin_theta * local_y
            world_y = pose.y + sin_theta * local_x + cos_theta * local_y
            
            # Check bounds
            if (world_x < self.car_config.edge_margin or 
                world_x > self.whiteboard_config.width_mm - self.car_config.edge_margin or
                world_y < self.car_config.edge_margin or
                world_y > self.whiteboard_config.height_mm - self.car_config.edge_margin):
                return False
                
        return True
        
    def find_nearest_marking(self, markings: List[Marking]) -> Optional[Marking]:
        """Find the nearest marking, preferring lower ones (easier to reach)"""
        if not markings:
            return None
            
        best_marking = None
        best_score = float('inf')
        
        for marking in markings:
            # Calculate distance
            dx = marking.center_x_mm - self.pose.x
            dy = marking.center_y_mm - self.pose.y
            distance = math.sqrt(dx**2 + dy**2)
            
            # Prefer markings below current position (gravity helps)
            gravity_bonus = 0 if marking.center_y_mm > self.pose.y else 50
            
            score = distance + gravity_bonus
            
            if score < best_score:
                best_score = score
                best_marking = marking
                
        return best_marking
        
    def plan_path_to_marking(self, marking: Marking) -> List[Pose]:
        """Plan a path to the marking using A* with car constraints"""
        start = PathNode(self.pose.x, self.pose.y, self.pose.theta)
        goal_x = marking.center_x_mm
        goal_y = marking.center_y_mm
        
        # A* search with custom car dynamics
        open_set = [start]
        closed_set = set()
        iterations = 0
        max_iterations = 5000  # Prevent infinite loops
        
        # Discretize angles for search
        angle_resolution = math.radians(15)
        
        # Movement primitives: forward and backward with different steering angles
        movement_distances = [50.0, -40.0]  # Forward 50mm, backward 40mm
        steering_angles = [0, self.car_config.front_axle_max_turning / 2, 
                          self.car_config.front_axle_max_turning,
                          -self.car_config.front_axle_max_turning / 2,
                          -self.car_config.front_axle_max_turning]
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)
            
            # Check if we reached the goal
            dist_to_goal = math.sqrt((current.x - goal_x)**2 + (current.y - goal_y)**2)
            if dist_to_goal < self.whiteboard_config.erasure_radius_mm:
                # Reconstruct path
                path = []
                node = current
                while node:
                    pose = Pose(node.x, node.y, node.theta)
                    # Store movement parameters with the pose
                    pose.speed = getattr(node, 'speed', 50.0)
                    pose.steering = getattr(node, 'steering', 0.0)
                    path.append(pose)
                    node = node.parent
                return list(reversed(path))
                
            # Mark as visited
            state_key = (int(current.x/10), int(current.y/10), int(current.theta/angle_resolution))
            if state_key in closed_set:
                continue
            closed_set.add(state_key)
            
            # Explore neighbors with different steering angles and directions
            for distance in movement_distances:
                for steering in steering_angles:
                    # Skip backward turns at high speed (unrealistic)
                    if distance < 0 and abs(steering) > self.car_config.front_axle_max_turning / 2:
                        continue
                        
                    # Simulate movement
                    speed = distance / 0.5  # Convert to speed (mm/s)
                    new_poses = self.simulate_movement(
                        Pose(current.x, current.y, current.theta),
                        steering, speed, 0.5  # 0.5 seconds duration
                    )
                
                    if new_poses and self.is_pose_valid(new_poses[-1]):
                        new_x, new_y, new_theta = new_poses[-1].x, new_poses[-1].y, new_poses[-1].theta
                        
                        # Calculate costs
                        movement_cost = abs(distance)
                        g = current.g + movement_cost
                        h = math.sqrt((new_x - goal_x)**2 + (new_y - goal_y)**2)
                        
                        # Add penalties
                        if current.parent:
                            # Penalty for steering changes
                            g += abs(steering) * 10
                            # Penalty for direction changes
                            if distance < 0:
                                g += 20  # Prefer forward motion
                                
                        new_node = PathNode(new_x, new_y, new_theta, g, h, current)
                        # Store the movement parameters in the node
                        new_node.speed = speed
                        new_node.steering = steering
                        heapq.heappush(open_set, new_node)
                    
        return []  # No path found
        
    def simulate_movement(self, start_pose: Pose, steering_angle: float, 
                         speed: float, duration: float, dt: float = 0.1) -> List[Pose]:
        """Simulate car movement with given controls"""
        poses = []
        current = start_pose.copy()
        
        steps = int(duration / dt)
        for _ in range(steps):
            # Calculate turning radius
            turning_radius = self.calculate_turning_radius(steering_angle)
            
            if abs(turning_radius) == float('inf'):
                # Straight line
                current.x += speed * dt * math.cos(current.theta)
                current.y += speed * dt * math.sin(current.theta)
            else:
                # Arc motion
                angular_velocity = speed / turning_radius
                # Update position first, then angle
                dx = turning_radius * (math.sin(current.theta + angular_velocity * dt) - math.sin(current.theta))
                dy = turning_radius * (math.cos(current.theta) - math.cos(current.theta + angular_velocity * dt))
                current.x += dx
                current.y += dy
                current.theta += angular_velocity * dt
                
            # Normalize theta
            current.theta = math.atan2(math.sin(current.theta), math.cos(current.theta))
            poses.append(current.copy())
            
        return poses
        
    def calculate_motor_commands(self, target_speed: float, target_steering: float) -> Tuple[float, float]:
        """Convert speed and steering to differential drive commands"""
        # For front-wheel steering with rear-wheel drive
        if abs(target_steering) < 0.001:
            # Straight line - equal speeds
            return target_speed, target_speed
            
        # Calculate required wheel speeds for turning
        turning_radius = self.calculate_turning_radius(target_steering)
        angular_velocity = target_speed / turning_radius
        
        # Differential speeds for rear wheels (using track width)
        left_speed = target_speed - angular_velocity * self.car_config.width / 2
        right_speed = target_speed + angular_velocity * self.car_config.width / 2
        
        return left_speed, right_speed
        
    def search_for_markings(self) -> Tuple[float, float, float]:
        """Spin slowly to search for markings"""
        # Rotate in place slowly
        search_speed = math.radians(30)  # 30 degrees per second
        # Use track width for differential drive rotation
        left_speed = -self.car_config.width / 2 * search_speed
        right_speed = self.car_config.width / 2 * search_speed
        
        return left_speed, right_speed, 0.0  # No front wheel steering needed
        
    def follow_path(self) -> Tuple[float, float, float]:
        """Follow the planned path"""
        if self.path_index >= len(self.planned_path):
            return 0.0, 0.0, 0.0
            
        current_waypoint = self.planned_path[self.path_index]
        
        # Check if we need to go backward (from path planning)
        planned_speed = getattr(current_waypoint, 'speed', self.car_config.max_speed)
        
        # Look ahead for smoother control (only for forward motion)
        if planned_speed > 0:
            lookahead_distance = 100.0  # mm
            target_pose = None
            
            for i in range(self.path_index, len(self.planned_path)):
                pose = self.planned_path[i]
                dist = math.sqrt((pose.x - self.pose.x)**2 + (pose.y - self.pose.y)**2)
                if dist >= lookahead_distance:
                    target_pose = pose
                    break
                    
            if target_pose is None:
                target_pose = self.planned_path[-1]
        else:
            # For backward motion, just follow the immediate waypoint
            target_pose = current_waypoint
            
        # Calculate steering angle
        dx = target_pose.x - self.pose.x
        dy = target_pose.y - self.pose.y
        
        if planned_speed > 0:
            # Forward motion
            target_heading = math.atan2(dy, dx)
            heading_error = target_heading - self.pose.theta
        else:
            # Backward motion - reverse the heading calculation
            target_heading = math.atan2(-dy, -dx)
            heading_error = target_heading - (self.pose.theta + math.pi)
        
        # Normalize to [-pi, pi]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        # Calculate steering command
        steering = np.clip(heading_error * 2.0, -self.car_config.front_axle_max_turning, 
                          self.car_config.front_axle_max_turning)
        
        # Use planned speed with curvature adjustment
        speed = abs(planned_speed) * (1.0 - abs(steering) / self.car_config.front_axle_max_turning * 0.5)
        if planned_speed < 0:
            speed = -speed  # Maintain backward direction
        
        # Update path index
        current_dist = math.sqrt((self.planned_path[self.path_index].x - self.pose.x)**2 + 
                                (self.planned_path[self.path_index].y - self.pose.y)**2)
        if current_dist < 20.0:
            self.path_index += 1
            
        left_speed, right_speed = self.calculate_motor_commands(speed, steering)
        return left_speed, right_speed, steering
        
    def run(self):
        """Main control loop"""
        print("RC Car Controller Starting...")
        print(f"Whiteboard: {self.whiteboard_config.width_mm}x{self.whiteboard_config.height_mm}mm")
        print(f"Car dimensions: {self.car_config.length}x{self.car_config.width}mm")
        print(f"Max steering angle: {math.degrees(self.car_config.front_axle_max_turning)}°")
        
        try:
            while True:
                # Get camera frame and current state
                frame = self.tracker.camera.capture_array()
                camera_pos, object_pos, markings, erased = self.tracker.get_current_state(frame)
                
                # Update pose - vision is primary, odometry is backup
                self.update_pose_from_camera(camera_pos)
                
                # Only use odometry if vision has been unavailable for a while
                if time.time() - self.last_vision_update > 2.0 or camera_pos.confidence < 0.3:
                    # Get encoder readings here (implement your hardware interface)
                    # left_ticks, right_ticks = self.read_encoders()
                    # self.update_pose_from_odometry(left_ticks, right_ticks)
                    pass
                
                # State machine
                left_speed = 0.0
                right_speed = 0.0
                steering_angle = 0.0
                
                if self.state == CarState.SEARCHING:
                    # Look for markings
                    visible_markings = [m for m in markings if self.is_marking_reachable(m)]
                    
                    if visible_markings:
                        self.target_marking = self.find_nearest_marking(visible_markings)
                        if self.target_marking:
                            print(f"Found marking at ({self.target_marking.center_x_mm}, {self.target_marking.center_y_mm})")
                            self.state = CarState.PATHFINDING
                    else:
                        # Spin to search
                        left_speed, right_speed, steering_angle = self.search_for_markings()
                        
                elif self.state == CarState.PATHFINDING:
                    # Plan path to marking
                    print("Planning path...")
                    self.planned_path = self.plan_path_to_marking(self.target_marking)
                    
                    if self.planned_path:
                        self.path_index = 0
                        self.state = CarState.FOLLOWING_PATH
                        print(f"Path found with {len(self.planned_path)} waypoints")
                    else:
                        print("No path found, searching for new marking")
                        self.target_marking = None
                        self.state = CarState.SEARCHING
                        
                elif self.state == CarState.FOLLOWING_PATH:
                    # Follow the planned path
                    left_speed, right_speed, steering_angle = self.follow_path()
                    
                    # Check if we reached the marking
                    if self.target_marking:
                        dist = math.sqrt((self.target_marking.center_x_mm - self.pose.x)**2 + 
                                       (self.target_marking.center_y_mm - self.pose.y)**2)
                        if dist < self.whiteboard_config.erasure_radius_mm:
                            print("Reached marking, erasing...")
                            self.state = CarState.ERASING
                            
                    # Check if path is complete
                    if self.path_index >= len(self.planned_path):
                        self.state = CarState.SEARCHING
                        
                elif self.state == CarState.ERASING:
                    # Marking should be erased by proximity
                    # Wait a moment then search for next
                    time.sleep(0.5)
                    self.target_marking = None
                    self.state = CarState.SEARCHING
                    
                # Safety check - avoid edges
                if not self.is_pose_valid(self.pose):
                    print("Too close to edge! Backing up...")
                    # Back up slowly with slight steering to avoid getting stuck
                    left_speed = -self.car_config.min_speed
                    right_speed = -self.car_config.min_speed * 0.9  # Slight turn while backing
                    steering_angle = -self.steering_angle * 0.5  # Reverse steering direction
                    
                # Send motor commands (this would interface with actual motor controllers)
                self.send_motor_commands(left_speed, right_speed, steering_angle)
                
                # Small delay for control loop
                time.sleep(0.05)  # 20Hz control loop
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.send_motor_commands(0, 0, 0)
            self.tracker.camera.stop()
            
    def is_marking_reachable(self, marking: Marking) -> bool:
        """Check if a marking can be reached without hitting edges"""
        # Simple check - ensure marking is not too close to edges
        margin = self.car_config.erasure_radius_mm + self.car_config.width / 2
        return (margin < marking.center_x_mm < self.whiteboard_config.width_mm - margin and
                margin < marking.center_y_mm < self.whiteboard_config.height_mm - margin)
                
    def send_motor_commands(self, left_speed: float, right_speed: float, steering_angle: float):
        """Send commands to motors and servo (implement hardware interface here)"""
        # This is where you would interface with actual motor controllers
        # For now, just print the commands
        pass
        # Example:
        # self.left_motor.set_speed(left_speed)
        # self.right_motor.set_speed(right_speed)
        # self.steering_servo.set_angle(steering_angle)

def main():
    # Configuration
    car_config = RCCarConfig(
        length=200.0,
        width=150.0,
        back_wheel_dist=50.0,
        front_wheel_dist=50.0,
        camera_mount_forward=30.0,
        camera_height=50.0,
        front_axle_max_turning=math.radians(30),
        edge_margin=30.0
    )
    
    whiteboard_config = WhiteboardConfig(
        width_mm=2400.0,
        height_mm=1200.0,
        distance_mm=car_config.camera_height,
        camera_x_mm=1200.0,  # Starting position
        camera_y_mm=600.0,
        camera_placement_mm=car_config.camera_mount_forward,
        erasure_radius_mm=50.0
    )
    
    # Create and run controller
    controller = RCCarController(car_config, whiteboard_config)
    controller.run()

if __name__ == "__main__":
    main()