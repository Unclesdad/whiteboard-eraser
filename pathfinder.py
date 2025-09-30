#!/usr/bin/env python3
"""
Pathfinding Module for Whiteboard Eraser Car
Implements A* pathfinding with Ackermann steering constraints
"""

import numpy as np
import heapq
import time
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import math

@dataclass
class PathNode:
    """Node in the path planning graph"""
    x: float  # mm
    y: float  # mm
    theta: float  # radians
    g_cost: float = float('inf')  # Cost from start
    h_cost: float = 0.0  # Heuristic cost to goal
    f_cost: float = float('inf')  # Total cost (g + h)
    parent: Optional['PathNode'] = None
    steering_angle: float = 0.0  # Steering angle to reach this node
    distance_traveled: float = 0.0  # Distance from parent node

    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost

@dataclass
class WayPoint:
    """Single waypoint in a planned path"""
    x: float  # mm
    y: float  # mm
    theta: float  # radians
    steering_angle: float  # radians (-32° to +32°)
    speed: float  # mm/s
    distance_to_next: float = 0.0  # Distance to next waypoint

@dataclass
class Path:
    """Complete path from start to goal"""
    waypoints: List[WayPoint]
    total_distance: float
    estimated_time: float
    path_id: int

class CarConfig:
    """Car configuration for pathfinding"""
    def __init__(self):
        # Physical dimensions (mm)
        self.wheelbase = 110.0  # Distance between front and back wheels
        self.track_width = 110.0  # Distance between left and right wheels
        self.length = 200.0  # Total car length
        self.width = 150.0  # Total car width

        # Steering constraints
        self.max_steering_angle = np.radians(45.0)  # Max steering angle (±45° from center)
        self.min_turning_radius = self.wheelbase / np.tan(self.max_steering_angle)

        # Motion constraints
        self.max_speed = 200.0  # mm/s
        self.min_speed = 20.0   # mm/s
        self.max_acceleration = 100.0  # mm/s²

        # Gravity-aware motion (whiteboard is vertical)
        self.uphill_speed_factor = 0.65    # 65% speed when going up (against gravity)
        self.downhill_speed_factor = 0.85  # 85% speed when going down (more cautious)
        self.lateral_speed_factor = 1.0    # 100% speed when going left/right
        self.uphill_cost_penalty = 150.0   # Extra cost for uphill paths (mm equivalent)
        self.elevation_change_threshold = 50.0  # mm - minimum change to apply penalties

        # Safety margins
        self.obstacle_clearance = 80.0  # mm clearance from obstacles
        self.edge_clearance = 50.0  # mm clearance from whiteboard edges
        self.top_edge_clearance = 100.0  # Extra clearance from top edge (harder to stop uphill)

        # Planning resolution
        self.position_resolution = 50.0  # mm
        self.angle_resolution = np.radians(15.0)  # radians

class ObstacleMap:
    """Simple obstacle map for path planning"""
    def __init__(self, width_mm: float = 2000.0, height_mm: float = 1500.0):
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.obstacles: List[Tuple[float, float, float]] = []  # (x, y, radius)

    def add_obstacle(self, x: float, y: float, radius: float):
        """Add circular obstacle"""
        self.obstacles.append((x, y, radius))

    def is_point_free(self, x: float, y: float, clearance: float = 0.0) -> bool:
        """Check if point is free of obstacles"""
        # Check bounds
        if (x < clearance or x > self.width_mm - clearance or
            y < clearance or y > self.height_mm - clearance):
            return False

        # Check obstacles
        for ox, oy, radius in self.obstacles:
            if np.sqrt((x - ox)**2 + (y - oy)**2) < radius + clearance:
                return False

        return True

    def is_path_free(self, x1: float, y1: float, x2: float, y2: float,
                     clearance: float = 0.0, num_checks: int = 10) -> bool:
        """Check if straight line path is free"""
        for i in range(num_checks + 1):
            t = i / num_checks
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not self.is_point_free(x, y, clearance):
                return False
        return True

class AckermannPathfinder:
    """
    A* pathfinder with Ackermann steering model constraints and gravity awareness
    """

    def __init__(self, car_config: CarConfig = None, obstacle_map: ObstacleMap = None,
                 whiteboard_up_direction: float = None):
        """
        Initialize pathfinder

        Args:
            car_config: Car configuration parameters
            obstacle_map: Map of obstacles
            whiteboard_up_direction: Angle (radians) pointing up on whiteboard (against gravity)
        """
        self.config = car_config or CarConfig()
        self.obstacle_map = obstacle_map or ObstacleMap()
        self.whiteboard_up_direction = whiteboard_up_direction or (np.pi / 2)  # Default: 90° = up

        # A* search parameters
        self.max_search_nodes = 5000
        self.goal_tolerance_position = 30.0  # mm
        self.goal_tolerance_angle = np.radians(15.0)  # radians

        # Steering angle options for path generation
        self.steering_angles = np.linspace(-self.config.max_steering_angle,
                                         self.config.max_steering_angle, 7)

        # Path smoothing parameters
        self.smoothing_enabled = True
        self.max_smoothing_iterations = 10

        print(f"AckermannPathfinder initialized:")
        print(f"  Min turning radius: {self.config.min_turning_radius:.1f}mm")
        print(f"  Max steering: ±{np.degrees(self.config.max_steering_angle):.1f}°")
        print(f"  Planning resolution: {self.config.position_resolution:.1f}mm")
        print(f"  Gravity-aware: Up direction = {np.degrees(self.whiteboard_up_direction):.1f}°")

    def plan_path(self, start_x: float, start_y: float, start_theta: float,
                  goal_x: float, goal_y: float, goal_theta: float = None,
                  max_planning_time: float = 5.0) -> Optional[Path]:
        """
        Plan path from start to goal using A* with Ackermann constraints

        Args:
            start_x, start_y, start_theta: Start pose
            goal_x, goal_y: Goal position
            goal_theta: Goal orientation (optional)
            max_planning_time: Maximum planning time in seconds

        Returns:
            Path object or None if no path found
        """
        start_time = time.time()

        # Validate start and goal
        if not self.obstacle_map.is_point_free(start_x, start_y, self.config.obstacle_clearance):
            print("Start position is not free")
            return None

        if not self.obstacle_map.is_point_free(goal_x, goal_y, self.config.obstacle_clearance):
            print("Goal position is not free")
            return None

        # Initialize A* search
        start_node = PathNode(start_x, start_y, start_theta, g_cost=0.0)
        start_node.h_cost = self._heuristic_cost(start_node, goal_x, goal_y, goal_theta)
        start_node.f_cost = start_node.g_cost + start_node.h_cost

        open_set = [start_node]
        closed_set: Set[Tuple[int, int, int]] = set()
        node_map = {self._node_key(start_node): start_node}

        nodes_explored = 0

        while open_set and time.time() - start_time < max_planning_time:
            # Get node with lowest f_cost
            current = heapq.heappop(open_set)
            nodes_explored += 1

            # Check if we reached the goal
            if self._is_goal_reached(current, goal_x, goal_y, goal_theta):
                print(f"Path found in {time.time() - start_time:.2f}s, {nodes_explored} nodes explored")
                return self._construct_path(current)

            # Add to closed set
            current_key = self._node_key(current)
            closed_set.add(current_key)

            # Generate successors
            successors = self._generate_successors(current)

            for successor in successors:
                successor_key = self._node_key(successor)

                if successor_key in closed_set:
                    continue

                # Check if position is free
                if not self.obstacle_map.is_point_free(successor.x, successor.y,
                                                     self.config.obstacle_clearance):
                    continue

                # Check if we've seen this node before
                if successor_key in node_map:
                    existing_node = node_map[successor_key]
                    if successor.g_cost < existing_node.g_cost:
                        # Found better path to this node
                        existing_node.g_cost = successor.g_cost
                        existing_node.f_cost = existing_node.g_cost + existing_node.h_cost
                        existing_node.parent = successor.parent
                        existing_node.steering_angle = successor.steering_angle
                        existing_node.distance_traveled = successor.distance_traveled
                else:
                    # New node
                    successor.h_cost = self._heuristic_cost(successor, goal_x, goal_y, goal_theta)
                    successor.f_cost = successor.g_cost + successor.h_cost
                    node_map[successor_key] = successor
                    heapq.heappush(open_set, successor)

            # Limit search size
            if nodes_explored > self.max_search_nodes:
                print(f"Search limit reached ({self.max_search_nodes} nodes)")
                break

        print(f"No path found in {time.time() - start_time:.2f}s, {nodes_explored} nodes explored")
        return None

    def plan_simple_path(self, start_x: float, start_y: float, start_theta: float,
                        goal_x: float, goal_y: float) -> Optional[Path]:
        """
        Plan simple path using direct line with basic obstacle avoidance

        Args:
            start_x, start_y, start_theta: Start pose
            goal_x, goal_y: Goal position

        Returns:
            Simple path or None if blocked
        """
        # Check if direct path is free
        if self.obstacle_map.is_path_free(start_x, start_y, goal_x, goal_y,
                                        self.config.obstacle_clearance):
            # Create simple path
            waypoints = []

            # Start waypoint
            waypoints.append(WayPoint(start_x, start_y, start_theta, 0.0, self.config.max_speed))

            # Calculate required heading to goal
            dx = goal_x - start_x
            dy = goal_y - start_y
            goal_theta = np.arctan2(dy, dx)
            distance = np.sqrt(dx*dx + dy*dy)

            # Add intermediate waypoint for turning if needed
            angle_diff = self._normalize_angle(goal_theta - start_theta)
            if abs(angle_diff) > np.radians(10):  # Need significant turn
                # Add turning waypoint
                turn_distance = min(100.0, distance * 0.3)  # Turn over first 30% or 100mm
                turn_x = start_x + turn_distance * np.cos(start_theta)
                turn_y = start_y + turn_distance * np.sin(start_theta)

                waypoints.append(WayPoint(turn_x, turn_y, goal_theta, 0.0, self.config.min_speed))

            # Goal waypoint
            waypoints.append(WayPoint(goal_x, goal_y, goal_theta, 0.0, self.config.min_speed))

            # Calculate distances
            for i in range(len(waypoints) - 1):
                dx = waypoints[i+1].x - waypoints[i].x
                dy = waypoints[i+1].y - waypoints[i].y
                waypoints[i].distance_to_next = np.sqrt(dx*dx + dy*dy)

            return Path(
                waypoints=waypoints,
                total_distance=distance,
                estimated_time=distance / self.config.max_speed,
                path_id=int(time.time() * 1000) % 100000
            )

        return None

    def smooth_path(self, path: Path) -> Path:
        """Apply path smoothing to reduce sharp turns"""
        if not self.smoothing_enabled or len(path.waypoints) < 3:
            return path

        smoothed_waypoints = path.waypoints.copy()

        for iteration in range(self.max_smoothing_iterations):
            improved = False

            for i in range(1, len(smoothed_waypoints) - 1):
                prev_wp = smoothed_waypoints[i-1]
                curr_wp = smoothed_waypoints[i]
                next_wp = smoothed_waypoints[i+1]

                # Try smoothing this waypoint
                new_x = (prev_wp.x + 2*curr_wp.x + next_wp.x) / 4
                new_y = (prev_wp.y + 2*curr_wp.y + next_wp.y) / 4

                # Check if smoothed position is valid
                if (self.obstacle_map.is_point_free(new_x, new_y, self.config.obstacle_clearance) and
                    self.obstacle_map.is_path_free(prev_wp.x, prev_wp.y, new_x, new_y,
                                                 self.config.obstacle_clearance) and
                    self.obstacle_map.is_path_free(new_x, new_y, next_wp.x, next_wp.y,
                                                 self.config.obstacle_clearance)):

                    # Apply smoothing
                    smoothed_waypoints[i].x = new_x
                    smoothed_waypoints[i].y = new_y
                    improved = True

            if not improved:
                break

        # Recalculate distances and headings
        for i in range(len(smoothed_waypoints)):
            if i < len(smoothed_waypoints) - 1:
                dx = smoothed_waypoints[i+1].x - smoothed_waypoints[i].x
                dy = smoothed_waypoints[i+1].y - smoothed_waypoints[i].y
                smoothed_waypoints[i].distance_to_next = np.sqrt(dx*dx + dy*dy)
                smoothed_waypoints[i].theta = np.arctan2(dy, dx)

        return Path(
            waypoints=smoothed_waypoints,
            total_distance=path.total_distance,
            estimated_time=path.estimated_time,
            path_id=path.path_id
        )

    def calculate_path_cost(self, path: Path) -> float:
        """Calculate total cost of a path (distance + turn penalties)"""
        total_cost = 0.0
        turn_penalty_factor = 100.0  # mm equivalent penalty per radian

        for i in range(len(path.waypoints) - 1):
            # Distance cost
            total_cost += path.waypoints[i].distance_to_next

            # Turn penalty
            if i > 0:
                prev_theta = path.waypoints[i-1].theta
                curr_theta = path.waypoints[i].theta
                turn_angle = abs(self._normalize_angle(curr_theta - prev_theta))
                total_cost += turn_angle * turn_penalty_factor

        return total_cost

    def get_elevation_change(self, from_x: float, from_y: float, to_x: float, to_y: float) -> float:
        """
        Calculate elevation change from one point to another
        Positive = going up (against gravity), Negative = going down

        Args:
            from_x, from_y: Starting position
            to_x, to_y: Ending position

        Returns:
            Elevation change in mm (positive = uphill, negative = downhill)
        """
        # Calculate movement vector
        dx = to_x - from_x
        dy = to_y - from_y

        if abs(dx) < 1.0 and abs(dy) < 1.0:
            return 0.0  # No movement

        # Calculate movement direction
        movement_angle = np.arctan2(dy, dx)

        # Calculate angle relative to "up" direction (against gravity)
        up_vector_angle = self.whiteboard_up_direction
        angle_to_up = self._normalize_angle(movement_angle - up_vector_angle)

        # Calculate distance moved
        distance = np.sqrt(dx*dx + dy*dy)

        # Project movement onto vertical axis (up/down component)
        elevation_change = distance * np.cos(angle_to_up)

        return elevation_change

    def get_movement_type(self, from_x: float, from_y: float, to_x: float, to_y: float) -> str:
        """
        Determine movement type: uphill, downhill, or lateral

        Returns:
            'uphill', 'downhill', or 'lateral'
        """
        elevation_change = self.get_elevation_change(from_x, from_y, to_x, to_y)

        if elevation_change > self.config.elevation_change_threshold:
            return 'uphill'
        elif elevation_change < -self.config.elevation_change_threshold:
            return 'downhill'
        else:
            return 'lateral'

    def get_speed_factor_for_movement(self, from_x: float, from_y: float,
                                    to_x: float, to_y: float) -> float:
        """
        Get speed factor based on movement type

        Returns:
            Speed multiplier (0.0 to 1.0)
        """
        movement_type = self.get_movement_type(from_x, from_y, to_x, to_y)

        if movement_type == 'uphill':
            return self.config.uphill_speed_factor
        elif movement_type == 'downhill':
            return self.config.downhill_speed_factor
        else:
            return self.config.lateral_speed_factor

    def _generate_successors(self, node: PathNode) -> List[PathNode]:
        """Generate successor nodes using Ackermann steering model"""
        successors = []
        step_distance = self.config.position_resolution

        for steering_angle in self.steering_angles:
            # Calculate motion using Ackermann steering model
            if abs(steering_angle) < 1e-6:
                # Straight motion
                new_x = node.x + step_distance * np.cos(node.theta)
                new_y = node.y + step_distance * np.sin(node.theta)
                new_theta = node.theta
            else:
                # Curved motion
                turning_radius = self.config.wheelbase / np.tan(abs(steering_angle))
                angular_velocity = (1.0 if steering_angle > 0 else -1.0) / turning_radius

                # Calculate arc motion
                arc_length = step_distance
                delta_theta = arc_length * angular_velocity

                # Center of rotation relative to rear axle
                center_x = node.x - turning_radius * np.sin(node.theta) * np.sign(steering_angle)
                center_y = node.y + turning_radius * np.cos(node.theta) * np.sign(steering_angle)

                # New position after arc
                new_theta = node.theta + delta_theta
                new_x = center_x + turning_radius * np.sin(new_theta) * np.sign(steering_angle)
                new_y = center_y - turning_radius * np.cos(new_theta) * np.sign(steering_angle)

            # Create successor node
            successor = PathNode(new_x, new_y, new_theta)
            successor.parent = node
            successor.steering_angle = steering_angle
            successor.distance_traveled = step_distance
            successor.g_cost = node.g_cost + step_distance

            # Add turn penalty
            if node.parent is not None:
                angle_change = abs(self._normalize_angle(new_theta - node.theta))
                successor.g_cost += angle_change * 50.0  # Turn penalty

            # Add gravity-aware elevation penalty
            elevation_change = self.get_elevation_change(node.x, node.y, new_x, new_y)
            if elevation_change > self.config.elevation_change_threshold:
                # Uphill movement - add significant penalty
                successor.g_cost += self.config.uphill_cost_penalty
            elif elevation_change < -self.config.elevation_change_threshold:
                # Downhill movement - small penalty for being more cautious
                successor.g_cost += 25.0

            successors.append(successor)

        return successors

    def _heuristic_cost(self, node: PathNode, goal_x: float, goal_y: float,
                       goal_theta: Optional[float] = None) -> float:
        """Calculate heuristic cost (admissible lower bound)"""
        # Euclidean distance
        dx = goal_x - node.x
        dy = goal_y - node.y
        distance = np.sqrt(dx*dx + dy*dy)

        # Add orientation cost if goal orientation is specified
        if goal_theta is not None:
            angle_diff = abs(self._normalize_angle(goal_theta - node.theta))
            # Convert angle difference to equivalent distance
            distance += angle_diff * 100.0  # 100mm per radian

        return distance

    def _is_goal_reached(self, node: PathNode, goal_x: float, goal_y: float,
                        goal_theta: Optional[float] = None) -> bool:
        """Check if node is close enough to goal"""
        # Position tolerance
        dx = goal_x - node.x
        dy = goal_y - node.y
        distance = np.sqrt(dx*dx + dy*dy)

        if distance > self.goal_tolerance_position:
            return False

        # Orientation tolerance (if specified)
        if goal_theta is not None:
            angle_diff = abs(self._normalize_angle(goal_theta - node.theta))
            if angle_diff > self.goal_tolerance_angle:
                return False

        return True

    def _node_key(self, node: PathNode) -> Tuple[int, int, int]:
        """Generate unique key for node based on discretized position and orientation"""
        x_key = int(round(node.x / self.config.position_resolution))
        y_key = int(round(node.y / self.config.position_resolution))
        theta_key = int(round(node.theta / self.config.angle_resolution))
        return (x_key, y_key, theta_key)

    def _construct_path(self, goal_node: PathNode) -> Path:
        """Construct path from goal node by following parent pointers"""
        waypoints = []
        current = goal_node
        total_distance = 0.0

        # Build path backwards
        path_nodes = []
        while current is not None:
            path_nodes.append(current)
            if current.parent is not None:
                total_distance += current.distance_traveled
            current = current.parent

        # Reverse to get forward path
        path_nodes.reverse()

        # Convert to waypoints
        for i, node in enumerate(path_nodes):
            # Calculate appropriate speed (slower for turns)
            if abs(node.steering_angle) > np.radians(10):
                base_speed = self.config.min_speed
            else:
                base_speed = self.config.max_speed

            # Apply gravity-aware speed adjustment
            if i < len(path_nodes) - 1:
                next_node = path_nodes[i + 1]
                speed_factor = self.get_speed_factor_for_movement(
                    node.x, node.y, next_node.x, next_node.y
                )
                speed = base_speed * speed_factor
            else:
                speed = base_speed * self.config.lateral_speed_factor

            waypoint = WayPoint(
                x=node.x,
                y=node.y,
                theta=node.theta,
                steering_angle=node.steering_angle,
                speed=speed
            )

            if i < len(path_nodes) - 1:
                next_node = path_nodes[i + 1]
                dx = next_node.x - node.x
                dy = next_node.y - node.y
                waypoint.distance_to_next = np.sqrt(dx*dx + dy*dy)

            waypoints.append(waypoint)

        # Estimate travel time
        estimated_time = sum(wp.distance_to_next / wp.speed for wp in waypoints[:-1])

        return Path(
            waypoints=waypoints,
            total_distance=total_distance,
            estimated_time=estimated_time,
            path_id=int(time.time() * 1000) % 100000
        )

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


# Test function
def test_pathfinder():
    """Test the pathfinding system"""
    print("Testing AckermannPathfinder...")

    # Create test environment
    obstacle_map = ObstacleMap(1000, 800)
    obstacle_map.add_obstacle(300, 300, 80)  # Circular obstacle
    obstacle_map.add_obstacle(600, 200, 60)

    pathfinder = AckermannPathfinder(obstacle_map=obstacle_map)

    # Test simple path (no obstacles)
    print("\nTesting simple path...")
    simple_path = pathfinder.plan_simple_path(100, 100, 0, 200, 150)
    if simple_path:
        print(f"Simple path: {len(simple_path.waypoints)} waypoints, {simple_path.total_distance:.1f}mm")
        for i, wp in enumerate(simple_path.waypoints):
            print(f"  WP{i}: ({wp.x:.1f}, {wp.y:.1f}), θ={np.degrees(wp.theta):.1f}°")

    # Test A* path (with obstacles)
    print("\nTesting A* path around obstacles...")
    start_time = time.time()
    astar_path = pathfinder.plan_path(100, 100, 0, 700, 300)
    planning_time = time.time() - start_time

    if astar_path:
        print(f"A* path found in {planning_time:.2f}s:")
        print(f"  {len(astar_path.waypoints)} waypoints")
        print(f"  Total distance: {astar_path.total_distance:.1f}mm")
        print(f"  Estimated time: {astar_path.estimated_time:.2f}s")

        # Show first few waypoints
        for i, wp in enumerate(astar_path.waypoints[:5]):
            print(f"  WP{i}: ({wp.x:.1f}, {wp.y:.1f}), θ={np.degrees(wp.theta):.1f}°, "
                  f"steer={np.degrees(wp.steering_angle):.1f}°")

        # Test path smoothing
        print("\nApplying path smoothing...")
        smoothed_path = pathfinder.smooth_path(astar_path)
        print(f"Smoothed path: {len(smoothed_path.waypoints)} waypoints")

        # Calculate costs
        original_cost = pathfinder.calculate_path_cost(astar_path)
        smoothed_cost = pathfinder.calculate_path_cost(smoothed_path)
        print(f"Original cost: {original_cost:.1f}, Smoothed cost: {smoothed_cost:.1f}")

    else:
        print("No A* path found")

    print("\nPathfinder test complete!")


if __name__ == "__main__":
    test_pathfinder()