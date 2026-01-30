#!/usr/bin/env python3
"""
Global Mapping System for Whiteboard Eraser Car
Maintains a map of detected markings and tracks erasing progress
"""

import numpy as np
import time
import threading
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json

@dataclass
class GlobalMarking:
    """Represents a marking in global coordinates"""
    id: int
    x: float  # mm in global frame
    y: float  # mm in global frame
    radius: float = 15.0  # mm (approximate marking size)
    confidence: float = 1.0  # Combined confidence from all observations
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    observation_count: int = 1
    is_erased: bool = False
    erased_time: Optional[float] = None

    def update_observation(self, new_x: float, new_y: float, new_confidence: float):
        """Update marking with new observation"""
        # Weighted average of position (more weight to higher confidence observations)
        total_weight = self.confidence * self.observation_count + new_confidence

        if total_weight > 0:
            self.x = (self.x * self.confidence * self.observation_count +
                     new_x * new_confidence) / total_weight
            self.y = (self.y * self.confidence * self.observation_count +
                     new_y * new_confidence) / total_weight

        # Update confidence and timing
        self.confidence = min(1.0, total_weight / (self.observation_count + 1))
        self.observation_count += 1
        self.last_seen = time.time()

    def mark_erased(self):
        """Mark this marking as erased"""
        self.is_erased = True
        self.erased_time = time.time()

    def distance_to(self, x: float, y: float) -> float:
        """Calculate distance to a point"""
        return np.sqrt((self.x - x)**2 + (self.y - y)**2)

    def is_within_erase_radius(self, x: float, y: float, erase_radius: float = 25.0) -> bool:
        """Check if a position is within erasing range"""
        return self.distance_to(x, y) <= erase_radius

@dataclass
class MarkingCluster:
    """Represents a cluster of closely spaced markings"""
    markings: List[GlobalMarking]
    center_x: float
    center_y: float
    radius: float

    def update_center(self):
        """Recalculate cluster center"""
        if not self.markings:
            return

        active_markings = [m for m in self.markings if not m.is_erased]
        if not active_markings:
            return

        self.center_x = np.mean([m.x for m in active_markings])
        self.center_y = np.mean([m.y for m in active_markings])

        # Calculate cluster radius (max distance from center + marking radius)
        max_dist = max([m.distance_to(self.center_x, self.center_y) for m in active_markings])
        self.radius = max_dist + 20.0  # Add buffer

class GlobalMap:
    """
    Maintains a global map of whiteboard markings
    """

    def __init__(self,
                 merge_distance_mm: float = 30.0,
                 min_confidence: float = 0.3,
                 max_age_seconds: float = 300.0,  # 5 minutes
                 erase_confirmation_distance: float = 25.0):
        """
        Initialize global mapping system

        Args:
            merge_distance_mm: Distance within which markings are considered the same
            min_confidence: Minimum confidence to keep a marking
            max_age_seconds: Maximum age before removing unconfirmed markings
            erase_confirmation_distance: Distance within which marking is considered erased
        """
        self.merge_distance_mm = merge_distance_mm
        self.min_confidence = min_confidence
        self.max_age_seconds = max_age_seconds
        self.erase_confirmation_distance = erase_confirmation_distance

        # Thread-safe storage
        self.markings: List[GlobalMarking] = []
        self.next_id = 1
        self.map_lock = threading.Lock()

        # Clustering for efficient pathfinding
        self.clusters: List[MarkingCluster] = []
        self.cluster_distance_threshold = 80.0  # mm

        # Statistics
        self.total_detected = 0
        self.total_erased = 0
        self.false_positives_removed = 0

        print(f"GlobalMap initialized:")
        print(f"  Merge distance: {merge_distance_mm}mm")
        print(f"  Min confidence: {min_confidence}")
        print(f"  Erase distance: {erase_confirmation_distance}mm")

    def add_markings(self, detections: List[Tuple[float, float, float]]) -> List[int]:
        """
        Add new marking detections to the global map

        Args:
            detections: List of (x, y, confidence) tuples in global coordinates

        Returns:
            List of marking IDs that were added or updated
        """
        if not detections:
            return []

        updated_ids = []

        with self.map_lock:
            for x, y, confidence in detections:
                if confidence < self.min_confidence:
                    continue

                # Find existing marking within merge distance
                existing_marking = self._find_nearest_marking(x, y, self.merge_distance_mm)

                if existing_marking is not None:
                    # Update existing marking
                    existing_marking.update_observation(x, y, confidence)
                    updated_ids.append(existing_marking.id)
                else:
                    # Create new marking
                    new_marking = GlobalMarking(
                        id=self.next_id,
                        x=x,
                        y=y,
                        confidence=confidence
                    )
                    self.markings.append(new_marking)
                    updated_ids.append(new_marking.id)
                    self.next_id += 1
                    self.total_detected += 1

            # Clean up old or low-confidence markings
            self._cleanup_markings()

            # Update clustering
            self._update_clusters()

        return updated_ids

    def mark_area_erased(self, x: float, y: float, erase_radius: float = None) -> List[int]:
        """
        Mark markings in an area as erased

        Args:
            x: X coordinate of erased area center
            y: Y coordinate of erased area center
            erase_radius: Radius of erased area (default uses class setting)

        Returns:
            List of IDs of markings that were marked as erased
        """
        if erase_radius is None:
            erase_radius = self.erase_confirmation_distance

        erased_ids = []

        with self.map_lock:
            for marking in self.markings:
                if not marking.is_erased and marking.is_within_erase_radius(x, y, erase_radius):
                    marking.mark_erased()
                    erased_ids.append(marking.id)
                    self.total_erased += 1

            # Update clusters after erasing
            self._update_clusters()

        return erased_ids

    def get_active_markings(self) -> List[GlobalMarking]:
        """Get all markings that haven't been erased"""
        with self.map_lock:
            return [m for m in self.markings if not m.is_erased]

    def get_all_markings(self) -> List[GlobalMarking]:
        """Get all markings including erased ones"""
        with self.map_lock:
            return self.markings.copy()

    def get_nearest_unerased_marking(self, x: float, y: float) -> Optional[GlobalMarking]:
        """Get the nearest marking that hasn't been erased"""
        active_markings = self.get_active_markings()

        if not active_markings:
            return None

        nearest = min(active_markings, key=lambda m: m.distance_to(x, y))
        return nearest

    def get_markings_in_radius(self, x: float, y: float, radius: float,
                              include_erased: bool = False) -> List[GlobalMarking]:
        """Get all markings within a radius of a point"""
        markings_to_check = self.markings if include_erased else self.get_active_markings()
        return [m for m in markings_to_check if m.distance_to(x, y) <= radius]

    def get_clusters(self, include_erased: bool = False) -> List[MarkingCluster]:
        """Get marking clusters for efficient pathfinding"""
        with self.map_lock:
            if include_erased:
                return self.clusters.copy()
            else:
                # Return only clusters with active markings
                active_clusters = []
                for cluster in self.clusters:
                    active_markings = [m for m in cluster.markings if not m.is_erased]
                    if active_markings:
                        # Create new cluster with only active markings
                        new_cluster = MarkingCluster(
                            markings=active_markings,
                            center_x=cluster.center_x,
                            center_y=cluster.center_y,
                            radius=cluster.radius
                        )
                        new_cluster.update_center()
                        active_clusters.append(new_cluster)
                return active_clusters

    def get_erase_targets_ordered(self, current_x: float, current_y: float) -> List[Tuple[float, float]]:
        """
        Get ordered list of targets to erase, prioritized by distance and clustering

        Args:
            current_x: Current X position
            current_y: Current Y position

        Returns:
            List of (x, y) coordinates ordered by priority
        """
        clusters = self.get_clusters(include_erased=False)

        if not clusters:
            # No clusters, return individual markings
            markings = self.get_active_markings()
            markings.sort(key=lambda m: m.distance_to(current_x, current_y))
            return [(m.x, m.y) for m in markings]

        # Sort clusters by distance from current position
        clusters.sort(key=lambda c: np.sqrt((c.center_x - current_x)**2 + (c.center_y - current_y)**2))

        targets = []
        for cluster in clusters:
            # For each cluster, add center as primary target
            targets.append((cluster.center_x, cluster.center_y))

        return targets

    def estimate_completion_progress(self) -> dict:
        """Estimate erasing progress"""
        total_markings = len(self.markings)
        erased_markings = len([m for m in self.markings if m.is_erased])

        return {
            'total_detected': total_markings,
            'total_erased': erased_markings,
            'remaining': total_markings - erased_markings,
            'progress_percent': (erased_markings / max(1, total_markings)) * 100,
            'false_positives_removed': self.false_positives_removed
        }

    def save_map(self, filename: str):
        """Save map to JSON file"""
        with self.map_lock:
            map_data = {
                'markings': [
                    {
                        'id': m.id,
                        'x': m.x,
                        'y': m.y,
                        'radius': m.radius,
                        'confidence': m.confidence,
                        'first_seen': m.first_seen,
                        'last_seen': m.last_seen,
                        'observation_count': m.observation_count,
                        'is_erased': m.is_erased,
                        'erased_time': m.erased_time
                    }
                    for m in self.markings
                ],
                'stats': self.estimate_completion_progress(),
                'timestamp': time.time()
            }

        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=2)

        print(f"Map saved to {filename}")

    def load_map(self, filename: str):
        """Load map from JSON file"""
        try:
            with open(filename, 'r') as f:
                map_data = json.load(f)

            with self.map_lock:
                self.markings.clear()

                for marking_data in map_data['markings']:
                    marking = GlobalMarking(
                        id=marking_data['id'],
                        x=marking_data['x'],
                        y=marking_data['y'],
                        radius=marking_data['radius'],
                        confidence=marking_data['confidence'],
                        first_seen=marking_data['first_seen'],
                        last_seen=marking_data['last_seen'],
                        observation_count=marking_data['observation_count'],
                        is_erased=marking_data['is_erased'],
                        erased_time=marking_data.get('erased_time')
                    )
                    self.markings.append(marking)

                # Update next ID
                if self.markings:
                    self.next_id = max(m.id for m in self.markings) + 1

                self._update_clusters()

            print(f"Map loaded from {filename}: {len(self.markings)} markings")

        except Exception as e:
            print(f"Error loading map: {e}")

    def get_map_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of all markings (min_x, min_y, max_x, max_y)"""
        if not self.markings:
            return (0, 0, 0, 0)

        x_coords = [m.x for m in self.markings]
        y_coords = [m.y for m in self.markings]

        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    def get_diagnostics(self) -> dict:
        """Get detailed diagnostics about the map state"""
        with self.map_lock:
            active_markings = [m for m in self.markings if not m.is_erased]

            # Calculate average confidence
            if active_markings:
                avg_confidence = np.mean([m.confidence for m in active_markings])
                avg_observations = np.mean([m.observation_count for m in active_markings])
            else:
                avg_confidence = 0
                avg_observations = 0

            bounds = self.get_map_bounds()

            return {
                'total_markings': len(self.markings),
                'active_markings': len(active_markings),
                'erased_markings': len([m for m in self.markings if m.is_erased]),
                'clusters': len(self.clusters),
                'avg_confidence': avg_confidence,
                'avg_observations': avg_observations,
                'map_bounds': {
                    'min_x': bounds[0], 'min_y': bounds[1],
                    'max_x': bounds[2], 'max_y': bounds[3]
                },
                'false_positives_removed': self.false_positives_removed
            }

    def _find_nearest_marking(self, x: float, y: float, max_distance: float) -> Optional[GlobalMarking]:
        """Find nearest marking within max_distance"""
        nearest = None
        min_distance = float('inf')

        for marking in self.markings:
            if marking.is_erased:
                continue

            distance = marking.distance_to(x, y)
            if distance <= max_distance and distance < min_distance:
                min_distance = distance
                nearest = marking

        return nearest

    def _cleanup_markings(self):
        """Remove old or low-confidence markings"""
        current_time = time.time()
        before_count = len(self.markings)

        # Remove markings that are too old or have too low confidence
        self.markings = [
            m for m in self.markings
            if (m.confidence >= self.min_confidence or
                current_time - m.last_seen <= self.max_age_seconds)
        ]

        removed = before_count - len(self.markings)
        if removed > 0:
            self.false_positives_removed += removed

    def _update_clusters(self):
        """Update marking clusters for efficient pathfinding"""
        active_markings = [m for m in self.markings if not m.is_erased]

        if not active_markings:
            self.clusters.clear()
            return

        # Simple clustering algorithm
        self.clusters.clear()
        unassigned = active_markings.copy()

        while unassigned:
            # Start new cluster with first unassigned marking
            seed = unassigned.pop(0)
            cluster_markings = [seed]

            # Find all markings within cluster distance
            to_remove = []
            for i, marking in enumerate(unassigned):
                if seed.distance_to(marking.x, marking.y) <= self.cluster_distance_threshold:
                    cluster_markings.append(marking)
                    to_remove.append(i)

            # Remove assigned markings
            for i in reversed(to_remove):
                unassigned.pop(i)

            # Create cluster
            center_x = np.mean([m.x for m in cluster_markings])
            center_y = np.mean([m.y for m in cluster_markings])
            max_dist = max([m.distance_to(center_x, center_y) for m in cluster_markings])

            cluster = MarkingCluster(
                markings=cluster_markings,
                center_x=center_x,
                center_y=center_y,
                radius=max_dist + 20.0
            )
            self.clusters.append(cluster)


# Test function
def test_global_map():
    """Test the global mapping system"""
    print("Testing GlobalMap...")

    global_map = GlobalMap()

    # Add some test markings
    print("\nAdding test markings...")
    detections1 = [(100, 200, 0.8), (150, 220, 0.7), (300, 100, 0.9)]
    ids1 = global_map.add_markings(detections1)
    print(f"Added markings with IDs: {ids1}")

    # Add overlapping detection (should merge)
    detections2 = [(105, 205, 0.6)]  # Close to first marking
    ids2 = global_map.add_markings(detections2)
    print(f"Updated markings with IDs: {ids2}")

    # Get all markings
    markings = global_map.get_all_markings()
    print(f"\nAll markings ({len(markings)}):")
    for m in markings:
        print(f"  ID {m.id}: ({m.x:.1f}, {m.y:.1f}), conf={m.confidence:.2f}, obs={m.observation_count}")

    # Test erasing
    print(f"\nMarking area around (100, 200) as erased...")
    erased_ids = global_map.mark_area_erased(100, 200, 30)
    print(f"Erased markings: {erased_ids}")

    # Get remaining markings
    active = global_map.get_active_markings()
    print(f"Active markings remaining: {len(active)}")

    # Test clustering
    clusters = global_map.get_clusters()
    print(f"\nClusters: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i}: center=({cluster.center_x:.1f}, {cluster.center_y:.1f}), "
              f"radius={cluster.radius:.1f}, markings={len(cluster.markings)}")

    # Test target ordering
    targets = global_map.get_erase_targets_ordered(0, 0)
    print(f"\nErase targets from origin: {targets}")

    # Print diagnostics
    diag = global_map.get_diagnostics()
    print(f"\nDiagnostics: {diag}")

    # Test save/load
    global_map.save_map("test_map.json")

    # Create new map and load
    new_map = GlobalMap()
    new_map.load_map("test_map.json")
    print(f"Loaded map has {len(new_map.get_all_markings())} markings")


if __name__ == "__main__":
    test_global_map()