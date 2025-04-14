"""
Polyline visualization tools for diff_wost.
"""

import polyscope as ps
import numpy as np
import time
from typing import Optional, List, Tuple, Dict, Any, Union
import matplotlib.pyplot as plt

from demo.visualizer import Visualizer
from gquery.shapes.polyline import Polyline
from gquery.core.fwd import Float, Array2


class PolylineViewer(Visualizer):
    """
    Class for visualizing polyline data with Polyscope.

    This class provides functionality to visualize and inspect polylines,
    including vertices, edges, and associated scalar/vector fields.
    """

    def __init__(
        self,
        polyline: Polyline,
        window_size: Tuple[int, int] = (1024, 768),
        background_color: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        auto_init: bool = True,
    ):
        """
        Initialize the polyline viewer.

        Args:
            polyline: The polyline to visualize
            window_size: Tuple of (width, height) for the polyscope window
            background_color: Background color as RGB tuple of floats between 0 and 1
            auto_init: Automatically initialize Polyscope
        """
        # Create a new visualizer
        super().__init__(window_size, background_color, auto_init)
        self.add_polyline(polyline.vertices.numpy().T,
                          polyline.indices.numpy().T)
        self.polyline = polyline

    def _calculate_bounds(self, initial_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Helper to calculate bounds for UI controls based on scene content."""
        if initial_bounds is not None:
            min_bounds, max_bounds = initial_bounds
            center = (min_bounds + max_bounds) / 2
            return min_bounds, max_bounds, center

        # Try to determine bounds from existing scene structures
        all_points = []
        for _, structure in self._structures.items():
            if hasattr(structure, 'get_vertices') and callable(getattr(structure, 'get_vertices')):
                try:
                    points = structure.get_vertices()
                    if points is not None and len(points) > 0:
                        all_points.append(points)
                except:
                    pass

        if all_points:
            # Concatenate all points and compute bounds
            combined_points = np.vstack(all_points)
            min_bounds = np.min(combined_points, axis=0)
            max_bounds = np.max(combined_points, axis=0)
            center = (min_bounds + max_bounds) / 2
        else:
            # Default if no structures with vertices found
            center = np.array([0.0, 0.0, 0.0])
            size = np.array([2.0, 2.0, 2.0])
            min_bounds = center - size/2
            max_bounds = center + size/2

        return min_bounds, max_bounds, center

    def _create_circle_points(self, center, radius, segments):
        """Helper to create points and edges for a circle visualization."""
        # Generate points around a circle
        theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)

        # Handle z-coordinate for 2D or 3D points
        if center.size > 2:
            z = np.zeros_like(x) + center[2]
        else:
            z = np.zeros_like(x)

        points = np.column_stack([x, y, z])

        # Create edges connecting consecutive points and closing the circle
        edges = np.column_stack(
            [np.arange(segments), np.roll(np.arange(segments), -1)])

        return points, edges

    def _position_to_array2(self, position):
        """Convert a numpy position to a diff_wost Array2 for queries."""
        return Array2(float(position[0]), float(position[1]))

    def set_up_camera(
        self,
        camera_position: np.ndarray = np.array([0, 0, 5]),
        look_at: np.ndarray = np.array([0, 0, 0]),
        up_direction: np.ndarray = np.array([0, 1, 0])
    ):
        """
        Set up the camera view for the visualization.

        Args:
            camera_position: Position of the camera in 3D space
            look_at: Point the camera is looking at
            up_direction: The up direction for the camera
        """
        # Make sure polyscope is initialized
        if not self.is_initialized:
            self.init()

        # Set the camera position and orientation
        ps.look_at(camera_position, look_at, up_direction)
