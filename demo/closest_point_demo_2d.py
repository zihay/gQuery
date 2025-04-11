#!/usr/bin/env python
"""
Demo script showing how to use the closest point visualization in PolylineViewer.

This example creates a simple polyline shape and displays an interactive
visualization that allows the user to move a query point and see the closest
point on the polyline in real-time.
"""

import time
from typing import Any, Optional, Tuple
import numpy as np
from examples.polyline_viewer import PolylineViewer
from gquery.shapes.polyline import Polyline
from gquery.core.fwd import *
from gquery.util.obj_loader import load_obj_2d
import polyscope as ps


class ClosestPointVisualizer(PolylineViewer):
    def add_closest_point_visualization(
        self,
        initial_position: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        # Point visualization options
        query_point_name: str = "query_point",
        query_point_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        query_point_radius: float = 0.01,
        closest_point_name: str = "closest_point",
        closest_point_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        closest_point_radius: float = 0.01,
        # Line and circle options
        line_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
        line_width: float = 0.001,
        connection_line_name: str = "connection_line",
        distance_circle_name: str = "distance_circle",
        distance_circle_color: Tuple[float, float, float] = (0.8, 0.4, 0.4),
        distance_circle_width: float = 0.001,
        circle_segments: int = 64,
        # UI options
        window_title: str = "Query Point Controls",
        window_width: int = 380,
        window_height: int = 420,
        show_closest_info: bool = True,
        show_distance_circle: bool = True,
        use_bvh: bool = False,
    ) -> Tuple[Any, np.ndarray, Any, np.ndarray]:
        """
        Add an interactive query point controller with closest point visualization.

        This method adds an interactive query point and displays the closest point on the polyline
        in real-time as the query point is moved. It shows a line connecting the two points
        and optionally a circle centered at the query point with radius equal to the distance.

        Args:
            initial_position: Initial position of the query point
            bounds: Optional (min_bounds, max_bounds) tuple for control limits

            query_point_name: Name for the query point structure
            query_point_color: RGB color tuple for the query point
            query_point_radius: Radius of the query point
            closest_point_name: Name for the closest point structure
            closest_point_color: RGB color tuple for the closest point
            closest_point_radius: Radius of the closest point

            line_color: RGB color tuple for the connecting line
            line_width: Width of the connecting line
            connection_line_name: Name for the connecting line structure
            distance_circle_name: Name for the distance circle structure
            distance_circle_color: RGB color tuple for the distance circle
            distance_circle_width: Width of the distance circle
            circle_segments: Number of segments for circle approximation

            window_title: Title for the control window
            window_width: Width of the control window
            window_height: Height of the control window
            show_closest_info: Whether to show information about the closest point
            show_distance_circle: Whether to show the distance circle visualization
            use_bvh: Initial state for using BVH acceleration

        Returns:
            Tuple containing:
            - The query point structure
            - The query point position array
            - The closest point structure
            - The closest point record from the last update
        """
        # Make sure polyscope is initialized and we have a polyline
        if not self.is_initialized:
            self.init()

        if not hasattr(self, 'polyline'):
            raise ValueError(
                "PolylineViewer instance needs a polyline attribute.")

        # Configure Polyscope options
        ps.set_open_imgui_window_for_user_callback(False)

        # Determine bounds and initial position
        min_bounds, max_bounds, center = self._calculate_bounds(bounds)
        query_point_pos = np.array(
            initial_position) if initial_position is not None else center.copy()

        # Create initial point structures
        query_point = self.add_point_cloud(
            points=np.array([query_point_pos]),
            name=query_point_name,
            color=query_point_color,
            point_radius=query_point_radius
        )

        # Initialize state variables
        current_use_bvh = use_bvh
        last_query_time = 0.0

        # Query the closest point (with BVH if specified)
        query_array = self._position_to_array2(query_point_pos)

        start_time = time.time()
        if current_use_bvh:
            closest_record = self.polyline.closest_point_bvh(query_array)
        else:
            closest_record = self.polyline.closest_point_baseline(query_array)
        dr.eval(closest_record)
        last_query_time = (time.time() - start_time) * \
            1000  # Convert to milliseconds

        # Extract the closest point position as numpy array
        closest_point_pos = np.array([
            closest_record.p.x.numpy(),
            closest_record.p.y.numpy(),
            # Add z-coordinate for 3D viz
            np.zeros_like(closest_record.p.x.numpy())
        ]).T

        # Create the closest point visualization
        closest_point = self.add_point_cloud(
            points=closest_point_pos,
            name=closest_point_name,
            color=closest_point_color,
            point_radius=closest_point_radius
        )

        # Create a line connecting the query point and closest point
        connection_line = self.add_polyline(
            vertices=np.vstack([query_point_pos, closest_point_pos]),
            edges=np.array([[0, 1]]),
            name=connection_line_name,
            color=line_color,
            width=line_width
        )

        # Create a circle visualization for the distance
        distance = closest_record.d.numpy()[0]
        circle_points, circle_edges = self._create_circle_points(
            query_point_pos, distance, circle_segments)

        # Register the circle with Polyscope
        distance_circle = None
        if show_distance_circle:
            distance_circle = self.add_polyline(
                vertices=circle_points,
                edges=circle_edges,
                name=distance_circle_name,
                color=distance_circle_color,
                width=distance_circle_width
            )

        # Keep track of window position and first frame state
        window_pos = np.array([self.window_size[0] - window_width - 20, 30])
        first_frame = True

        # Define the callback function for the GUI
        def closest_point_gui_callback():
            nonlocal query_point_pos, closest_point_pos, window_pos, first_frame
            nonlocal closest_record, current_use_bvh, last_query_time

            # Only set the window position on the first frame to avoid jitter
            if first_frame:
                ps.imgui.SetNextWindowPos((window_pos[0], window_pos[1]))
                ps.imgui.SetNextWindowSize((window_width, window_height))
                first_frame = False

            # Create a standalone ImGui window for the controls
            window_open = True
            opened, _ = ps.imgui.Begin(window_title, window_open)

            if opened:
                # Store the window position for the next frame
                current_pos = ps.imgui.GetWindowPos()
                window_pos = np.array([current_pos[0], current_pos[1]])

                # Header section
                ps.imgui.TextColored((0.4, 0.8, 1.0, 1.0),
                                     "Query Point Controller")
                ps.imgui.Separator()
                ps.imgui.Text("Control the position of the query point")
                ps.imgui.Spacing()

                # Query method selection section
                ps.imgui.TextColored((0.9, 0.6, 0.3, 1.0), "Query Method")
                changed_bvh, current_use_bvh = ps.imgui.Checkbox(
                    "Use BVH Acceleration", current_use_bvh)
                ps.imgui.Spacing()
                ps.imgui.Separator()

                # Position sliders section
                changed_x, query_point_pos[0] = ps.imgui.SliderFloat("X Position",
                                                                     query_point_pos[0],
                                                                     min_bounds[0],
                                                                     max_bounds[0])
                ps.imgui.Spacing()

                changed_y, query_point_pos[1] = ps.imgui.SliderFloat("Y Position",
                                                                     query_point_pos[1],
                                                                     min_bounds[1],
                                                                     max_bounds[1])
                ps.imgui.Spacing()

                # For 3D compatibility, keep z coordinate but fix at 0 for 2D
                query_point_pos[2] = 0.0
                changed_any = changed_x or changed_y or changed_bvh

                # Update visualizations if anything changed
                if changed_any:
                    # Query the closest point with timing
                    query_array = self._position_to_array2(query_point_pos)

                    start_time = time.time()
                    if current_use_bvh:
                        closest_record = self.polyline.closest_point_bvh(
                            query_array)
                    else:
                        closest_record = self.polyline.closest_point_baseline(
                            query_array)
                    last_query_time = (time.time() - start_time) * 1000

                    # Update all visualizations
                    closest_point_pos = self._update_closest_point_visualizations(
                        query_point_pos,
                        closest_record,
                        query_point,
                        closest_point,
                        connection_line,
                        distance_circle,
                        circle_segments,
                        show_distance_circle
                    )

                # Display the current coordinates
                ps.imgui.Separator()
                ps.imgui.Text(
                    f"Query point: ({query_point_pos[0]:.3f}, {query_point_pos[1]:.3f})")
                ps.imgui.Spacing()

                # Add a checkbox to toggle the distance circle if it exists
                if distance_circle is not None:
                    is_visible = distance_circle.is_enabled()
                    changed, new_visible = ps.imgui.Checkbox(
                        "Show Distance Circle", is_visible)
                    if changed:
                        distance_circle.set_enabled(new_visible)
                    ps.imgui.Spacing()

                # Reset button
                if ps.imgui.Button("Reset to Center", (150, 30)):
                    query_point_pos = center.copy()

                    # Always update the query point position first
                    query_point.update_point_positions(
                        np.array([query_point_pos]))

                    # Query the closest point with timing
                    query_array = self._position_to_array2(query_point_pos)

                    start_time = time.time()
                    if current_use_bvh:
                        closest_record = self.polyline.closest_point_bvh(
                            query_array)
                    else:
                        closest_record = self.polyline.closest_point_baseline(
                            query_array)
                    last_query_time = (time.time() - start_time) * 1000

                    # Update all visualizations
                    closest_point_pos = self._update_closest_point_visualizations(
                        query_point_pos,
                        closest_record,
                        query_point,
                        closest_point,
                        connection_line,
                        distance_circle,
                        circle_segments,
                        show_distance_circle
                    )

                # Show closest point information if enabled
                if show_closest_info:
                    ps.imgui.Separator()
                    ps.imgui.TextColored(
                        (0.0, 0.9, 0.3, 1.0), "Closest Point Information")
                    ps.imgui.Spacing()

                    # Get the closest point position for display
                    cp_x = closest_point_pos[0][0]
                    cp_y = closest_point_pos[0][1]

                    # Display all information about the closest point
                    ps.imgui.Text(f"Position: ({cp_x:.3f}, {cp_y:.3f})")
                    ps.imgui.Text(
                        f"Distance: {closest_record.d.numpy()[0]:.5f}")
                    ps.imgui.Text(
                        f"Primitive ID: {closest_record.prim_id.numpy()[0]}")
                    ps.imgui.Text(
                        f"Parameter t: {closest_record.t.numpy()[0]:.3f}")
                    ps.imgui.Text(f"Query Time: {last_query_time:.3f} ms")
                    ps.imgui.Text(
                        f"Method: {'BVH' if current_use_bvh else 'Baseline'}")

            ps.imgui.End()

        # Register the GUI callback
        ps.set_user_callback(closest_point_gui_callback)

        # Return the created structures and data
        return query_point, query_point_pos, closest_point, closest_record

    def _update_closest_point_visualizations(
        self,
        query_point_pos,
        closest_record,
        query_point,
        closest_point,
        connection_line,
        distance_circle=None,
        circle_segments=64,
        show_distance_circle=True
    ):
        """Update all visualization elements for a closest point query."""
        # Extract the closest point position as numpy array
        closest_point_pos = np.array([
            closest_record.p.x.numpy(),
            closest_record.p.y.numpy(),
            # Add z-coordinate for 3D viz
            np.zeros_like(closest_record.p.x.numpy())
        ]).T

        # Update the query and closest points
        query_point.update_point_positions(np.array([query_point_pos]))
        closest_point.update_point_positions(closest_point_pos)

        # Update the connection line
        connection_line.update_node_positions(
            np.vstack([query_point_pos, closest_point_pos])
        )

        # Update the distance circle if enabled
        if show_distance_circle and distance_circle is not None:
            distance = closest_record.d.numpy()[0]
            circle_points, _ = self._create_circle_points(
                query_point_pos, distance, circle_segments)
            distance_circle.update_node_positions(circle_points)

        return closest_point_pos


def run_closest_point_demo():
    """
    Run a demonstration of the closest point visualization feature.
    """
    # Create an example polyline (spiral, star, or rectangle)
    vertices, indices = load_obj_2d(BASE_DIR / "data/bunny2d.obj")
    polyline = Polyline(Array2(vertices.T), Array2i(indices.T))

    # Create a polyline viewer
    viewer = ClosestPointVisualizer(
        polyline=polyline
    )

    # Add the closest point visualization
    viewer.add_closest_point_visualization(
        query_point_color=(1.0, 0.2, 0.2),
        closest_point_color=(0.2, 1.0, 0.2),
        line_color=(0.9, 0.9, 0.9),
        line_width=0.002,
        show_closest_info=True,
        # Enable distance circle visualization
        show_distance_circle=True,
        # Pinkish-red color for the circle
        distance_circle_color=(0.9, 0.5, 0.5),
        # Slightly thinner than the connection line
        distance_circle_width=0.0015,
        circle_segments=60,  # Number of segments for a smoother circle
        use_bvh=True  # Start with BVH acceleration enabled
    )

    # Show the visualization
    viewer.show()


if __name__ == "__main__":
    run_closest_point_demo()
