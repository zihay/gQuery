#!/usr/bin/env python
"""
Demo script showing how to use the ray intersection visualization in PolylineViewer.

This example creates a simple polyline shape and displays an interactive
visualization that allows the user to control a ray and see its intersection
with the polyline in real-time.
"""

import numpy as np
from examples.polyline_viewer import PolylineViewer
from gquery.shapes.polyline import Polyline
from gquery.core.fwd import *
import polyscope as ps
from typing import Optional, Tuple, Any
import time

from gquery.shapes.primitive import Intersection
from gquery.util.obj_loader import load_obj_2d


class RayIntersectionVisualizer(PolylineViewer):
    def add_ray_intersection_visualization(
        self,
        initial_origin: Optional[np.ndarray] = None,
        initial_direction: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        # Ray and point visualization options
        ray_origin_name: str = "ray_origin",
        ray_origin_color: Tuple[float, float, float] = (1.0, 0.5, 0.0),
        ray_origin_radius: float = 0.01,
        intersection_point_name: str = "intersection_point",
        intersection_point_color: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        intersection_point_radius: float = 0.01,
        ray_line_name: str = "ray_line",
        ray_line_color: Tuple[float, float, float] = (1.0, 0.8, 0.0),
        ray_line_width: float = 0.001,
        ray_length: float = 10.0,
        # Ray intersection parameters
        initial_r_max: float = float('inf'),
        use_bvh: bool = False,
        # UI options
        window_title: str = "Ray Controls",
        window_width: int = 380,
        window_height: int = 820,
        show_intersection_info: bool = True,
    ) -> Tuple[Any, np.ndarray, np.ndarray, Any, Any]:
        """
        Add an interactive ray with intersection visualization.

        This method adds an interactive ray and displays the intersection point
        on the polyline in real-time as the ray is moved or redirected.

        Args:
            initial_origin: Initial origin point of the ray
            initial_direction: Initial direction vector of the ray
            bounds: Optional (min_bounds, max_bounds) tuple for control limits

            ray_origin_name: Name for the ray origin point structure
            ray_origin_color: RGB color tuple for the ray origin point
            ray_origin_radius: Radius of the ray origin point
            intersection_point_name: Name for the intersection point structure
            intersection_point_color: RGB color tuple for the intersection point
            intersection_point_radius: Radius of the intersection point

            ray_line_name: Name for the ray line structure
            ray_line_color: RGB color tuple for the ray line
            ray_line_width: Width of the ray line
            ray_length: Length of the ray for visualization purposes

            initial_r_max: Initial maximum ray distance for intersection queries
            use_bvh: Initial state for using BVH acceleration for ray intersections

            window_title: Title for the control window
            window_width: Width of the control window
            window_height: Height of the control window
            show_intersection_info: Whether to show information about the intersection

        Returns:
            Tuple containing:
            - The ray origin point structure
            - The ray origin position array
            - The ray direction array
            - The intersection point structure
            - The intersection record from the last update
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

        # Initialize ray origin and direction
        ray_origin_pos = np.array(
            initial_origin) if initial_origin is not None else center.copy()
        # Default direction: pointing right (1, 0, 0)
        ray_direction = np.array(
            initial_direction) if initial_direction is not None else np.array([1.0, 0.0, 0.0])
        # Normalize the direction vector
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        # Initialize r_max and BVH usage
        r_max = initial_r_max
        current_use_bvh = use_bvh

        # Create ray origin point structure
        ray_origin = self.add_point_cloud(
            points=np.array([ray_origin_pos]),
            name=ray_origin_name,
            color=ray_origin_color,
            point_radius=ray_origin_radius
        )

        query_origin = self._position_to_array2(ray_origin_pos)
        query_direction = Array2(
            float(ray_direction[0]), float(ray_direction[1]))

        # Query the intersection with r_max and use_bvh
        start_time = time.time()
        if current_use_bvh:
            intersection_record = self.polyline.intersect_bvh(
                query_origin, query_direction, Float(r_max))
        else:
            intersection_record = self.polyline.intersect_baseline(
                query_origin, query_direction, Float(r_max))
        dr.eval(intersection_record)
        last_query_time = (time.time() - start_time) * \
            1000  # Convert to milliseconds

        # Check if there's a valid intersection
        is_valid_intersection = bool(
            intersection_record.valid.numpy()[0])

        # Calculate the end point of the ray line and intersection point
        if is_valid_intersection:
            # Use the intersection distance
            intersection_distance = float(
                intersection_record.d.numpy()[0])
            # Limit to ray_length if needed
            ray_line_end = ray_origin_pos + ray_direction * \
                min(intersection_distance, ray_length)

            # Extract the intersection point position
            intersection_point_pos = np.array([
                intersection_record.p.x.numpy(),
                intersection_record.p.y.numpy(),
                # Add z-coordinate for 3D viz
                np.zeros_like(intersection_record.p.x.numpy())
            ]).T
        else:
            # Just use the ray_length for visualization
            ray_line_end = ray_origin_pos + ray_direction * ray_length
            # Place a dummy intersection point at the end of the ray
            intersection_point_pos = np.array([ray_line_end])

        # Create the ray line visualization
        ray_line = self.add_polyline(
            vertices=np.vstack([ray_origin_pos, ray_line_end]),
            edges=np.array([[0, 1]]),
            name=ray_line_name,
            color=ray_line_color,
            width=ray_line_width
        )

        # Create the intersection point visualization
        intersection_point = self.add_point_cloud(
            points=intersection_point_pos,
            name=intersection_point_name,
            color=intersection_point_color,
            point_radius=intersection_point_radius
        )

        # Hide the intersection point if there's no valid intersection
        if not is_valid_intersection:
            intersection_point.set_enabled(False)

        # Keep track of window position and first frame state
        window_pos = np.array([self.window_size[0] - window_width - 20, 30])
        first_frame = True

        # Define the callback function for the GUI
        def ray_intersection_gui_callback():
            nonlocal ray_origin_pos, ray_direction, window_pos, first_frame
            nonlocal intersection_record, last_query_time, intersection_point_pos
            nonlocal is_valid_intersection, r_max, current_use_bvh

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
                                     "Ray Intersection Controls")
                ps.imgui.Separator()
                ps.imgui.Text("Control the ray origin and direction")
                ps.imgui.Spacing()

                # Query method selection section
                ps.imgui.TextColored((0.9, 0.6, 0.3, 1.0), "Query Method")
                changed_bvh, current_use_bvh = ps.imgui.Checkbox(
                    "Use BVH Acceleration", current_use_bvh)
                ps.imgui.Spacing()
                ps.imgui.Separator()

                # Origin position controls
                ps.imgui.TextColored((1.0, 0.5, 0.0, 1.0), "Ray Origin")
                changed_x, ray_origin_pos[0] = ps.imgui.SliderFloat("Origin X",
                                                                    ray_origin_pos[0],
                                                                    min_bounds[0],
                                                                    max_bounds[0])
                ps.imgui.Spacing()

                changed_y, ray_origin_pos[1] = ps.imgui.SliderFloat("Origin Y",
                                                                    ray_origin_pos[1],
                                                                    min_bounds[1],
                                                                    max_bounds[1])
                ps.imgui.Spacing()

                # For 3D compatibility, keep z coordinate but fix at 0 for 2D
                ray_origin_pos[2] = 0.0

                # Direction controls (using angle)
                ps.imgui.TextColored((1.0, 0.8, 0.0, 1.0), "Ray Direction")

                # Convert current direction to angle in degrees
                current_angle = np.degrees(np.arctan2(
                    ray_direction[1], ray_direction[0]))
                changed_angle, new_angle = ps.imgui.SliderFloat("Angle (degrees)",
                                                                current_angle,
                                                                -180.0,
                                                                180.0)

                # If angle changed, update the direction vector
                if changed_angle:
                    angle_rad = np.radians(new_angle)
                    ray_direction[0] = np.cos(angle_rad)
                    ray_direction[1] = np.sin(angle_rad)
                    # Keep z at 0 for 2D
                    ray_direction[2] = 0.0

                # Ray max distance control
                ps.imgui.Spacing()
                ps.imgui.TextColored((0.5, 0.8, 1.0, 1.0), "Ray Parameters")
                # Calculate a reasonable max value for the slider based on the scene bounds
                scene_diameter = np.linalg.norm(max_bounds - min_bounds)
                # Ensure a reasonable upper limit
                slider_max = scene_diameter

                # Determine if we should use a log slider for r_max (better for large ranges)
                if r_max >= slider_max * 0.9:  # If close to infinity
                    # Use a checkbox for "unlimited" or not
                    is_unlimited = r_max >= float('inf') * 0.9
                    changed_unlimited, is_unlimited = ps.imgui.Checkbox(
                        "Unlimited Range", is_unlimited)
                    if changed_unlimited:
                        r_max = float('inf') if is_unlimited else slider_max
                        changed_r_max = True
                    else:
                        changed_r_max = False

                    if not is_unlimited:
                        # Only show the slider if not unlimited
                        ps.imgui.Text("Maximum Ray Distance")
                        _changed, r_max = ps.imgui.SliderFloat("##r_max_slider",
                                                               float(r_max),
                                                               0.1,
                                                               slider_max,
                                                               format="%.1f")
                        changed_r_max = changed_r_max or _changed
                else:
                    # Use a standard slider
                    ps.imgui.Text("Maximum Ray Distance")
                    changed_r_max, r_max = ps.imgui.SliderFloat("##r_max_slider",
                                                                float(r_max),
                                                                0.1,
                                                                slider_max,
                                                                format="%.1f")

                    # Add a button to set to unlimited
                    if ps.imgui.Button("Set Unlimited"):
                        r_max = float('inf')
                        changed_r_max = True

                ps.imgui.Spacing()

                changed_any = changed_x or changed_y or changed_angle or changed_r_max or changed_bvh

                # Update visualizations if anything changed
                if changed_any:
                    # Update ray origin in visualization
                    ray_origin.update_point_positions(
                        np.array([ray_origin_pos]))

                    # Query the intersection with timing
                    query_origin = self._position_to_array2(ray_origin_pos)
                    query_direction = Array2(
                        float(ray_direction[0]), float(ray_direction[1]))
                    start_time = time.time()

                    # Use the appropriate intersect method based on use_bvh
                    if current_use_bvh:
                        intersection_record = self.polyline.intersect_bvh(
                            query_origin, query_direction, Float(r_max))
                    else:
                        intersection_record = self.polyline.intersect_baseline(
                            query_origin, query_direction, Float(r_max))
                    dr.eval(intersection_record)

                    last_query_time = (time.time() - start_time) * 1000

                    # Check if there's a valid intersection
                    is_valid_intersection = bool(
                        intersection_record.valid.numpy()[0])

                    # Calculate the end point of the ray line and intersection point
                    if is_valid_intersection:
                        # Use the intersection distance
                        intersection_distance = float(
                            intersection_record.d.numpy()[0])
                        # Limit to ray_length if needed
                        ray_line_end = ray_origin_pos + ray_direction * \
                            min(intersection_distance, ray_length)

                        # Extract the intersection point position
                        intersection_point_pos = np.array([
                            intersection_record.p.x.numpy(),
                            intersection_record.p.y.numpy(),
                            np.zeros_like(intersection_record.p.x.numpy())
                        ]).T

                        # Show and update intersection point
                        intersection_point.set_enabled(True)
                        intersection_point.update_point_positions(
                            intersection_point_pos)
                    else:
                        # Just use the ray_length for visualization
                        ray_line_end = ray_origin_pos + ray_direction * ray_length
                        # Hide intersection point
                        intersection_point.set_enabled(False)

                    # Update ray line
                    ray_line.update_node_positions(
                        np.vstack([ray_origin_pos, ray_line_end]))

                # Display the current ray information
                ps.imgui.Separator()
                ps.imgui.Text(
                    f"Origin: ({ray_origin_pos[0]:.3f}, {ray_origin_pos[1]:.3f})")
                ps.imgui.Text(
                    f"Direction: ({ray_direction[0]:.3f}, {ray_direction[1]:.3f})")
                ps.imgui.Text(f"Angle: {current_angle:.1f}°")
                ps.imgui.Text(
                    f"Max Distance: {r_max if r_max != float('inf') else 'Unlimited'}")
                ps.imgui.Text(
                    f"Method: {'BVH' if current_use_bvh else 'Baseline'}")
                ps.imgui.Spacing()

                # Reset button
                if ps.imgui.Button("Reset Ray", (150, 30)):
                    ray_origin_pos = center.copy()
                    ray_direction = np.array([1.0, 0.0, 0.0])
                    r_max = initial_r_max
                    current_use_bvh = use_bvh

                    # Update ray origin in visualization
                    ray_origin.update_point_positions(
                        np.array([ray_origin_pos]))

                    # Query the intersection with timing
                    query_origin = self._position_to_array2(ray_origin_pos)
                    query_direction = Array2(
                        float(ray_direction[0]), float(ray_direction[1]))

                    start_time = time.time()

                    # Use the appropriate intersect method based on use_bvh
                    if current_use_bvh:
                        intersection_record = self.polyline.intersect_bvh(
                            query_origin, query_direction, Float(r_max))
                    else:
                        intersection_record = self.polyline.intersect_baseline(
                            query_origin, query_direction, Float(r_max))
                    dr.eval(intersection_record)

                    last_query_time = (time.time() - start_time) * 1000

                    # Check if there's a valid intersection
                    is_valid_intersection = bool(
                        intersection_record.valid.numpy()[0])

                    # Calculate the end point of the ray line and intersection point
                    if is_valid_intersection:
                        # Use the intersection distance
                        intersection_distance = float(
                            intersection_record.d.numpy()[0])
                        # Limit to ray_length if needed
                        ray_line_end = ray_origin_pos + ray_direction * \
                            min(intersection_distance, ray_length)

                        # Extract the intersection point position
                        intersection_point_pos = np.array([
                            intersection_record.p.x.numpy(),
                            intersection_record.p.y.numpy(),
                            np.zeros_like(intersection_record.p.x.numpy())
                        ]).T

                        # Show and update intersection point
                        intersection_point.set_enabled(True)
                        intersection_point.update_point_positions(
                            intersection_point_pos)
                    else:
                        # Just use the ray_length for visualization
                        ray_line_end = ray_origin_pos + ray_direction * ray_length
                        # Hide intersection point
                        intersection_point.set_enabled(False)

                    # Update ray line
                    ray_line.update_node_positions(
                        np.vstack([ray_origin_pos, ray_line_end]))

                # Show intersection information if enabled and there's a valid intersection
                if show_intersection_info:
                    ps.imgui.Separator()
                    ps.imgui.TextColored(
                        (0.0, 1.0, 1.0, 1.0), "Intersection Information")
                    ps.imgui.Spacing()

                    if is_valid_intersection:
                        # Get the intersection point for display
                        ip_x = intersection_point_pos[0][0]
                        ip_y = intersection_point_pos[0][1]

                        # Distance from origin to intersection
                        distance = intersection_record.d.numpy()[0]

                        # Display all information about the intersection
                        ps.imgui.Text(f"Position: ({ip_x:.3f}, {ip_y:.3f})")
                        ps.imgui.Text(f"Distance: {distance:.5f}")
                        ps.imgui.Text(
                            f"Primitive ID: {intersection_record.prim_id.numpy()[0]}")
                        ps.imgui.Text(
                            f"Normal: ({intersection_record.n.x.numpy()[0]:.3f}, {intersection_record.n.y.numpy()[0]:.3f})")
                        ps.imgui.Text(f"Query Time: {last_query_time:.3f} ms")
                        ps.imgui.Text(
                            f"On Boundary: {intersection_record.on_boundary.numpy()[0]}")
                    else:
                        ps.imgui.TextColored(
                            (1.0, 0.3, 0.3, 1.0), "Status: No intersection found")
                        ps.imgui.Text(f"Query Time: {last_query_time:.3f} ms")

            ps.imgui.End()

        # Register the GUI callback
        ps.set_user_callback(ray_intersection_gui_callback)

        # Return the created structures and data
        return ray_origin, ray_origin_pos, ray_direction, intersection_point, intersection_record


def make_cube():
    vertices = np.array([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
    ])
    indices = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ])
    return vertices, indices


def run_ray_intersection_demo():
    """
    Run a demonstration of the ray intersection visualization feature.
    """
    # Create a polyline from an example shape file
    vertices, indices = load_obj_2d(BASE_DIR / "data/bunny2d.obj")
    # vertices, indices = make_cube()
    polyline = Polyline(Array2(vertices.T), Array2i(indices.T))

    # Create a polyline viewer
    viewer = RayIntersectionVisualizer(
        polyline=polyline
    )

    # Add the ray intersection visualization
    viewer.add_ray_intersection_visualization(
        # Orange ray origin
        ray_origin_color=(1.0, 0.5, 0.0),
        ray_origin_radius=0.015,
        # Cyan intersection point
        intersection_point_color=(0.0, 1.0, 1.0),
        intersection_point_radius=0.015,
        # Gold ray line
        ray_line_color=(1.0, 0.8, 0.0),
        ray_line_width=0.002,
        ray_length=20.0,
        # Set initial maximum ray distance (can be adjusted in the UI)
        initial_r_max=5.0,  # Start with a limited search distance
        # Enable BVH acceleration for faster ray intersection
        use_bvh=False,
        # Show information about the intersection
        show_intersection_info=True
    )

    # Show the visualization
    viewer.show()


if __name__ == "__main__":
    run_ray_intersection_demo()
