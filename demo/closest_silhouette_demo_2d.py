#!/usr/bin/env python
"""
Demo script showing how to use the closest silhouette point visualization in PolylineViewer.

This example creates a simple polyline shape and displays an interactive
visualization that allows the user to control a query point and see the closest
silhouette point (a vertex where two edges meet) on the polyline in real-time.
"""

import numpy as np
from examples.polyline_viewer import PolylineViewer
from gquery.shapes.polyline import Polyline
from gquery.core.fwd import *
from gquery.util.obj_loader import load_obj_2d
from typing import Optional, Tuple, Any
import time
import polyscope as ps


class SilhouettePointVisualizer(PolylineViewer):
    def add_closest_silhouette_visualization(
        self,
        initial_position: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        # Point visualization options
        query_point_name: str = "query_point_silhouette",
        query_point_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        query_point_radius: float = 0.01,
        silhouette_point_name: str = "silhouette_point",
        silhouette_point_color: Tuple[float, float, float] = (0.5, 0.0, 1.0),
        silhouette_point_radius: float = 0.01,
        # Line options
        line_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
        line_width: float = 0.001,
        connection_line_name: str = "silhouette_connection_line",
        # UI options
        window_title: str = "Silhouette Query Controls",
        window_width: int = 380,
        window_height: int = 420,
        show_silhouette_info: bool = True,
        use_snch: bool = False,
        max_distance: float = float('inf'),
    ) -> Tuple[Any, np.ndarray, Any, Any]:
        """
        Add an interactive query point controller with closest silhouette point visualization.

        This method adds an interactive query point and displays the closest silhouette point
        on the polyline in real-time as the query point is moved. It shows a line connecting 
        the two points.

        Args:
            initial_position: Initial position of the query point
            bounds: Optional (min_bounds, max_bounds) tuple for control limits

            query_point_name: Name for the query point structure
            query_point_color: RGB color tuple for the query point
            query_point_radius: Radius of the query point
            silhouette_point_name: Name for the silhouette point structure
            silhouette_point_color: RGB color tuple for the silhouette point
            silhouette_point_radius: Radius of the silhouette point

            line_color: RGB color tuple for the connecting line
            line_width: Width of the connecting line
            connection_line_name: Name for the connecting line structure

            window_title: Title for the control window
            window_width: Width of the control window
            window_height: Height of the control window
            show_silhouette_info: Whether to show information about the silhouette point
            use_snch: Initial state for using SNCH acceleration
            max_distance: Maximum search distance for silhouette points

        Returns:
            Tuple containing:
            - The query point structure
            - The query point position array
            - The silhouette point structure
            - The silhouette record from the last update
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
        current_use_snch = use_snch
        current_max_distance = max_distance
        last_query_time = 0.0

        # Query the closest silhouette point
        query_array = self._position_to_array2(query_point_pos)

        start_time = time.time()
        if current_use_snch:
            silhouette_record = self.polyline.closest_silhouette_snch(
                query_array, Float(current_max_distance))
        else:
            silhouette_record = self.polyline.closest_silhouette_baseline(
                query_array, Float(current_max_distance))
        dr.eval(silhouette_record)
        last_query_time = (time.time() - start_time) * \
            1000  # Convert to milliseconds

        # Extract the silhouette point position as numpy array
        silhouette_point_pos = np.array([
            silhouette_record.p.x.numpy(),
            silhouette_record.p.y.numpy(),
            # Add z-coordinate for 3D viz
            np.zeros_like(silhouette_record.p.x.numpy())
        ]).T

        # Create the silhouette point visualization
        silhouette_point = self.add_point_cloud(
            points=silhouette_point_pos,
            name=silhouette_point_name,
            color=silhouette_point_color,
            point_radius=silhouette_point_radius
        )

        # Create a line connecting the query point and silhouette point
        connection_line = self.add_polyline(
            vertices=np.vstack([query_point_pos, silhouette_point_pos]),
            edges=np.array([[0, 1]]),
            name=connection_line_name,
            color=line_color,
            width=line_width
        )

        # Keep track of window position and first frame state
        window_pos = np.array([self.window_size[0] - window_width - 20, 30])
        first_frame = True

        # Define the callback function for the GUI
        def silhouette_query_gui_callback():
            nonlocal query_point_pos, silhouette_point_pos, window_pos, first_frame
            nonlocal silhouette_record, current_use_snch, last_query_time, current_max_distance

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
                                     "Silhouette Query Controller")
                ps.imgui.Separator()
                ps.imgui.Text("Control the position of the query point")
                ps.imgui.Spacing()

                # Query method selection section
                ps.imgui.TextColored((0.9, 0.6, 0.3, 1.0), "Query Method")
                changed_snch, current_use_snch = ps.imgui.Checkbox(
                    "Use SNCH Acceleration", current_use_snch)
                ps.imgui.Spacing()

                # Maximum distance control
                scene_diameter = np.linalg.norm(max_bounds - min_bounds)
                slider_max = scene_diameter

                # Handle unlimited distance option
                is_unlimited = current_max_distance >= float('inf') * 0.9
                changed_unlimited, is_unlimited = ps.imgui.Checkbox(
                    "Unlimited Search Distance", is_unlimited)

                if changed_unlimited:
                    current_max_distance = float(
                        'inf') if is_unlimited else slider_max
                    changed_max_distance = True
                else:
                    changed_max_distance = False

                if not is_unlimited:
                    ps.imgui.Text("Maximum Search Distance")
                    _changed, current_max_distance = ps.imgui.SliderFloat(
                        "##max_distance_slider",
                        float(current_max_distance),
                        0.1,
                        slider_max,
                        format="%.1f"
                    )
                    changed_max_distance = changed_max_distance or _changed

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
                changed_any = changed_x or changed_y or changed_snch or changed_max_distance

                # Update visualizations if anything changed
                if changed_any:
                    # Always update the query point position first
                    query_point.update_point_positions(
                        np.array([query_point_pos]))

                    # Query the closest silhouette point with timing
                    query_array = self._position_to_array2(query_point_pos)

                    start_time = time.time()
                    if current_use_snch:
                        silhouette_record = self.polyline.closest_silhouette_snch(
                            query_array, Float(current_max_distance))
                    else:
                        silhouette_record = self.polyline.closest_silhouette_baseline(
                            query_array, Float(current_max_distance))
                    dr.eval(silhouette_record)
                    last_query_time = (time.time() - start_time) * 1000

                    # Check if the record is valid
                    is_valid = bool(silhouette_record.valid.numpy()[0])

                    if is_valid:
                        # Extract the silhouette point position
                        silhouette_point_pos = np.array([
                            silhouette_record.p.x.numpy(),
                            silhouette_record.p.y.numpy(),
                            # Add z-coordinate for 3D viz
                            np.zeros_like(silhouette_record.p.x.numpy())
                        ]).T

                        # Update silhouette point and connection line
                        silhouette_point.update_point_positions(
                            silhouette_point_pos)
                        silhouette_point.set_enabled(True)

                        connection_line.update_node_positions(
                            np.vstack([query_point_pos, silhouette_point_pos])
                        )
                        connection_line.set_enabled(True)
                    else:
                        # If no valid silhouette point found, hide relevant visualizations
                        silhouette_point.set_enabled(False)
                        connection_line.set_enabled(False)

                # Display the current coordinates
                ps.imgui.Separator()
                ps.imgui.Text(
                    f"Query point: ({query_point_pos[0]:.3f}, {query_point_pos[1]:.3f})")
                ps.imgui.Spacing()

                # Reset button
                if ps.imgui.Button("Reset to Center", (150, 30)):
                    query_point_pos = center.copy()

                    # Always update the query point position first
                    query_point.update_point_positions(
                        np.array([query_point_pos]))

                    # Query the closest silhouette point with timing
                    query_array = self._position_to_array2(query_point_pos)

                    start_time = time.time()
                    if current_use_snch:
                        silhouette_record = self.polyline.closest_silhouette_snch(
                            query_array, Float(current_max_distance))
                    else:
                        silhouette_record = self.polyline.closest_silhouette_baseline(
                            query_array, Float(current_max_distance))
                    dr.eval(silhouette_record)
                    last_query_time = (time.time() - start_time) * 1000

                    # Check if the record is valid
                    is_valid = bool(silhouette_record.valid.numpy()[0])

                    if is_valid:
                        # Extract the silhouette point position
                        silhouette_point_pos = np.array([
                            silhouette_record.p.x.numpy(),
                            silhouette_record.p.y.numpy(),
                            # Add z-coordinate for 3D viz
                            np.zeros_like(silhouette_record.p.x.numpy())
                        ]).T

                        # Update silhouette point and connection line
                        silhouette_point.update_point_positions(
                            silhouette_point_pos)
                        silhouette_point.set_enabled(True)

                        connection_line.update_node_positions(
                            np.vstack([query_point_pos, silhouette_point_pos])
                        )
                        connection_line.set_enabled(True)
                    else:
                        # If no valid silhouette point found, hide relevant visualizations
                        silhouette_point.set_enabled(False)
                        connection_line.set_enabled(False)

                # Show silhouette point information if enabled
                if show_silhouette_info:
                    ps.imgui.Separator()
                    ps.imgui.TextColored(
                        (0.5, 0.0, 1.0, 1.0), "Silhouette Point Information")
                    ps.imgui.Spacing()

                    # Check if the record is valid
                    is_valid = bool(silhouette_record.valid.numpy()[0])

                    if is_valid:
                        # Get the silhouette point position for display
                        sp_x = silhouette_point_pos[0][0]
                        sp_y = silhouette_point_pos[0][1]

                        # Display all information about the silhouette point
                        ps.imgui.Text(f"Position: ({sp_x:.3f}, {sp_y:.3f})")
                        ps.imgui.Text(
                            f"Distance: {silhouette_record.d.numpy()[0]:.5f}")
                        ps.imgui.Text(
                            f"Primitive ID: {silhouette_record.prim_id.numpy()[0]}")

                        # Display normal information if available
                        if hasattr(silhouette_record, 'n'):
                            n_x = silhouette_record.n.x.numpy()[0]
                            n_y = silhouette_record.n.y.numpy()[0]
                            ps.imgui.Text(f"Normal: ({n_x:.3f}, {n_y:.3f})")

                        # Display tangent information if available
                        if hasattr(silhouette_record, 't1') and hasattr(silhouette_record, 't2'):
                            t1_x = silhouette_record.t1.x.numpy()[0]
                            t1_y = silhouette_record.t1.y.numpy()[0]
                            t2_x = silhouette_record.t2.x.numpy()[0]
                            t2_y = silhouette_record.t2.y.numpy()[0]
                            ps.imgui.Text(
                                f"Tangent 1: ({t1_x:.3f}, {t1_y:.3f})")
                            ps.imgui.Text(
                                f"Tangent 2: ({t2_x:.3f}, {t2_y:.3f})")

                        ps.imgui.Text(f"Query Time: {last_query_time:.3f} ms")
                        ps.imgui.Text(
                            f"Method: {'SNCH' if current_use_snch else 'Baseline'}")
                    else:
                        ps.imgui.TextColored(
                            (1.0, 0.3, 0.3, 1.0), "No valid silhouette point found")
                        ps.imgui.Text(f"Query Time: {last_query_time:.3f} ms")
                        if current_max_distance < float('inf'):
                            ps.imgui.Text(
                                f"Try increasing the maximum search distance")

            ps.imgui.End()

        # Register the GUI callback
        ps.set_user_callback(silhouette_query_gui_callback)

        # Return the created structures and data
        return query_point, query_point_pos, silhouette_point, silhouette_record


def create_star_shape(n_points=10, inner_radius=0.5, outer_radius=1.0):
    """Create a star-shaped polyline for demonstration."""
    angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False)
    radii = np.array([outer_radius, inner_radius] * n_points)

    vertices = np.zeros((2 * n_points, 2))
    vertices[:, 0] = radii * np.cos(angles)
    vertices[:, 1] = radii * np.sin(angles)

    # Connect points to form a closed star
    indices = np.column_stack(
        (np.arange(2 * n_points), np.roll(np.arange(2 * n_points), -1)))

    return vertices, indices


def run_silhouette_point_demo(shape_type="star"):
    """
    Run a demonstration of the closest silhouette point visualization feature.

    Args:
        shape_type: Type of shape to demonstrate ("star", "file", or "obj")
    """
    if shape_type == "star":
        # Create a star-shaped polyline
        # vertices, indices = create_star_shape(
        #     n_points=8, inner_radius=0.3, outer_radius=1.0)
        vertices, indices = load_obj_2d(BASE_DIR / "data/bunny2d.obj")
        polyline = Polyline(Array2(vertices.T), Array2i(indices.T))
    elif shape_type == "file":
        # Create a polyline from a text file
        try:
            # Try to load from file
            vertices = np.loadtxt(BASE_DIR / "data/custom_shape.txt")
            n_vertices = vertices.shape[0]
            indices = np.column_stack(
                (np.arange(n_vertices), np.roll(np.arange(n_vertices), -1)))
            polyline = Polyline(Array2(vertices.T), Array2i(indices.T))
        except FileNotFoundError:
            # Fallback to star shape
            print("Custom shape file not found, using star shape instead.")
            vertices, indices = create_star_shape()
            polyline = Polyline(Array2(vertices.T), Array2i(indices.T))
    else:  # "obj"
        # Create a polyline from an OBJ file
        vertices, indices = load_obj_2d(BASE_DIR / "data/bunny2d.obj")
        polyline = Polyline(Array2(vertices.T), Array2i(indices.T))

    # Create a polyline viewer
    viewer = SilhouettePointVisualizer(
        polyline=polyline
    )

    # Add the closest silhouette point visualization
    viewer.add_closest_silhouette_visualization(
        # Red query point
        query_point_color=(1.0, 0.0, 0.0),
        query_point_radius=0.02,
        # Purple silhouette point
        silhouette_point_color=(0.5, 0.0, 1.0),
        silhouette_point_radius=0.02,
        # Show information panel
        show_silhouette_info=True,
        # Use SNCH acceleration if available
        use_snch=True
    )

    # Show the visualization
    viewer.show()


if __name__ == "__main__":
    import sys
    shape = "star"
    if len(sys.argv) > 1:
        shape = sys.argv[1]
    run_silhouette_point_demo(shape)
