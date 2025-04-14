#!/usr/bin/env python
"""
Demo script showing how to use the closest silhouette point visualization for meshes.

This example creates a 3D mesh and displays an interactive visualization
that allows the user to move a query point and see the closest silhouette point
on the mesh in real-time.
"""

import time
import numpy as np
from demo.mesh_viewer import MeshViewer
from gquery.shapes.mesh import Mesh
from gquery.core.fwd import *
from gquery.util.obj_loader import load_obj_3d
import polyscope as ps
from typing import Optional, Tuple, Any


class SilhouetteMeshVisualizer(MeshViewer):
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
        window_height: int = 520,
        show_silhouette_info: bool = True,
        use_snch: bool = True,
        max_distance: float = float('inf'),
    ) -> Tuple[Any, np.ndarray, Any, Any]:
        """
        Add an interactive query point controller with closest silhouette point visualization.

        This method adds an interactive query point and displays the closest silhouette point
        on the mesh in real-time as the query point is moved. It shows a line connecting 
        the two points.

        Args:
            initial_position: Initial position of the query point (3D)
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
        # Make sure polyscope is initialized and we have a mesh
        if not self.is_initialized:
            self.init()

        if self.mesh is None:
            raise ValueError(
                "SilhouetteMeshVisualizer instance needs a mesh attribute.")

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
        query_array = Array3(float(query_point_pos[0]), float(
            query_point_pos[1]), float(query_point_pos[2]))

        start_time = time.time()
        if current_use_snch:
            silhouette_record = self.mesh.closest_silhouette_snch(
                query_array, Float(current_max_distance))
        else:
            silhouette_record = self.mesh.closest_silhouette_baseline(
                query_array, Float(current_max_distance))
        dr.eval(silhouette_record)
        last_query_time = (time.time() - start_time) * \
            1000  # Convert to milliseconds

        # Check if the record is valid
        is_valid = bool(silhouette_record.valid.numpy()[0])

        # Extract the silhouette point position as numpy array
        if is_valid:
            silhouette_point_pos = np.array([
                silhouette_record.p.x.numpy(),
                silhouette_record.p.y.numpy(),
                silhouette_record.p.z.numpy()
            ]).T
        else:
            # Create a dummy position if no valid silhouette point
            silhouette_point_pos = np.array([query_point_pos])

        # Create the silhouette point visualization
        silhouette_point = self.add_point_cloud(
            points=silhouette_point_pos,
            name=silhouette_point_name,
            color=silhouette_point_color,
            point_radius=silhouette_point_radius,
            enabled=is_valid
        )

        # Create a line connecting the query point and silhouette point
        connection_line = self.add_polyline(
            points=np.vstack([query_point_pos, silhouette_point_pos]),
            name=connection_line_name,
            color=line_color,
            width=line_width,
            enabled=is_valid
        )

        # Hide structures if there's no valid silhouette point
        if not is_valid:
            silhouette_point.set_enabled(False)
            connection_line.set_enabled(False)

        # Keep track of window position and first frame state
        window_pos = np.array([self.window_size[0] - window_width - 20, 30])
        first_frame = True

        # Define the callback function for the GUI
        def silhouette_query_gui_callback():
            nonlocal query_point_pos, silhouette_point_pos, window_pos, first_frame
            nonlocal silhouette_record, current_use_snch, last_query_time, current_max_distance
            nonlocal is_valid

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
                slider_max = scene_diameter * 2

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
                ps.imgui.TextColored((1.0, 0.0, 0.0, 1.0),
                                     "Query Point Position")
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

                changed_z, query_point_pos[2] = ps.imgui.SliderFloat("Z Position",
                                                                     query_point_pos[2],
                                                                     min_bounds[2],
                                                                     max_bounds[2])
                ps.imgui.Spacing()

                changed_any = changed_x or changed_y or changed_z or changed_snch or changed_max_distance

                # Update visualizations if anything changed
                if changed_any:
                    # Always update the query point position first
                    query_point.update_point_positions(
                        np.array([query_point_pos]))

                    # Query the closest silhouette point with timing
                    query_array = Array3(float(query_point_pos[0]), float(
                        query_point_pos[1]), float(query_point_pos[2]))

                    start_time = time.time()
                    if current_use_snch:
                        silhouette_record = self.mesh.closest_silhouette_snch(
                            query_array, Float(current_max_distance))
                    else:
                        silhouette_record = self.mesh.closest_silhouette_baseline(
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
                            silhouette_record.p.z.numpy()
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
                    f"Query point: ({query_point_pos[0]:.3f}, {query_point_pos[1]:.3f}, {query_point_pos[2]:.3f})")
                ps.imgui.Spacing()

                # Reset button
                if ps.imgui.Button("Reset to Center", (150, 30)):
                    query_point_pos = center.copy()

                    # Always update the query point position first
                    query_point.update_point_positions(
                        np.array([query_point_pos]))

                    # Query the closest silhouette point with timing
                    query_array = Array3(float(query_point_pos[0]), float(
                        query_point_pos[1]), float(query_point_pos[2]))

                    start_time = time.time()
                    if current_use_snch:
                        silhouette_record = self.mesh.closest_silhouette_snch(
                            query_array, Float(current_max_distance))
                    else:
                        silhouette_record = self.mesh.closest_silhouette_baseline(
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
                            silhouette_record.p.z.numpy()
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

                    if is_valid:
                        # Get the silhouette point position for display
                        sp_x = silhouette_point_pos[0][0]
                        sp_y = silhouette_point_pos[0][1]
                        sp_z = silhouette_point_pos[0][2]

                        # Display position and distance information
                        ps.imgui.Text(
                            f"Position: ({sp_x:.3f}, {sp_y:.3f}, {sp_z:.3f})")
                        ps.imgui.Text(
                            f"Distance: {silhouette_record.d.numpy()[0]:.5f}")
                        ps.imgui.Text(
                            f"Primitive ID: {silhouette_record.prim_id.numpy()[0]}")

                        # Display performance information
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


def run_silhouette_mesh_demo():
    """
    Run a demonstration of the closest silhouette point visualization feature for meshes.
    """
    # Load a 3D mesh (bunny model)
    vertices, faces = load_obj_3d(BASE_DIR / "data/bunny_hi.obj")

    # Create the mesh object
    mesh = Mesh(Array3(vertices.T), Array3i(faces.T))
    # No need to call configure() as it's done in the constructor

    # Create the mesh viewer
    viewer = SilhouetteMeshVisualizer(
        mesh=mesh,
        window_size=(1280, 960),
        mesh_color=(0.8, 0.8, 0.8, 0.5),  # Set translucent color for the mesh
        mesh_material="wax",  # The "wax" material supports transparency
        show_wireframe=True
    )

    # Add the closest silhouette point visualization
    viewer.add_closest_silhouette_visualization(
        # Red query point
        query_point_color=(1.0, 0.0, 0.0),
        query_point_radius=0.015,
        # Purple silhouette point
        silhouette_point_color=(0.5, 0.0, 1.0),
        silhouette_point_radius=0.015,
        # White connection line
        line_color=(0.9, 0.9, 0.9),
        line_width=0.002,
        # Set initial maximum search distance (can be adjusted in the UI)
        max_distance=float('inf'),  # Start with unlimited distance
        # Enable SNCH acceleration for faster silhouette point finding
        use_snch=True,
        # Show information about the silhouette point
        show_silhouette_info=True
    )

    # Show the visualization
    viewer.show()


if __name__ == "__main__":
    run_silhouette_mesh_demo()
