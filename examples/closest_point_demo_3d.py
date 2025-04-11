#!/usr/bin/env python
"""
Demo script showing how to use the closest point visualization for meshes.

This example creates a 3D mesh and displays an interactive visualization
that allows the user to move a query point and see the closest point
on the mesh in real-time.
"""

import time
from typing import Any, Optional, Tuple
import numpy as np
from examples.mesh_viewer import MeshViewer
from gquery.shapes.mesh import Mesh
from gquery.core.fwd import *
from gquery.util.obj_loader import load_obj_3d
import polyscope as ps


class ClosestPointMeshVisualizer(MeshViewer):
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
        # Line and normal options
        line_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
        line_width: float = 0.001,
        connection_line_name: str = "connection_line",
        normal_line_name: str = "normal_line",
        normal_line_color: Tuple[float, float, float] = (0.0, 0.8, 0.8),
        normal_line_width: float = 0.001,
        normal_line_length: float = 0.1,
        # Distance sphere options
        distance_sphere_name: str = "distance_sphere",
        distance_sphere_color: Tuple[float, float,
                                     float, float] = (0.8, 0.4, 0.4, 0.3),
        distance_sphere_material: str = "wax",
        show_distance_sphere: bool = True,
        sphere_resolution: int = 20,
        # UI options
        window_title: str = "Query Point Controls",
        window_width: int = 380,
        window_height: int = 420,
        show_closest_info: bool = True,
        use_bvh: bool = False,
    ) -> Tuple[Any, np.ndarray, Any, Any]:
        """
        Add an interactive query point controller with closest point visualization for meshes.

        This method adds an interactive query point and displays the closest point on the mesh
        in real-time as the query point is moved. It shows a line connecting the two points
        and optionally a sphere centered at the query point with radius equal to the distance.

        Args:
            initial_position: Initial position of the query point (3D)
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
            normal_line_name: Name for the normal line at closest point
            normal_line_color: RGB color tuple for the normal line
            normal_line_width: Width of the normal line
            normal_line_length: Length of the normal line

            distance_sphere_name: Name for the distance sphere structure
            distance_sphere_color: RGB color tuple for the distance sphere
            distance_sphere_material: Material for the distance sphere
            show_distance_sphere: Whether to show the distance sphere visualization
            sphere_resolution: Resolution of the sphere mesh

            window_title: Title for the control window
            window_width: Width of the control window
            window_height: Height of the control window
            show_closest_info: Whether to show information about the closest point
            use_bvh: Initial state for using BVH acceleration

        Returns:
            Tuple containing:
            - The query point structure
            - The query point position array
            - The closest point structure
            - The closest point record from the last update
        """
        # Make sure polyscope is initialized
        if not self.is_initialized:
            self.init()

        if self.mesh is None:
            raise ValueError(
                "ClosestPointMeshVisualizer instance needs a mesh attribute.")

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
        query_array = self._position_to_array3(query_point_pos)

        start_time = time.time()
        if current_use_bvh:
            closest_record = self.mesh.closest_point_bvh(query_array)
        else:
            closest_record = self.mesh.closest_point_baseline(query_array)
        last_query_time = (time.time() - start_time) * \
            1000  # Convert to milliseconds

        # Extract the closest point position as numpy array
        closest_point_pos = np.array([
            closest_record.p.x.numpy(),
            closest_record.p.y.numpy(),
            closest_record.p.z.numpy()
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
            points=np.vstack([query_point_pos, closest_point_pos]),
            name=connection_line_name,
            color=line_color,
            width=line_width
        )

        # Create a line for the normal at the closest point
        normal_vec = np.array([
            closest_record.n.x.numpy(),
            closest_record.n.y.numpy(),
            closest_record.n.z.numpy()
        ]).T * normal_line_length

        normal_line = self.add_polyline(
            points=np.vstack(
                [closest_point_pos, closest_point_pos + normal_vec]),
            name=normal_line_name,
            color=normal_line_color,
            width=normal_line_width
        )

        # Create a sphere visualization for the distance
        distance_sphere = None
        if show_distance_sphere:
            distance = closest_record.d.numpy()[0]
            # Generate a sphere mesh
            u = np.linspace(0, 2 * np.pi, sphere_resolution)
            v = np.linspace(0, np.pi, sphere_resolution)
            x = distance * np.outer(np.cos(u), np.sin(v)) + query_point_pos[0]
            y = distance * np.outer(np.sin(u), np.sin(v)) + query_point_pos[1]
            z = distance * np.outer(np.ones(np.size(u)),
                                    np.cos(v)) + query_point_pos[2]

            # Extract vertices and faces for the sphere
            vertices = np.zeros((sphere_resolution * sphere_resolution, 3))
            for i in range(sphere_resolution):
                for j in range(sphere_resolution):
                    vertices[i * sphere_resolution +
                             j] = [x[i, j], y[i, j], z[i, j]]

            # Create faces for the sphere
            faces = []
            for i in range(sphere_resolution - 1):
                for j in range(sphere_resolution - 1):
                    p1 = i * sphere_resolution + j
                    p2 = i * sphere_resolution + (j + 1)
                    p3 = (i + 1) * sphere_resolution + j
                    p4 = (i + 1) * sphere_resolution + (j + 1)
                    faces.append([p1, p2, p4])
                    faces.append([p1, p4, p3])

            # Register the sphere mesh
            distance_sphere = self.add_mesh(
                vertices=vertices,
                faces=np.array(faces),
                name=distance_sphere_name,
                color=distance_sphere_color,
                material=distance_sphere_material,
                smooth_shade=False,
                enabled=True
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

                changed_z, query_point_pos[2] = ps.imgui.SliderFloat("Z Position",
                                                                     query_point_pos[2],
                                                                     min_bounds[2],
                                                                     max_bounds[2])
                ps.imgui.Spacing()

                changed_any = changed_x or changed_y or changed_z or changed_bvh

                # Update visualizations if anything changed
                if changed_any:
                    # Query the closest point with timing
                    query_array = self._position_to_array3(query_point_pos)

                    start_time = time.time()
                    if current_use_bvh:
                        closest_record = self.mesh.closest_point_bvh(
                            query_array)
                    else:
                        closest_record = self.mesh.closest_point_baseline(
                            query_array)
                    dr.eval(closest_record)
                    last_query_time = (time.time() - start_time) * 1000

                    # Update all visualizations
                    self._update_closest_point_visualizations(
                        query_point_pos,
                        closest_record,
                        query_point,
                        closest_point,
                        connection_line,
                        normal_line,
                        distance_sphere,
                        normal_line_length,
                        sphere_resolution,
                        show_distance_sphere
                    )

                # Display the current coordinates
                ps.imgui.Separator()
                ps.imgui.Text(
                    f"Query point: ({query_point_pos[0]:.3f}, {query_point_pos[1]:.3f}, {query_point_pos[2]:.3f})")
                ps.imgui.Spacing()

                # Add a checkbox to toggle the distance sphere if it exists
                if distance_sphere is not None:
                    is_visible = distance_sphere.is_enabled()
                    changed, new_visible = ps.imgui.Checkbox(
                        "Show Distance Sphere", is_visible)
                    if changed:
                        distance_sphere.set_enabled(new_visible)
                    ps.imgui.Spacing()

                # Reset button
                if ps.imgui.Button("Reset to Center", (150, 30)):
                    query_point_pos = center.copy()

                    # Update the query point position
                    query_point.update_point_positions(
                        np.array([query_point_pos]))

                    # Query the closest point with timing
                    query_array = self._position_to_array3(query_point_pos)

                    start_time = time.time()
                    if current_use_bvh:
                        closest_record = self.mesh.closest_point_bvh(
                            query_array)
                    else:
                        closest_record = self.mesh.closest_point_baseline(
                            query_array)
                    last_query_time = (time.time() - start_time) * 1000

                    # Update all visualizations
                    self._update_closest_point_visualizations(
                        query_point_pos,
                        closest_record,
                        query_point,
                        closest_point,
                        connection_line,
                        normal_line,
                        distance_sphere,
                        normal_line_length,
                        sphere_resolution,
                        show_distance_sphere
                    )

                # Show closest point information if enabled
                if show_closest_info:
                    ps.imgui.Separator()
                    ps.imgui.TextColored(
                        (0.0, 0.9, 0.3, 1.0), "Closest Point Information")
                    ps.imgui.Spacing()

                    # Get the closest point position and normal for display
                    cp_x = closest_record.p.x.numpy()[0]
                    cp_y = closest_record.p.y.numpy()[0]
                    cp_z = closest_record.p.z.numpy()[0]

                    # Get normal components
                    n_x = closest_record.n.x.numpy()[0]
                    n_y = closest_record.n.y.numpy()[0]
                    n_z = closest_record.n.z.numpy()[0]

                    # Display all information about the closest point
                    ps.imgui.Text(
                        f"Position: ({cp_x:.3f}, {cp_y:.3f}, {cp_z:.3f})")
                    ps.imgui.Text(f"Normal: ({n_x:.3f}, {n_y:.3f}, {n_z:.3f})")
                    ps.imgui.Text(
                        f"Distance: {closest_record.d.numpy()[0]:.5f}")
                    ps.imgui.Text(
                        f"Triangle ID: {closest_record.prim_id.numpy()[0]}")
                    ps.imgui.Text(
                        f"UV: ({closest_record.uv.x.numpy()[0]:.3f}, {closest_record.uv.y.numpy()[0]:.3f})")
                    ps.imgui.Text(f"Query Time: {last_query_time:.3f} ms")
                    ps.imgui.Text(
                        f"Method: {'BVH' if current_use_bvh else 'Baseline'}")

                # Show tips for the user
                ps.imgui.Separator()
                ps.imgui.TextColored((0.8, 0.8, 0.4, 1.0),
                                     "Tip: You can drag this window")
                ps.imgui.TextColored((0.8, 0.8, 0.4, 1.0),
                                     "if it's occluding other UI elements")

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
        normal_line,
        distance_sphere=None,
        normal_line_length=0.1,
        sphere_resolution=20,
        show_distance_sphere=True
    ):
        """Update all visualization elements for a closest point query on a mesh."""
        # Extract the closest point position as numpy array
        closest_point_pos = np.array([
            closest_record.p.x.numpy(),
            closest_record.p.y.numpy(),
            closest_record.p.z.numpy()
        ]).T

        # Extract the normal at the closest point
        normal_vec = np.array([
            closest_record.n.x.numpy(),
            closest_record.n.y.numpy(),
            closest_record.n.z.numpy()
        ]).T * normal_line_length

        # Update the query and closest points
        query_point.update_point_positions(np.array([query_point_pos]))
        closest_point.update_point_positions(closest_point_pos)

        # Update the connection line
        connection_line.update_node_positions(
            np.vstack([query_point_pos, closest_point_pos])
        )

        # Update the normal line
        normal_line.update_node_positions(
            np.vstack([closest_point_pos, closest_point_pos + normal_vec])
        )

        # Update the distance sphere if enabled
        if show_distance_sphere and distance_sphere is not None:
            distance = closest_record.d.numpy()[0]

            # Generate new sphere vertices
            u = np.linspace(0, 2 * np.pi, sphere_resolution)
            v = np.linspace(0, np.pi, sphere_resolution)
            x = distance * np.outer(np.cos(u), np.sin(v)) + query_point_pos[0]
            y = distance * np.outer(np.sin(u), np.sin(v)) + query_point_pos[1]
            z = distance * np.outer(np.ones(np.size(u)),
                                    np.cos(v)) + query_point_pos[2]

            # Reshape vertices
            vertices = np.zeros((sphere_resolution * sphere_resolution, 3))
            for i in range(sphere_resolution):
                for j in range(sphere_resolution):
                    vertices[i * sphere_resolution +
                             j] = [x[i, j], y[i, j], z[i, j]]

            # Update sphere vertices
            distance_sphere.update_vertex_positions(vertices)

        return closest_point_pos

    def _position_to_array3(self, position):
        """Convert a numpy position to a gquery Array3 for queries."""
        return Array3(float(position[0]), float(position[1]), float(position[2]))


def run_closest_point_mesh_demo():
    """
    Run a demonstration of the closest point visualization feature for meshes.
    """
    # Load a 3D mesh (adjust path as needed)
    vertices, faces = load_obj_3d(BASE_DIR / "data/bunny.obj")

    # Create the mesh object
    mesh = Mesh(Array3(vertices.T), Array3i(faces.T))
    # No need to call configure() as it's done in the constructor

    # Create the mesh viewer with the mesh as a member
    viewer = ClosestPointMeshVisualizer(
        mesh=mesh,  # Pass the mesh directly to the constructor
        window_size=(1280, 960),
        mesh_color=(0.8, 0.8, 0.8, 0.5),  # Set translucent color
        mesh_material="wax",  # The "wax" material supports transparency
        show_wireframe=True
    )

    # Add the closest point visualization with translucent sphere
    viewer.add_closest_point_visualization(
        query_point_color=(1.0, 0.2, 0.2),
        query_point_radius=0.01,
        closest_point_color=(0.2, 1.0, 0.2),
        closest_point_radius=0.01,
        line_color=(0.9, 0.9, 0.9),
        line_width=0.002,
        normal_line_color=(0.0, 0.8, 0.8),
        normal_line_width=0.002,
        normal_line_length=0.1,
        show_closest_info=True,
        # Enable distance sphere visualization with transparency
        show_distance_sphere=True,
        # Added alpha value for transparency
        distance_sphere_color=(0.9, 0.5, 0.5, 0.3),
        distance_sphere_material="wax",  # The "wax" material supports transparency
        sphere_resolution=20,
        use_bvh=True  # Start with BVH acceleration enabled
    )

    # Show the visualization
    viewer.show()


if __name__ == "__main__":
    run_closest_point_mesh_demo()
