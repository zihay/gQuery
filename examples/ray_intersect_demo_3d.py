#!/usr/bin/env python
"""
Demo script showing how to use the ray intersection visualization for meshes.

This example creates a 3D mesh and displays an interactive visualization
that allows the user to control a ray and see its intersection with the mesh
in real-time.
"""

import time
import numpy as np
from examples.mesh_viewer import MeshViewer
from gquery.shapes.mesh import Mesh
from gquery.core.fwd import *
from gquery.util.obj_loader import load_obj_3d
import polyscope as ps
from typing import Optional, Tuple, Any, Dict, List


class RayIntersectionMeshVisualizer(MeshViewer):
    def _generate_random_rays(self,
                              num_rays: int,
                              min_bounds: np.ndarray,
                              max_bounds: np.ndarray) -> Tuple[List[Array3], List[Array3]]:
        """
        Generate random ray origins and directions for testing.

        Args:
            num_rays: Number of rays to generate
            min_bounds: Minimum bounds of the scene
            max_bounds: Maximum bounds of the scene

        Returns:
            Tuple of (origins, directions) lists
        """
        # Expand bounds slightly to ensure rays can start outside the mesh
        bounds_range = max_bounds - min_bounds
        expanded_min = min_bounds - bounds_range * 0.2
        expanded_max = max_bounds + bounds_range * 0.2

        # Generate random origins within the expanded bounds
        origins_np = np.random.uniform(
            expanded_min, expanded_max, size=(num_rays, 3))

        # Generate random directions (uniformly distributed on unit sphere)
        theta = np.random.uniform(0, np.pi, size=num_rays)
        phi = np.random.uniform(0, 2*np.pi, size=num_rays)

        directions_np = np.zeros((num_rays, 3))
        directions_np[:, 0] = np.sin(theta) * np.cos(phi)  # x
        directions_np[:, 1] = np.sin(theta) * np.sin(phi)  # y
        directions_np[:, 2] = np.cos(theta)  # z

        # Convert to Array3 for gquery
        origins = Array3(origins_np.T)
        directions = -dr.normalize(origins)
        return origins, directions

    def _run_baseline_test(self,
                           num_rays: int,
                           min_bounds: np.ndarray,
                           max_bounds: np.ndarray,
                           r_max: float = float('inf')) -> Dict[str, Any]:
        """
        Run a performance test for the baseline ray intersection method.

        Args:
            num_rays: Number of rays to test
            min_bounds: Minimum bounds of the scene
            max_bounds: Maximum bounds of the scene
            r_max: Maximum ray distance

        Returns:
            Dictionary with timing results
        """
        # Generate random rays for testing
        origins, directions = self._generate_random_rays(
            num_rays, min_bounds, max_bounds)

        # Test baseline method
        baseline_start = time.time()
        intersection = self.mesh.intersect_baseline(
            origins, directions, Float(r_max))
        dr.eval(intersection.d)
        print(dr.mean(intersection.d))
        baseline_time = time.time() - baseline_start

        # Calculate performance metrics
        results = {
            "total_time": baseline_time,
            "avg_time": (baseline_time * 1000) / num_rays,  # ms per ray
            "num_rays": num_rays
        }

        return results

    def _run_bvh_test(self,
                      num_rays: int,
                      min_bounds: np.ndarray,
                      max_bounds: np.ndarray,
                      r_max: float = float('inf')) -> Dict[str, Any]:
        """
        Run a performance test for the BVH ray intersection method.

        Args:
            num_rays: Number of rays to test
            min_bounds: Minimum bounds of the scene
            max_bounds: Maximum bounds of the scene
            r_max: Maximum ray distance

        Returns:
            Dictionary with timing results
        """
        # Generate random rays for testing
        origins, directions = self._generate_random_rays(
            num_rays, min_bounds, max_bounds)

        # Test BVH method
        bvh_start = time.time()
        intersection = self.mesh.intersect_bvh(
            origins, directions, Float(r_max))
        dr.eval(intersection.d)
        print(dr.mean(intersection.d))
        bvh_time = time.time() - bvh_start

        # Calculate performance metrics
        results = {
            "total_time": bvh_time,
            "avg_time": (bvh_time * 1000) / num_rays,  # ms per ray
            "num_rays": num_rays
        }

        return results

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
        # Normal visualization options
        show_normal: bool = False,
        normal_line_name: str = "normal_line",
        normal_line_color: Tuple[float, float, float] = (0.0, 0.8, 0.8),
        normal_line_width: float = 0.001,
        normal_line_length: float = 0.2,
        # Ray intersection parameters
        initial_r_max: float = float('inf'),
        use_bvh: bool = True,
        # UI options
        window_title: str = "Ray Controls",
        window_width: int = 380,
        window_height: int = 820,
        show_intersection_info: bool = True,
    ) -> Tuple[Any, np.ndarray, np.ndarray, Any, Any]:
        """
        Add an interactive ray with intersection visualization for meshes.

        This method adds an interactive ray and displays the intersection point
        on the mesh in real-time as the ray is moved or redirected.

        Args:
            initial_origin: Initial origin point of the ray (3D)
            initial_direction: Initial direction vector of the ray (3D)
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

            show_normal: Whether to show the normal at the intersection point
            normal_line_name: Name for the normal line structure
            normal_line_color: RGB color tuple for the normal line
            normal_line_width: Width of the normal line
            normal_line_length: Length of the normal line

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
        # Make sure polyscope is initialized and we have a mesh
        if not self.is_initialized:
            self.init()

        if self.mesh is None:
            raise ValueError(
                "RayIntersectionMeshVisualizer instance needs a mesh attribute.")

        # Configure Polyscope options
        ps.set_open_imgui_window_for_user_callback(False)

        # Determine bounds and initial position
        min_bounds, max_bounds, center = self._calculate_bounds(bounds)

        # Initialize ray origin and direction
        ray_origin_pos = np.array(
            initial_origin) if initial_origin is not None else center.copy()
        # Default direction: pointing forward
        ray_direction = np.array(
            initial_direction) if initial_direction is not None else np.array([0.0, 0.0, 1.0])
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

        # Initialize intersection data
        query_origin = Array3(float(ray_origin_pos[0]), float(
            ray_origin_pos[1]), float(ray_origin_pos[2]))
        query_direction = Array3(float(ray_direction[0]), float(
            ray_direction[1]), float(ray_direction[2]))

        # Query the intersection with r_max and use_bvh
        start_time = time.time()
        if current_use_bvh:
            intersection_record = self.mesh.intersect_bvh(
                query_origin, query_direction, Float(r_max))
        else:
            intersection_record = self.mesh.intersect_baseline(
                query_origin, query_direction, Float(r_max))
        dr.eval(intersection_record)
        last_query_time = (time.time() - start_time) * \
            1000  # Convert to milliseconds

        # Check if there's a valid intersection
        is_valid_intersection = bool(intersection_record.valid.numpy()[0])

        # Calculate the end point of the ray line and intersection point
        if is_valid_intersection:
            # Use the intersection distance
            intersection_distance = float(intersection_record.d.numpy()[0])
            # Limit to ray_length if needed
            ray_line_end = ray_origin_pos + ray_direction * \
                min(intersection_distance, ray_length)

            # Extract the intersection point position
            intersection_point_pos = np.array([
                intersection_record.p.x.numpy(),
                intersection_record.p.y.numpy(),
                intersection_record.p.z.numpy()
            ]).T
        else:
            # Just use the ray_length for visualization
            ray_line_end = ray_origin_pos + ray_direction * ray_length
            # Place a dummy intersection point at the end of the ray
            intersection_point_pos = np.array([ray_line_end])

        # Create the ray line visualization
        ray_line = self.add_polyline(
            points=np.vstack([ray_origin_pos, ray_line_end]),
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

        # Create normal line at intersection point if valid
        normal_line = None
        if show_normal and is_valid_intersection:
            normal_vec = np.array([
                intersection_record.n.x.numpy(),
                intersection_record.n.y.numpy(),
                intersection_record.n.z.numpy()
            ]).T * normal_line_length

            normal_line = self.add_polyline(
                points=np.vstack(
                    [intersection_point_pos, intersection_point_pos + normal_vec]),
                name=normal_line_name,
                color=normal_line_color,
                width=normal_line_width
            )
        else:
            # Create empty normal line structure
            normal_line = self.add_polyline(
                points=np.vstack(
                    [intersection_point_pos, intersection_point_pos]),
                name=normal_line_name,
                color=normal_line_color,
                width=normal_line_width,
                enabled=False
            )

        # Hide the intersection point and normal if there's no valid intersection
        if not is_valid_intersection:
            intersection_point.set_enabled(False)
            normal_line.set_enabled(False)

        # Keep track of window position and first frame state
        window_pos = np.array([self.window_size[0] - window_width - 20, 30])
        first_frame = True

        # Performance test variables
        perf_test_results = None
        is_running_baseline_test = False
        is_running_bvh_test = False
        test_num_rays = 10000
        baseline_results = None
        bvh_results = None

        # Define the callback function for the GUI
        def ray_intersection_gui_callback():
            nonlocal ray_origin_pos, ray_direction, window_pos, first_frame
            nonlocal intersection_record, last_query_time, intersection_point_pos
            nonlocal is_valid_intersection, r_max, current_use_bvh
            nonlocal perf_test_results, test_num_rays
            nonlocal is_running_baseline_test, is_running_bvh_test
            nonlocal baseline_results, bvh_results

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

                # Performance testing section
                ps.imgui.TextColored((1.0, 0.2, 0.6, 1.0),
                                     "Performance Testing")
                ps.imgui.Spacing()

                # Number of rays slider
                _, test_num_rays = ps.imgui.SliderInt("Number of Rays",
                                                      test_num_rays,
                                                      1000,
                                                      100000)
                ps.imgui.Spacing()

                # Run test buttons - side by side
                is_running_any_test = is_running_baseline_test or is_running_bvh_test

                if not is_running_any_test:
                    ps.imgui.BeginGroup()
                    # Baseline test button
                    if ps.imgui.Button("Test Brute Force", (150, 30)):
                        is_running_baseline_test = True
                        ps.info(
                            f"Starting brute force test with {test_num_rays} rays...")
                        baseline_results = self._run_baseline_test(
                            test_num_rays, min_bounds, max_bounds, r_max)
                        is_running_baseline_test = False
                        ps.info("Brute force test completed!")

                    ps.imgui.SameLine()

                    # BVH test button
                    if ps.imgui.Button("Test BVH", (150, 30)):
                        is_running_bvh_test = True
                        ps.info(
                            f"Starting BVH test with {test_num_rays} rays...")
                        bvh_results = self._run_bvh_test(
                            test_num_rays, min_bounds, max_bounds, r_max)
                        is_running_bvh_test = False
                        ps.info("BVH test completed!")
                    ps.imgui.EndGroup()

                else:
                    # Show that test is running
                    if is_running_baseline_test:
                        ps.imgui.Text("Running baseline test... please wait")
                    if is_running_bvh_test:
                        ps.imgui.Text("Running BVH test... please wait")

                ps.imgui.Spacing()

                # Display individual test results
                ps.imgui.Separator()
                ps.imgui.TextColored((1.0, 0.8, 0.2, 1.0), "Test Results")
                ps.imgui.Spacing()

                # Display baseline results if available
                if baseline_results is not None:
                    ps.imgui.TextColored(
                        (1.0, 0.6, 0.4, 1.0), "Baseline Method:")
                    ps.imgui.Text(
                        f"  Number of rays: {baseline_results['num_rays']}")
                    ps.imgui.Text(
                        f"  Total time: {baseline_results['total_time']:.3f} seconds")
                    ps.imgui.Text(
                        f"  Average per ray: {baseline_results['avg_time']:.3f} ms")
                    ps.imgui.Spacing()

                # Display BVH results if available
                if bvh_results is not None:
                    ps.imgui.TextColored((0.4, 0.8, 1.0, 1.0), "BVH Method:")
                    ps.imgui.Text(
                        f"  Number of rays: {bvh_results['num_rays']}")
                    ps.imgui.Text(
                        f"  Total time: {bvh_results['total_time']:.3f} seconds")
                    ps.imgui.Text(
                        f"  Average per ray: {bvh_results['avg_time']:.3f} ms")
                    ps.imgui.Spacing()

                # Display comparison if both results are available
                if baseline_results is not None and bvh_results is not None:
                    ps.imgui.Separator()
                    ps.imgui.TextColored(
                        (0.2, 1.0, 0.6, 1.0), "Performance Comparison")
                    ps.imgui.Spacing()

                    # Calculate speedup
                    speedup = baseline_results['total_time'] / \
                        bvh_results['total_time'] if bvh_results['total_time'] > 0 else float(
                            'inf')

                    # Highlight the speedup
                    if speedup > 1.0:
                        ps.imgui.TextColored((0.0, 1.0, 0.0, 1.0),
                                             f"BVH is {speedup:.2f}x faster than baseline")
                    else:
                        ps.imgui.TextColored((1.0, 0.5, 0.0, 1.0),
                                             f"Baseline is {1.0/speedup:.2f}x faster than BVH")

                    # Additional stats
                    speedup_percent = (
                        speedup - 1.0) * 100 if speedup > 1.0 else (1.0 - 1.0/speedup) * 100
                    ps.imgui.Text(
                        f"Performance improvement: {abs(speedup_percent):.1f}%")

                    # Time difference
                    time_diff = abs(
                        baseline_results['total_time'] - bvh_results['total_time'])
                    ps.imgui.Text(f"Time saved: {time_diff:.3f} seconds")

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

                changed_z, ray_origin_pos[2] = ps.imgui.SliderFloat("Origin Z",
                                                                    ray_origin_pos[2],
                                                                    min_bounds[2],
                                                                    max_bounds[2])
                ps.imgui.Spacing()

                # Direction controls using spherical coordinates
                ps.imgui.TextColored((1.0, 0.8, 0.0, 1.0), "Ray Direction")

                # Convert current direction to spherical coordinates
                theta = np.arccos(ray_direction[2])  # Polar angle (0 to π)
                # Azimuthal angle (-π to π)
                phi = np.arctan2(ray_direction[1], ray_direction[0])

                # Convert to degrees for UI
                theta_deg = np.degrees(theta)
                phi_deg = np.degrees(phi)

                # Slider for polar angle (0 to 180 degrees)
                changed_theta, new_theta_deg = ps.imgui.SliderFloat("Polar Angle",
                                                                    theta_deg,
                                                                    0.0,
                                                                    180.0)
                ps.imgui.Spacing()

                # Slider for azimuthal angle (-180 to 180 degrees)
                changed_phi, new_phi_deg = ps.imgui.SliderFloat("Azimuthal Angle",
                                                                phi_deg,
                                                                -180.0,
                                                                180.0)
                ps.imgui.Spacing()

                # If angles changed, update the direction vector
                if changed_theta or changed_phi:
                    # Convert back to radians
                    new_theta = np.radians(new_theta_deg)
                    new_phi = np.radians(new_phi_deg)

                    # Convert spherical to Cartesian
                    ray_direction[0] = np.sin(new_theta) * np.cos(new_phi)
                    ray_direction[1] = np.sin(new_theta) * np.sin(new_phi)
                    ray_direction[2] = np.cos(new_theta)

                    # Normalize to ensure unit vector
                    ray_direction = ray_direction / \
                        np.linalg.norm(ray_direction)

                # Ray max distance control
                ps.imgui.Spacing()
                ps.imgui.TextColored((0.5, 0.8, 1.0, 1.0), "Ray Parameters")
                # Calculate a reasonable max value for the slider based on the scene bounds
                scene_diameter = np.linalg.norm(max_bounds - min_bounds)
                # Ensure a reasonable upper limit
                slider_max = scene_diameter * 2

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

                changed_any = changed_x or changed_y or changed_z or \
                    changed_theta or changed_phi or \
                    changed_r_max or changed_bvh

                # Update visualizations if anything changed
                if changed_any:
                    # Update ray origin in visualization
                    ray_origin.update_point_positions(
                        np.array([ray_origin_pos]))

                    # Query the intersection with timing
                    query_origin = Array3(float(ray_origin_pos[0]), float(
                        ray_origin_pos[1]), float(ray_origin_pos[2]))
                    query_direction = Array3(float(ray_direction[0]), float(
                        ray_direction[1]), float(ray_direction[2]))

                    start_time = time.time()

                    # Use the updated intersect method with use_bvh parameter
                    if current_use_bvh:
                        intersection_record = self.mesh.intersect_bvh(
                            query_origin, query_direction, Float(r_max))
                    else:
                        intersection_record = self.mesh.intersect_baseline(
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
                            intersection_record.p.z.numpy()
                        ]).T

                        # Show and update intersection point
                        intersection_point.set_enabled(True)
                        intersection_point.update_point_positions(
                            intersection_point_pos)

                        # Update normal line if show_normal is enabled
                        if show_normal:
                            normal_vec = np.array([
                                intersection_record.n.x.numpy(),
                                intersection_record.n.y.numpy(),
                                intersection_record.n.z.numpy()
                            ]).T * normal_line_length

                            normal_line.set_enabled(True)
                            normal_line.update_node_positions(
                                np.vstack(
                                    [intersection_point_pos, intersection_point_pos + normal_vec])
                            )
                    else:
                        # Just use the ray_length for visualization
                        ray_line_end = ray_origin_pos + ray_direction * ray_length
                        # Hide intersection point and normal
                        intersection_point.set_enabled(False)
                        if normal_line is not None:
                            normal_line.set_enabled(False)

                    # Update ray line
                    ray_line.update_node_positions(
                        np.vstack([ray_origin_pos, ray_line_end]))

                # Display the current ray information
                ps.imgui.Separator()
                ps.imgui.Text(
                    f"Origin: ({ray_origin_pos[0]:.3f}, {ray_origin_pos[1]:.3f}, {ray_origin_pos[2]:.3f})")
                ps.imgui.Text(
                    f"Direction: ({ray_direction[0]:.3f}, {ray_direction[1]:.3f}, {ray_direction[2]:.3f})")
                ps.imgui.Text(f"Polar Angle: {theta_deg:.1f}°")
                ps.imgui.Text(f"Azimuthal Angle: {phi_deg:.1f}°")
                ps.imgui.Text(
                    f"Max Distance: {r_max if r_max != float('inf') else 'Unlimited'}")
                ps.imgui.Text(
                    f"Method: {'BVH' if current_use_bvh else 'Baseline'}")
                ps.imgui.Spacing()

                # Reset button
                if ps.imgui.Button("Reset Ray", (150, 30)):
                    ray_origin_pos = center.copy()
                    ray_direction = np.array([0.0, 0.0, 1.0])
                    r_max = initial_r_max
                    current_use_bvh = use_bvh

                    # Update ray origin in visualization
                    ray_origin.update_point_positions(
                        np.array([ray_origin_pos]))

                    # Query the intersection with timing
                    query_origin = Array3(float(ray_origin_pos[0]), float(
                        ray_origin_pos[1]), float(ray_origin_pos[2]))
                    query_direction = Array3(float(ray_direction[0]), float(
                        ray_direction[1]), float(ray_direction[2]))

                    start_time = time.time()

                    # Use the updated intersect method with use_bvh parameter
                    if current_use_bvh:
                        intersection_record = self.mesh.intersect_bvh(
                            query_origin, query_direction, Float(r_max))
                    else:
                        intersection_record = self.mesh.intersect_baseline(
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
                            intersection_record.p.z.numpy()
                        ]).T

                        # Show and update intersection point
                        intersection_point.set_enabled(True)
                        intersection_point.update_point_positions(
                            intersection_point_pos)

                        # Update normal line if show_normal is enabled
                        if show_normal:
                            normal_vec = np.array([
                                intersection_record.n.x.numpy(),
                                intersection_record.n.y.numpy(),
                                intersection_record.n.z.numpy()
                            ]).T * normal_line_length

                            normal_line.set_enabled(True)
                            normal_line.update_node_positions(
                                np.vstack(
                                    [intersection_point_pos, intersection_point_pos + normal_vec])
                            )
                    else:
                        # Just use the ray_length for visualization
                        ray_line_end = ray_origin_pos + ray_direction * ray_length
                        # Hide intersection point and normal
                        intersection_point.set_enabled(False)
                        if normal_line is not None:
                            normal_line.set_enabled(False)

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
                        ip_z = intersection_point_pos[0][2]

                        # Normal at intersection
                        n_x = intersection_record.n.x.numpy()[0]
                        n_y = intersection_record.n.y.numpy()[0]
                        n_z = intersection_record.n.z.numpy()[0]

                        # Distance from origin to intersection
                        distance = intersection_record.d.numpy()[0]

                        # Display all information about the intersection
                        ps.imgui.Text(
                            f"Position: ({ip_x:.3f}, {ip_y:.3f}, {ip_z:.3f})")
                        ps.imgui.Text(
                            f"Normal: ({n_x:.3f}, {n_y:.3f}, {n_z:.3f})")
                        ps.imgui.Text(f"Distance: {distance:.5f}")
                        ps.imgui.Text(
                            f"Triangle ID: {intersection_record.prim_id.numpy()[0]}")
                        if hasattr(intersection_record, 'uv'):
                            ps.imgui.Text(
                                f"UV: ({intersection_record.uv.x.numpy()[0]:.3f}, {intersection_record.uv.y.numpy()[0]:.3f})")
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


def run_ray_intersection_mesh_demo():
    """
    Run a demonstration of the ray intersection visualization feature for meshes.
    """
    # Load a 3D mesh (bunny model)
    vertices, faces = load_obj_3d(BASE_DIR / "data/bunny_hi.obj")

    # Create the mesh object
    mesh = Mesh(Array3(vertices.T), Array3i(faces.T))
    # No need to call configure() as it's done in the constructor

    # Create the mesh viewer
    viewer = RayIntersectionMeshVisualizer(
        mesh=mesh,
        window_size=(1280, 960),
        mesh_color=(0.8, 0.8, 0.8, 0.5),  # Set translucent color for the mesh
        mesh_material="wax",  # The "wax" material supports transparency
        show_wireframe=True
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
        ray_length=2.0,
        # Show normal at intersection point
        show_normal=False,
        normal_line_color=(0.0, 0.8, 0.8),
        normal_line_width=0.002,
        normal_line_length=0.1,
        # Set initial maximum ray distance (can be adjusted in the UI)
        initial_r_max=float('inf'),  # Start with unlimited distance
        # Enable BVH acceleration for faster ray intersection
        use_bvh=True,
        # Show information about the intersection
        show_intersection_info=True
    )

    # Show the visualization
    viewer.show()


if __name__ == "__main__":
    run_ray_intersection_mesh_demo()
