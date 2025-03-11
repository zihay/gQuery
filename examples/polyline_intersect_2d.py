import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import drjit as dr
from core.fwd import *
from shapes.polyline import Polyline
from enum import IntEnum, auto
import os
import glob
from util.obj_loader import load_obj_2d, load_obj_3d


class QueryType(IntEnum):
    RAY_INTERSECTION = 0
    CLOSEST_POINT = auto()
    CLOSEST_SILHOUETTE = auto()


class ShapeType(IntEnum):
    POLYLINE = 0
    MESH = auto()


class GeometryQueryDemo:
    def __init__(self):
        # Initialize parameters
        self.query_type = QueryType.RAY_INTERSECTION
        self.shape_type = ShapeType.POLYLINE

        # Find available obj files
        self.obj_files = self.find_obj_files()

        # Shape parameters
        self.shape_params = {
            'obj_path': self.obj_files[0] if self.obj_files else None,
            'obj_index': 0,  # Index of selected OBJ file
            'projection': 'xy',  # Projection plane for 3D meshes
        }

        # Query parameters
        self.query_params = {
            'origin': np.array([0.0, 0.0]),
            'direction': np.array([1.0, 0.0]),
            'max_distance': 10.0
        }

        # Results
        self.result = {
            'valid': False,
            'point': np.array([0.0, 0.0]),
            'distance': 0.0,
            'normal': np.array([0.0, 0.0])
        }

        # Initialize Polyscope
        ps.init()
        ps.set_up_dir("z")
        ps.set_ground_plane_mode("none")

        # Create shape and set up visualization
        self.create_shape()
        self.update_query_visualization()
        self.perform_query()

        # Set up callback for UI
        ps.set_user_callback(self.callback)

    def find_obj_files(self):
        """Find all OBJ files in the data directory"""
        obj_files = glob.glob("data/*.obj")
        return sorted(obj_files)

    def create_shape(self):
        """Create polyline shape and register with Polyscope"""
        # Load shape from OBJ file
        obj_path = self.shape_params['obj_path']

        if not obj_path or not os.path.exists(obj_path):
            # Fallback to a simple shape if no obj file is available
            vertices = np.array([(np.cos(t), np.sin(t))
                                for t in np.linspace(0, 2*np.pi, 30, endpoint=False)])
            edges = np.array([(i, (i + 1) % len(vertices))
                             for i in range(len(vertices))])
        else:
            # Load from OBJ file
            if self.shape_type == ShapeType.POLYLINE:
                vertices, edges = load_obj_2d(obj_path)
            else:  # ShapeType.MESH
                projection = self.shape_params['projection']
                vertices, edges = project_obj_3d_to_2d(
                    obj_path, projection=projection)

        # Convert to drjit arrays
        self.vertices = Array2(vertices.T)
        self.indices = Array2i(edges.T)

        # Create a polyline object
        self.polyline = Polyline()
        self.polyline.init(self.vertices, self.indices)

        # Register with Polyscope
        self.ps_points = ps.register_point_cloud("vertices", vertices)
        self.ps_points.add_color_quantity(
            "colors", np.ones((len(vertices), 3)) * [0.2, 0.5, 0.8])

        self.ps_curve = ps.register_curve_network("polyline", vertices, edges)
        self.ps_curve.set_color((0.2, 0.5, 0.8))
        self.ps_curve.set_radius(0.005)

        # Store current shape parameters
        self.last_shape_type = self.shape_type
        self.last_shape_params = self.shape_params.copy()

    def update_query_visualization(self):
        """Update the visualization for the current query type"""
        self.clean_up_query_visualization()

        if self.query_type == QueryType.RAY_INTERSECTION:
            self._update_ray_visualization()
        elif self.query_type == QueryType.CLOSEST_POINT:
            self._update_point_visualization()
        elif self.query_type == QueryType.CLOSEST_SILHOUETTE:
            self._update_direction_visualization()

    def _update_ray_visualization(self):
        """Visualize ray for intersection query"""
        origin = self.query_params['origin']
        direction = self.query_params['direction']
        max_dist = self.query_params['max_distance']

        # Normalize direction
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)

        # Create ray points and edges
        ray_end = origin + direction * max_dist
        ray_points = np.vstack([origin, ray_end])
        ray_edges = np.array([[0, 1]])

        # Register ray curve
        self.ps_query_viz = ps.register_curve_network(
            "query_ray", ray_points, ray_edges)
        self.ps_query_viz.set_color((1.0, 0.3, 0.3))
        self.ps_query_viz.set_radius(0.003)

    def _update_point_visualization(self):
        """Visualize point for closest point query"""
        point = self.query_params['origin'].reshape(1, 2)

        # Register query point
        self.ps_query_viz = ps.register_point_cloud("query_point", point)
        self.ps_query_viz.set_color((1.0, 0.5, 0.0))
        self.ps_query_viz.set_radius(0.015)

    def _update_direction_visualization(self):
        """Visualize direction for silhouette query (placeholder)"""
        # For future implementation
        pass

    def perform_query(self):
        """Perform the current query type"""
        # Clear previous results
        self.clear_result_visualization()
        self.result['valid'] = False

        # Perform appropriate query
        if self.query_type == QueryType.RAY_INTERSECTION:
            self._perform_intersection_test()
        elif self.query_type == QueryType.CLOSEST_POINT:
            self._perform_closest_point_test()
        elif self.query_type == QueryType.CLOSEST_SILHOUETTE:
            self._perform_closest_silhouette_test()

        # Update result visualization if valid
        if self.result['valid']:
            self.update_result_visualization()

    def _perform_intersection_test(self):
        """Perform ray intersection test"""
        # Get query parameters
        origin = self.query_params['origin']
        direction = self.query_params['direction']
        max_dist = self.query_params['max_distance']

        # Create drjit Arrays
        ray_origin = Array2(origin[0], origin[1])
        ray_direction = Array2(direction[0], direction[1])

        # Normalize the direction
        norm = dr.sqrt(dr.sum(ray_direction * ray_direction))
        if norm > 0:
            ray_direction = ray_direction / norm

        # Perform intersection test
        intersection = self.polyline.intersect(
            ray_origin, ray_direction, r_max=Float(max_dist))

        # Check if we have a valid intersection
        if dr.any(intersection.valid):
            # Convert drjit intersection point to numpy
            self.result['valid'] = True
            self.result['point'] = np.array([
                dr.numpy(intersection.p.x)[0],
                dr.numpy(intersection.p.y)[0]
            ])
            self.result['distance'] = dr.numpy(intersection.d)[0]
            self.result['normal'] = np.array([
                dr.numpy(intersection.n.x)[0],
                dr.numpy(intersection.n.y)[0]
            ])

    def _perform_closest_point_test(self):
        """Perform closest point test (placeholder)"""
        # Placeholder for future implementation
        pass

    def _perform_closest_silhouette_test(self):
        """Perform closest silhouette test (placeholder)"""
        # Placeholder for future implementation
        pass

    def update_result_visualization(self):
        """Update visualization of query result"""
        if not self.result['valid']:
            return

        # Visualize result point
        self.ps_result = ps.register_point_cloud("query_result",
                                                 self.result['point'].reshape(1, 2))
        self.ps_result.set_color((1.0, 0.1, 0.1))
        self.ps_result.set_radius(0.02)

        # For ray intersection, visualize the normal
        if self.query_type == QueryType.RAY_INTERSECTION:
            normal_start = self.result['point']
            normal_end = self.result['point'] + self.result['normal'] * 0.2
            normal_points = np.vstack([normal_start, normal_end])
            normal_edges = np.array([[0, 1]])

            self.ps_normal = ps.register_curve_network(
                "normal", normal_points, normal_edges)
            self.ps_normal.set_color((0.1, 0.9, 0.1))
            self.ps_normal.set_radius(0.002)

    def clean_up_query_visualization(self):
        """Clean up query visualization"""
        viz_names = {
            QueryType.RAY_INTERSECTION: "query_ray",
            QueryType.CLOSEST_POINT: "query_point",
            QueryType.CLOSEST_SILHOUETTE: "query_direction"
        }

        # Remove the appropriate visualization based on visualization type
        if hasattr(self, 'ps_query_viz') and self.ps_query_viz is not None:
            ps.remove_structure(viz_names.get(self.query_type, "query_viz"))
            self.ps_query_viz = None

    def clear_result_visualization(self):
        """Clear result visualization"""
        if hasattr(self, 'ps_result') and self.ps_result is not None:
            ps.remove_point_cloud("query_result")
            self.ps_result = None

        if hasattr(self, 'ps_normal') and self.ps_normal is not None:
            ps.remove_curve_network("normal")
            self.ps_normal = None

    def clean_up_visualizations(self):
        """Clean up all visualizations"""
        self.clean_up_query_visualization()
        self.clear_result_visualization()

    def shape_parameters_changed(self):
        """Check if shape parameters have changed enough to recreate the shape"""
        if self.shape_type != self.last_shape_type:
            return True

        if self.shape_params['obj_index'] != self.last_shape_params.get('obj_index'):
            return True

        if self.shape_type == ShapeType.MESH:
            if self.shape_params['projection'] != self.last_shape_params.get('projection'):
                return True

        return False

    def callback(self):
        """UI callback for Polyscope"""
        changed = False
        query_type_changed = False

        # Set consistent UI width
        psim.PushItemWidth(200)

        # Title
        psim.TextUnformatted("Shape and Query Controls")

        # --- Query type selection ---
        query_types = ["Ray Intersection",
                       "Closest Point (Coming Soon)", "Silhouette (Coming Soon)"]
        current_type = query_types[self.query_type]

        if psim.BeginCombo("Query Type", current_type):
            for i, query_name in enumerate(query_types):
                _, selected = psim.Selectable(query_name, self.query_type == i)
                if selected and i != self.query_type:
                    self.query_type = QueryType(i)
                    query_type_changed = True
                    changed = True
                    if i > 0:
                        ps.warning(
                            "This query type is not fully implemented yet")
            psim.EndCombo()

        # --- Shape type selection ---
        shape_types = ["Polyline", "Mesh"]
        current_shape = shape_types[self.shape_type]

        if psim.BeginCombo("Shape Type", current_shape):
            for i, shape_name in enumerate(shape_types):
                _, selected = psim.Selectable(shape_name, self.shape_type == i)
                if selected and i != self.shape_type:
                    self.shape_type = ShapeType(i)
                    changed = True
            psim.EndCombo()

        # --- OBJ file selection ---
        if self.obj_files:
            current_obj = os.path.basename(self.shape_params['obj_path'])
            if psim.BeginCombo("Object File", current_obj):
                for i, obj_file in enumerate(self.obj_files):
                    _, selected = psim.Selectable(os.path.basename(obj_file),
                                                  i == self.shape_params['obj_index'])
                    if selected and i != self.shape_params['obj_index']:
                        self.shape_params['obj_index'] = i
                        self.shape_params['obj_path'] = self.obj_files[i]
                        changed = True
                psim.EndCombo()
        else:
            psim.TextColored((1.0, 0.5, 0.0, 1.0),
                             "No OBJ files found in data folder")

        # --- 3D Projection selection (for Mesh type) ---
        if self.shape_type == ShapeType.MESH:
            projections = ["xy", "xz", "yz"]
            current_proj = self.shape_params['projection']

            if psim.BeginCombo("Projection Plane", current_proj):
                for proj in projections:
                    _, selected = psim.Selectable(proj, proj == current_proj)
                    if selected and proj != current_proj:
                        self.shape_params['projection'] = proj
                        changed = True
                psim.EndCombo()

        # Separator
        psim.Separator()

        # --- Query-specific controls ---
        if self.query_type == QueryType.RAY_INTERSECTION:
            psim.TextUnformatted("Ray Parameters:")

            # Ray origin controls
            _, x = psim.SliderFloat(
                "Origin X", self.query_params['origin'][0], -2.0, 2.0)
            _, y = psim.SliderFloat(
                "Origin Y", self.query_params['origin'][1], -2.0, 2.0)
            self.query_params['origin'] = np.array([x, y])

            # Ray direction controls
            _, dx = psim.SliderFloat(
                "Direction X", self.query_params['direction'][0], -1.0, 1.0)
            _, dy = psim.SliderFloat(
                "Direction Y", self.query_params['direction'][1], -1.0, 1.0)
            self.query_params['direction'] = np.array([dx, dy])

        elif self.query_type == QueryType.CLOSEST_POINT:
            psim.TextUnformatted("Point Parameters:")

            # Point controls
            _, x = psim.SliderFloat(
                "Point X", self.query_params['origin'][0], -2.0, 2.0)
            _, y = psim.SliderFloat(
                "Point Y", self.query_params['origin'][1], -2.0, 2.0)
            self.query_params['origin'] = np.array([x, y])

        elif self.query_type == QueryType.CLOSEST_SILHOUETTE:
            psim.TextUnformatted("Direction Parameters:")

            # Direction controls
            _, dx = psim.SliderFloat(
                "Direction X", self.query_params['direction'][0], -1.0, 1.0)
            _, dy = psim.SliderFloat(
                "Direction Y", self.query_params['direction'][1], -1.0, 1.0)
            self.query_params['direction'] = np.array([dx, dy])

        # --- Display result info ---
        psim.Separator()

        if self.result['valid']:
            pt = self.result['point']
            psim.TextUnformatted(f"Result Point: ({pt[0]:.3f}, {pt[1]:.3f})")
            psim.TextUnformatted(f"Distance: {self.result['distance']:.3f}")

            if self.query_type == QueryType.RAY_INTERSECTION:
                n = self.result['normal']
                psim.TextUnformatted(f"Normal: ({n[0]:.3f}, {n[1]:.3f})")
        else:
            psim.TextColored((1.0, 0.5, 0.0, 1.0), "No query result found")

        # Show shape statistics
        if hasattr(self, 'vertices') and hasattr(self, 'indices'):
            vertex_count = len(dr.numpy(self.vertices.x))
            edge_count = len(dr.numpy(self.indices.x))
            psim.TextUnformatted(
                f"Shape: {vertex_count} vertices, {edge_count} edges")
            psim.TextUnformatted(
                f"File: {os.path.basename(self.shape_params['obj_path'])}")

        # --- Update button ---
        if psim.Button("Update Visualization"):
            changed = True

        psim.PopItemWidth()

        # --- Handle visualization updates ---
        if changed:
            # If query type changed, clean up old visualizations
            if query_type_changed:
                self.clean_up_visualizations()

            # Update the query visualization
            self.update_query_visualization()

            # If shape parameters changed, recreate the shape
            if self.shape_parameters_changed():
                # Clean up existing shape visualizations
                ps.remove_curve_network("polyline")
                ps.remove_point_cloud("vertices")

                # Create new shape
                self.create_shape()

            # Perform query with updated parameters
            self.perform_query()


def main():
    demo = GeometryQueryDemo()
    ps.show()


if __name__ == "__main__":
    main()
