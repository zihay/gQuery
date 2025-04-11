"""
Mesh visualization tools for diff_wost.
"""

import polyscope as ps
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
import matplotlib.pyplot as plt
from gquery.shapes.mesh import Mesh

from .visualizer import Visualizer


class MeshViewer(Visualizer):
    """
    Class for visualizing mesh data with Polyscope.

    This class provides functionality to visualize and inspect meshes,
    including vertices, faces, vertex normals, and scalar/vector fields.
    """

    def __init__(
        self,
        mesh: Optional[Mesh] = None,
        visualizer: Optional[Visualizer] = None,
        window_size: Tuple[int, int] = (1024, 768),
        background_color: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        auto_init: bool = True,
        mesh_color: Optional[Tuple[float, float, float]] = (
            0.8, 0.8, 0.8, 0.3),
        mesh_material: str = "wax",
        show_wireframe: bool = True,
    ):
        """
        Initialize the mesh viewer.

        Args:
            mesh: Optional Mesh object to visualize (from diff_wost.shapes.mesh)
            visualizer: An existing Visualizer instance to use instead of creating a new one
            window_size: Tuple of (width, height) for the polyscope window
            background_color: Background color as RGB tuple of floats between 0 and 1
            auto_init: Automatically initialize Polyscope
            mesh_color: Color for the mesh if automatically visualized
            mesh_material: Material for the mesh if automatically visualized
            show_wireframe: Whether to show wireframe on the mesh
        """
        if visualizer is not None:
            # Inherit from an existing visualizer
            self.window_size = visualizer.window_size
            self.background_color = visualizer.background_color
            self._initialized = visualizer._initialized
            self._structures = visualizer._structures
        else:
            # Create a new visualizer
            super().__init__(window_size, background_color, auto_init)

        # Store the mesh object
        self.mesh = mesh

        # If a mesh is provided, automatically add it to the visualization
        if self.mesh is not None and self.is_initialized:
            self.add_mesh_from_object(
                color=mesh_color,
                material=mesh_material,
                show_wireframe=show_wireframe
            )
            self.add_face_normals(
                name="all_face_normals",
                radius=0.0008,
                enabled=False
            )

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

    def add_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        name: str = "mesh",
        color: Optional[Tuple[float, float, float]] = None,
        smooth_shade: bool = True,
        material: str = "wax",
        edge_width: float = 1.0,
        show_edges: bool = False,
        show_wireframe: bool = False,
        enabled: bool = True,
    ) -> Any:
        """
        Add a triangle mesh to the visualization.

        Args:
            vertices: Nx3 array of 3D vertex positions
            faces: Mx3 array of face indices (triangles)
            name: Name for the mesh structure
            color: RGB or RGBA color tuple (if None, uses default color)
            smooth_shade: Whether to use smooth shading
            material: Material name (e.g., 'wax', 'mud', 'candy', 'flat')
            edge_width: Width of edges when shown
            show_edges: Whether to show edges (deprecated, use show_wireframe instead)
            show_wireframe: Whether to show wireframe
            enabled: Whether the structure is visible

        Returns:
            The registered Polyscope surface mesh structure
        """
        # Make sure polyscope is initialized
        if not self.is_initialized:
            self.init()

        # Register the mesh
        mesh = ps.register_surface_mesh(
            name,
            vertices,
            faces,
            smooth_shade=smooth_shade,
            material=material,
            enabled=enabled
        )

        if color is not None:
            # Check if color includes alpha channel
            if len(color) == 4:
                rgb = color[:3]  # Get RGB values
                alpha = color[3]  # Get alpha value

                # Set color and transparency separately
                mesh.set_color(rgb)
                mesh.set_transparency(alpha)
            else:
                mesh.set_color(color)

        # Set edge width even if not showing edges initially
        if hasattr(mesh, 'set_edge_width'):
            mesh.set_edge_width(edge_width)

        # Handle wireframe visibility - different Polyscope versions may have different APIs
        if show_wireframe or show_edges:
            # Try different method names that might be available in different Polyscope versions
            if hasattr(mesh, 'set_wireframe_visible'):
                mesh.set_wireframe_visible(True)
            elif hasattr(mesh, 'set_wireframe'):
                mesh.set_wireframe(True)
            elif hasattr(mesh, 'set_wireframe_mode'):
                mesh.set_wireframe_mode(True)
            elif hasattr(mesh, 'set_edge_width') and hasattr(mesh, 'set_edge_color'):
                # Alternative approach: make edges visible with a width and color
                mesh.set_edge_width(edge_width)
                mesh.set_edge_color((0.2, 0.2, 0.2))  # Dark gray edges

        # Register in internal structures
        return self.register_structure(name, mesh)

    def add_vertex_scalar(
        self,
        mesh_name: str,
        values: np.ndarray,
        name: str = "scalar_field",
        cmap: str = "viridis",
        enabled: bool = True,
        vminmax: Optional[Tuple[float, float]] = None,
    ) -> Any:
        """
        Add a scalar field defined on mesh vertices.

        Args:
            mesh_name: Name of the mesh to add the scalar field to
            values: Array of scalar values (one per vertex)
            name: Name for the scalar field
            cmap: Colormap name
            enabled: Whether the scalar field is enabled
            vminmax: Optional (min, max) tuple for the colormap range

        Returns:
            The registered Polyscope scalar quantity
        """
        # Get the mesh structure
        mesh = self.get_structure(mesh_name)
        if mesh is None:
            raise ValueError(f"Mesh '{mesh_name}' not found")

        # Add the scalar quantity
        quantity = mesh.add_scalar_quantity(
            name,
            values,
            defined_on="vertices",
            enabled=enabled,
            cmap=cmap
        )

        # Set the colormap range if provided
        if vminmax is not None:
            quantity.set_map_range(vminmax[0], vminmax[1])

        return quantity

    def add_face_scalar(
        self,
        mesh_name: str,
        values: np.ndarray,
        name: str = "face_scalar",
        cmap: str = "viridis",
        enabled: bool = True,
        vminmax: Optional[Tuple[float, float]] = None,
    ) -> Any:
        """
        Add a scalar field defined on mesh faces.

        Args:
            mesh_name: Name of the mesh to add the scalar field to
            values: Array of scalar values (one per face)
            name: Name for the scalar field
            cmap: Colormap name
            enabled: Whether the scalar field is enabled
            vminmax: Optional (min, max) tuple for the colormap range

        Returns:
            The registered Polyscope scalar quantity
        """
        # Get the mesh structure
        mesh = self.get_structure(mesh_name)
        if mesh is None:
            raise ValueError(f"Mesh '{mesh_name}' not found")

        # Add the scalar quantity
        quantity = mesh.add_scalar_quantity(
            name,
            values,
            defined_on="faces",
            enabled=enabled,
            cmap=cmap
        )

        # Set the colormap range if provided
        if vminmax is not None:
            quantity.set_map_range(vminmax[0], vminmax[1])

        return quantity

    def add_vertex_vector(
        self,
        mesh_name: str,
        vectors: np.ndarray,
        name: str = "vector_field",
        enabled: bool = True,
        color: Optional[Tuple[float, float, float]] = None,
        radius: float = 0.001,
        length: float = 1.0,
    ) -> Any:
        """
        Add a vector field defined on mesh vertices.

        Args:
            mesh_name: Name of the mesh to add the vector field to
            vectors: Nx3 array of 3D vectors (one per vertex)
            name: Name for the vector field
            enabled: Whether the vector field is enabled
            color: RGB color tuple (if None, uses default color)
            radius: Radius of vector arrows
            length: Length scaling for the vectors

        Returns:
            The registered Polyscope vector quantity
        """
        # Get the mesh structure
        mesh = self.get_structure(mesh_name)
        if mesh is None:
            raise ValueError(f"Mesh '{mesh_name}' not found")

        # Add the vector quantity
        quantity = mesh.add_vector_quantity(
            name,
            vectors,
            defined_on="vertices",
            enabled=enabled,
            vectortype="standard",
            radius=radius,
            length=length
        )

        if color is not None:
            quantity.set_color(color)

        return quantity

    def add_vertex_colors(
        self,
        mesh_name: str,
        colors: np.ndarray,
        name: str = "colors",
        enabled: bool = True,
    ) -> Any:
        """
        Add per-vertex colors to a mesh.

        Args:
            mesh_name: Name of the mesh to add colors to
            colors: Nx3 array of RGB values (one per vertex)
            name: Name for the color field
            enabled: Whether the color field is enabled

        Returns:
            The registered Polyscope color quantity
        """
        # Get the mesh structure
        mesh = self.get_structure(mesh_name)
        if mesh is None:
            raise ValueError(f"Mesh '{mesh_name}' not found")

        # Add the color quantity
        quantity = mesh.add_color_quantity(
            name,
            colors,
            defined_on="vertices",
            enabled=enabled
        )

        return quantity

    def add_face_colors(
        self,
        mesh_name: str,
        colors: np.ndarray,
        name: str = "face_colors",
        enabled: bool = True,
    ) -> Any:
        """
        Add per-face colors to a mesh.

        Args:
            mesh_name: Name of the mesh to add colors to
            colors: Mx3 array of RGB values (one per face)
            name: Name for the color field
            enabled: Whether the color field is enabled

        Returns:
            The registered Polyscope color quantity
        """
        # Get the mesh structure
        mesh = self.get_structure(mesh_name)
        if mesh is None:
            raise ValueError(f"Mesh '{mesh_name}' not found")

        # Add the color quantity
        quantity = mesh.add_color_quantity(
            name,
            colors,
            defined_on="faces",
            enabled=enabled
        )

        return quantity

    def add_point_cloud(
        self,
        points: np.ndarray,
        name: str = "points",
        color: Optional[Tuple[float, float, float]] = None,
        point_radius: float = 0.005,
        enabled: bool = True,
    ) -> Any:
        """
        Add a point cloud to the visualization.

        Args:
            points: Nx3 array of point positions
            name: Name for the point cloud
            color: RGB color tuple (if None, uses default color)
            point_radius: Radius of the points
            enabled: Whether the structure is initially visible

        Returns:
            The polyscope point cloud structure
        """
        # Ensure polyscope is initialized
        if not self.is_initialized:
            self.init()

        # Register the point cloud
        ps_points = ps.register_point_cloud(
            name,
            points,
            radius=point_radius,
            enabled=enabled,
            color=color
        )

        return ps_points

    def add_polyline(
        self,
        points: np.ndarray,
        name: str = "polyline",
        color: Optional[Tuple[float, float, float]] = None,
        width: float = 0.002,
        closed: bool = False,
        enabled: bool = True,
    ) -> Any:
        """
        Add a polyline (sequence of connected line segments) to the visualization.

        Args:
            points: Nx3 array of point positions defining the polyline vertices
            name: Name for the polyline structure
            color: RGB color tuple (if None, uses default color)
            width: Width/radius of the polyline
            closed: Whether to close the polyline by connecting the last point to the first
            enabled: Whether the structure is initially visible

        Returns:
            The polyscope curve network structure
        """
        # Ensure polyscope is initialized
        if not self.is_initialized:
            self.init()

        # Create edges connecting consecutive points
        n_points = points.shape[0]
        edges = np.array([[i, i+1] for i in range(n_points-1)])

        # Add closing edge if requested
        if closed and n_points > 2:
            edges = np.vstack([edges, np.array([n_points-1, 0])])

        # Register the curve network
        ps_polyline = ps.register_curve_network(
            name,
            points,
            edges,
            radius=width,
            color=color,
            enabled=enabled
        )

        return ps_polyline

    def add_multi_polyline(
        self,
        points_list: List[np.ndarray],
        name: str = "multi_polyline",
        colors: Optional[List[Tuple[float, float, float]]] = None,
        width: float = 0.002,
        closed: bool = False,
        enabled: bool = True,
    ) -> Any:
        """
        Add multiple polylines to the visualization.

        Args:
            points_list: List of Nx3 arrays, each defining a separate polyline
            name: Base name for the polyline structures
            colors: List of RGB color tuples for each polyline (if None, uses default colors)
            width: Width/radius of the polylines
            closed: Whether to close the polylines by connecting the last point to the first
            enabled: Whether the structures are initially visible

        Returns:
            List of polyscope curve network structures
        """
        if colors is None:
            # Generate distinct colors if not provided
            n_lines = len(points_list)
            colors = []
            for i in range(n_lines):
                hue = i / max(1, n_lines - 1)  # Between 0 and 1
                # Simple HSV to RGB conversion for hue (assuming S=V=1)
                if hue < 1/3:
                    colors.append((1, 3*hue, 0))
                elif hue < 2/3:
                    colors.append((2-3*hue, 1, 0))
                else:
                    colors.append((0, 1, 3*hue-2))

        # Create polylines
        polylines = []
        for i, points in enumerate(points_list):
            color = colors[i] if i < len(colors) else None
            polyline_name = f"{name}_{i}"
            polyline = self.add_polyline(
                points,
                name=polyline_name,
                color=color,
                width=width,
                closed=closed,
                enabled=enabled
            )
            polylines.append(polyline)

        return polylines

    def add_mesh_from_object(
        self,
        name: str = "mesh",
        color: Optional[Tuple[float, float, float]] = (0.8, 0.8, 0.8),
        smooth_shade: bool = True,
        material: str = "wax",
        edge_width: float = 1.0,
        show_wireframe: bool = True,
        enabled: bool = True,
    ) -> Any:
        """
        Add the stored mesh object to the visualization.

        Args:
            name: Name for the mesh structure
            color: RGB or RGBA color tuple
            smooth_shade: Whether to use smooth shading
            material: Material name (e.g., 'wax', 'mud', 'candy', 'flat')
            edge_width: Width of edges when shown
            show_wireframe: Whether to show wireframe
            enabled: Whether the structure is visible

        Returns:
            The registered Polyscope surface mesh structure
        """
        if self.mesh is None:
            raise ValueError(
                "No mesh object is stored in this MeshViewer instance")

        # Extract vertices and faces from the mesh object
        vertices = self.mesh.vertices.numpy().T  # Transpose to get Nx3 format
        faces = self.mesh.indices.numpy().T  # Transpose to get Mx3 format

        # Add the mesh to the visualization
        return self.add_mesh(
            vertices=vertices,
            faces=faces,
            name=name,
            color=color,
            smooth_shade=smooth_shade,
            material=material,
            edge_width=edge_width,
            show_wireframe=show_wireframe,
            enabled=enabled
        )

    def add_face_normals(
        self,
        name: str = "face_normals",
        color: Tuple[float, float, float] = (0.0, 0.8, 0.8),
        length: float = 0.05,
        radius: float = 0.001,
        enabled: bool = True
    ) -> Any:
        """
        Add visualization of face normals for the mesh.

        Args:
            name: Name for the normal vectors visualization
            color: RGB color tuple for the normal vectors
            length: Length of the normal vectors
            radius: Radius of the normal vector lines
            enabled: Whether the visualization is initially visible

        Returns:
            The registered Polyscope vector field
        """
        if self.mesh is None:
            raise ValueError(
                "No mesh object is stored in this MeshViewer instance")

        # Get the mesh structure
        mesh_structure = self.get_structure("mesh")
        if mesh_structure is None:
            raise ValueError(
                "Mesh visualization not found. Call add_mesh_from_object first.")

        # Get face centers and normals from the mesh
        faces = self.mesh.indices.numpy().T
        vertices = self.mesh.vertices.numpy().T

        # Calculate face centers
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_centers = (v0 + v1 + v2) / 3.0

        # Use the precomputed face normals in the mesh
        face_normals = self.mesh.face_normals.numpy().T

        # Scale the normals by the specified length
        scaled_normals = face_normals * length

        # Create a vector field on the faces
        vector_field = mesh_structure.add_vector_quantity(
            name,
            scaled_normals,
            defined_on="faces",
            enabled=enabled,
            vectortype="ambient",
            radius=radius,
            color=color
        )

        return vector_field
