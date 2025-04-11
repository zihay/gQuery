"""
Base visualizer module that sets up and manages Polyscope.
"""

import polyscope as ps
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Union

rectangles = []

def add_rectangle(p_min, p_max, color=(1, 0, 0), index = [0]):
    vertices = np.array([
        [p_min[0], p_min[1], 0],
        [p_max[0], p_min[1], 0],
        [p_max[0], p_max[1], 0],
        [p_min[0], p_max[1], 0]
    ])
    edges = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
    ])
    rectangle = ps.register_curve_network(
        f"rectangle_{index[0]}",
        vertices,
        edges,
        radius=0.002,
        color=color
    )
    rectangles.append(f"rectangle_{index[0]}")
    index[0] += 1
    

def clear_rectangles():
    for rectangle in rectangles:
        ps.remove_curve_network(rectangle)
    rectangles.clear()

class Visualizer:
    """
    Base class for all visualizers in diff_wost using Polyscope.

    Handles initialization of Polyscope and provides global configuration options.
    Specialized viewers should inherit from this class.
    """

    def __init__(
        self,
        window_size: Tuple[int, int] = (1024, 768),
        background_color: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        auto_init: bool = True,
    ):
        """
        Initialize the visualizer with Polyscope.

        Args:
            window_size: Tuple of (width, height) for the polyscope window
            background_color: Background color as RGB tuple of floats between 0 and 1
            auto_init: Automatically initialize Polyscope
        """
        self.window_size = window_size
        self.background_color = background_color
        self._initialized = False
        self._structures = {}  # Keep track of registered structures

        if auto_init:
            self.init()

    def init(self):
        """Initialize Polyscope with configured settings."""
        if not self._initialized:
            ps.init()
            ps.set_program_name("diff_wost Visualization")
            ps.set_verbosity(0)  # Reduce console output
            ps.set_window_size(self.window_size[0], self.window_size[1])
            ps.set_ground_plane_mode("none")
            ps.set_background_color(self.background_color)
            self._initialized = True

    def show(self, callback=None):
        """
        Show the visualization.

        Args:
            callback: Optional callback function to run in the Polyscope main loop
        """
        if not self._initialized:
            self.init()

        if callback:
            ps.set_user_callback(callback)

        ps.show()

    def clear_all(self):
        """Remove all structures from the visualization."""
        ps.remove_all_structures()
        self._structures = {}

    def set_up_camera(
        self,
        camera_position: np.ndarray = None,
        look_at: np.ndarray = None,
        up_direction: np.ndarray = None
    ):
        """
        Set up the camera for the visualization.

        Args:
            camera_position: 3D position of the camera
            look_at: Point the camera is looking at
            up_direction: Up direction for the camera (Note: not directly supported by Polyscope)
        """
        # Make sure polyscope is initialized
        if not self.is_initialized:
            self.init()

        # Check if we have both camera position and look_at target
        if camera_position is not None and look_at is not None:
            # The look_at function in Polyscope doesn't accept up_direction as a parameter
            # Instead, it accepts a boolean for "fly_to" animation
            # We'll ignore up_direction and use the default Polyscope behavior
            ps.look_at(camera_position, look_at, fly_to=True)

            # Note: Unfortunately current Polyscope API doesn't directly support setting the up direction
            print(
                "Note: Camera position set. Use mouse controls to adjust the view if needed.")
        else:
            print(
                "Note: Both camera_position and look_at are required to set camera position.")
            print("      Use mouse controls to navigate in the viewer window.")

    def capture_screenshot(self, filename: str):
        """
        Capture a screenshot of the current view.

        Args:
            filename: Path to save the screenshot
        """
        ps.screenshot(filename)

    def register_structure(self, name: str, structure: Any):
        """
        Register a structure in the internal tracking dictionary.

        Args:
            name: Name of the structure
            structure: The Polyscope structure object
        """
        self._structures[name] = structure
        return structure

    def get_structure(self, name: str) -> Optional[Any]:
        """
        Get a registered structure by name.

        Args:
            name: Name of the structure to retrieve

        Returns:
            The structure if found, None otherwise
        """
        return self._structures.get(name)

    def remove_structure(self, name: str) -> bool:
        """
        Remove a structure by name.

        Args:
            name: Name of the structure to remove

        Returns:
            True if the structure was found and removed, False otherwise
        """
        if name in self._structures:
            ps.remove_structure(name)
            del self._structures[name]
            return True
        return False

    def add_polyline(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        name: str = "polyline",
        color: Optional[Tuple[float, float, float]] = None,
        width: float = 0.002,
        enabled: bool = True,
    ) -> Any:
        """
        Add a polyline to the visualization using explicit edges.

        Args:
            vertices: Nx3 array of 3D points
            edges: Mx2 array of edge indices
            name: Name for the polyline structure
            color: RGB color tuple for the polyline (if None, uses default color)
            width: Width of the polyline
            enabled: Whether the structure is visible

        Returns:
            The registered Polyscope curve network structure
        """
        # Make sure polyscope is initialized
        if not self.is_initialized:
            self.init()

        # Register the curve network
        curve = ps.register_curve_network(
            name,
            vertices,
            edges,
            radius=width,
            enabled=enabled
        )

        if color is not None:
            curve.set_color(color)

        # Register in internal structures
        return self.register_structure(name, curve)

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
            points: Nx3 array of 3D points
            name: Name for the point cloud structure
            color: RGB color tuple for the points (if None, uses default color)
            point_radius: Radius of the points
            enabled: Whether the structure is visible

        Returns:
            The registered Polyscope point cloud structure
        """
        # Make sure polyscope is initialized
        if not self.is_initialized:
            self.init()

        # Register the point cloud
        cloud = ps.register_point_cloud(
            name,
            points,
            radius=point_radius,
            enabled=enabled
        )

        if color is not None:
            cloud.set_color(color)

        # Register in internal structures
        return self.register_structure(name, cloud)

    def add_scalar_quantity(
        self,
        structure_name: str,
        values: np.ndarray,
        name: str = "scalar_field",
        defined_on: str = "vertices",
        cmap: str = "viridis",
        enabled: bool = True,
        vminmax: Optional[Tuple[float, float]] = None,
    ) -> Any:
        """
        Add a scalar field to an existing polyline or point cloud.

        Args:
            structure_name: Name of the structure to add the scalar field to
            values: Array of scalar values
            name: Name for the scalar field
            defined_on: 'vertices' or 'edges'
            cmap: Colormap name
            enabled: Whether the scalar field is enabled
            vminmax: Optional (min, max) tuple for the colormap range

        Returns:
            The registered Polyscope scalar quantity
        """
        # Get the structure
        structure = self.get_structure(structure_name)
        if structure is None:
            raise ValueError(f"Structure '{structure_name}' not found")

        # Add the scalar quantity
        quantity = structure.add_scalar_quantity(
            name,
            values,
            defined_on=defined_on,
            enabled=enabled,
            cmap=cmap
        )

        # Set the colormap range if provided
        if vminmax is not None:
            quantity.set_map_range(vminmax[0], vminmax[1])

        return quantity

    def add_color_quantity(
        self,
        structure_name: str,
        colors: np.ndarray,
        name: str = "color_field",
        defined_on: str = "vertices",
        enabled: bool = True,
    ) -> Any:
        """
        Add a color field to an existing polyline or point cloud.

        Args:
            structure_name: Name of the structure to add the color field to
            colors: Nx3 array of RGB values
            name: Name for the color field
            defined_on: 'vertices' or 'edges'
            enabled: Whether the color field is enabled

        Returns:
            The registered Polyscope color quantity
        """
        # Get the structure
        structure = self.get_structure(structure_name)
        if structure is None:
            raise ValueError(f"Structure '{structure_name}' not found")

        # Add the color quantity
        quantity = structure.add_color_quantity(
            name,
            colors,
            defined_on=defined_on,
            enabled=enabled
        )

        return quantity

    def add_vector_quantity(
        self,
        structure_name: str,
        vectors: np.ndarray,
        name: str = "vector_field",
        defined_on: str = "vertices",
        enabled: bool = True,
        vector_type: str = "standard",
        length: float = 1.0,
        radius: float = 0.0025,
        color: Optional[Tuple[float, float, float]] = None,
    ) -> Any:
        """
        Add a vector field to an existing polyline or point cloud.

        Args:
            structure_name: Name of the structure to add the vector field to
            vectors: Nx3 array of 3D vectors
            name: Name for the vector field
            defined_on: 'vertices' or 'edges'
            enabled: Whether the vector field is enabled
            vector_type: 'standard' or 'ambient'
            length: Length scaling for the vectors
            radius: Radius of vector arrows
            color: RGB color tuple for the vectors (if None, uses default color)

        Returns:
            The registered Polyscope vector quantity
        """
        # Get the structure
        structure = self.get_structure(structure_name)
        if structure is None:
            raise ValueError(f"Structure '{structure_name}' not found")

        # Add the vector quantity
        quantity = structure.add_vector_quantity(
            name,
            vectors,
            defined_on=defined_on,
            enabled=enabled,
            vectortype=vector_type,
            length=length,
            radius=radius
        )

        if color is not None:
            quantity.set_color(color)

        return quantity

    @property
    def is_initialized(self) -> bool:
        """Check if Polyscope has been initialized."""
        return self._initialized
