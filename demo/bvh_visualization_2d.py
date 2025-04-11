#!/usr/bin/env python3
"""
BVH Visualization Example

This script demonstrates how to visualize a 2D BVH (Bounding Volume Hierarchy) 
constructed from line segments using Polyscope.

The visualization includes:
- The input line segments
- The hierarchy of bounding boxes
- Interactive controls to explore the BVH levels
- Leaf node visualization
- Vertex validation within leaf nodes
"""

import numpy as np
import polyscope as ps
import gquery.gquery_ext as gq
import argparse
import os

from gquery.util.obj_loader import load_obj_2d


class BVHVisualizer:
    """Class for visualizing a BVH structure using Polyscope"""

    def __init__(self, bvh: gq.BVH, vertices, indices):
        """Initialize the visualizer with a BVH and input geometry

        Args:
            bvh: The BVH structure to visualize
            vertices: Vertex positions for the input geometry
            indices: Edge indices for the input geometry
        """
        self.bvh = bvh
        self.vertices = vertices
        self.indices = indices
        self.nodes = bvh.nodes

        # Process the BVH structure
        self.node_levels, self.levels, self.max_level = self._compute_bvh_levels()
        self.current_level = 0

        # Compute statistics
        self.stats = self._compute_statistics()

        # Initialize Polyscope
        ps.init()
        ps.set_ground_plane_mode("none")

    def visualize(self):
        """Start the visualization"""
        # Register the input geometry
        self._register_input_geometry()

        # Initial visualization
        self.visualize_level(0)

        # Register UI callback
        ps.set_user_callback(self._ui_callback)

        # Start the interactive view
        ps.show()

    def _register_input_geometry(self):
        """Register the input geometry with Polyscope"""
        # Register the input vertices
        ps_points = ps.register_point_cloud("input_vertices", self.vertices)
        ps_points.set_color((0.2, 0.5, 0.5))
        ps_points.set_radius(0.01)

        # Register the input line segments
        ps_lines = ps.register_curve_network(
            "input_segments", self.vertices, self.indices)
        ps_lines.set_color((0.0, 0.0, 0.0))
        ps_lines.set_radius(0.002)

    def _compute_bvh_levels(self):
        """Compute the levels of each node in the BVH"""
        node_levels = []
        level_map = {}

        def compute_node_levels(node_idx, level=0):
            """Recursively compute the level of each node in the BVH"""
            if node_idx < 0 or node_idx >= len(self.nodes):
                return

            node = self.nodes[node_idx]
            if node_idx not in level_map:
                level_map[node_idx] = level
                node_levels.append((node_idx, level))

            # If interior node, process children
            if node.n_primitives == 0:
                # Process left child (always next node in array)
                compute_node_levels(node_idx + 1, level + 1)

                # Process right child if valid
                second_child_idx = node_idx + node.second_child_offset
                if second_child_idx > 0 and second_child_idx < len(self.nodes):
                    compute_node_levels(second_child_idx, level + 1)

        # Start with the root node
        compute_node_levels(0)

        # Organize nodes by level
        levels = {}
        for node_idx, level in node_levels:
            if level not in levels:
                levels[level] = []
            levels[level].append(node_idx)

        max_level = max(levels.keys()) if levels else 0

        return node_levels, levels, max_level

    def _compute_statistics(self):
        """Compute various statistics about the BVH"""
        stats = {}

        # Count total nodes
        stats['total_nodes'] = len(self.nodes)

        # Count leaf and internal nodes
        leaf_nodes = [node for node in self.nodes if node.n_primitives > 0]
        stats['leaf_nodes'] = len(leaf_nodes)
        stats['internal_nodes'] = stats['total_nodes'] - stats['leaf_nodes']

        # Maximum depth
        stats['max_depth'] = self.max_level

        # Compute average leaf depth
        leaf_depths = []
        for node_idx, node in enumerate(self.nodes):
            if node.n_primitives > 0:  # Leaf node
                for level, nodes in self.levels.items():
                    if node_idx in nodes:
                        leaf_depths.append(level)
                        break

        stats['avg_leaf_depth'] = sum(
            leaf_depths) / len(leaf_depths) if leaf_depths else 0

        # Count total primitives and primitives per leaf
        stats['total_primitives'] = sum(
            node.n_primitives for node in self.nodes)

        primitives_per_leaf = [
            node.n_primitives for node in self.nodes if node.n_primitives > 0]
        stats['avg_primitives_per_leaf'] = sum(
            primitives_per_leaf) / len(primitives_per_leaf) if primitives_per_leaf else 0
        stats['max_primitives_per_leaf'] = max(
            primitives_per_leaf) if primitives_per_leaf else 0
        stats['min_primitives_per_leaf'] = min(
            primitives_per_leaf) if primitives_per_leaf else 0

        return stats

    def _display_statistics(self):
        """Display BVH statistics in ImGui"""
        if ps.imgui.TreeNode("BVH Statistics"):
            ps.imgui.Text(f"Total Nodes: {self.stats['total_nodes']}")
            ps.imgui.Text(f"Internal Nodes: {self.stats['internal_nodes']}")
            ps.imgui.Text(f"Leaf Nodes: {self.stats['leaf_nodes']}")
            ps.imgui.Text(f"Maximum Depth: {self.stats['max_depth']}")
            ps.imgui.Text(
                f"Average Leaf Depth: {self.stats['avg_leaf_depth']:.2f}")
            ps.imgui.Text(
                f"Total Primitives: {self.stats['total_primitives']}")
            ps.imgui.Text(f"Primitives per Leaf:")
            ps.imgui.Indent()
            ps.imgui.Text(
                f"Average: {self.stats['avg_primitives_per_leaf']:.2f}")
            ps.imgui.Text(f"Maximum: {self.stats['max_primitives_per_leaf']}")
            ps.imgui.Text(f"Minimum: {self.stats['min_primitives_per_leaf']}")
            ps.imgui.Unindent()
            ps.imgui.TreePop()

    def _ui_callback(self):
        """Handle UI interactions"""
        # Display BVH statistics
        self._display_statistics()

        # Level slider
        changed, new_level = ps.imgui.SliderInt(
            "BVH Level", self.current_level, 0, self.max_level)
        if changed:
            self.current_level = new_level
            self.visualize_level(self.current_level)

        # Visualization options
        ps.imgui.PushID("vis_buttons")
        if ps.imgui.Button("Show All Levels"):
            for i in range(self.max_level + 1):
                self.visualize_level(i)

        if ps.imgui.Button("Show Leaf Nodes"):
            self.visualize_leaf_nodes()

        if ps.imgui.Button("Validate Vertices in Leaf Nodes"):
            self.validate_vertices_in_leaf_nodes()

        if ps.imgui.Button("Clear All"):
            self.clear_visualizations()
        ps.imgui.PopID()

    def clear_visualizations(self):
        """Clear all visualization elements"""
        # Clear all possible BVH level networks
        for i in range(self.max_level + 1):
            if ps.has_curve_network(f"bvh_level_{i}"):
                ps.remove_curve_network(f"bvh_level_{i}")

        # Clear other possible visualizations
        for name in ["leaf_nodes", "colored_vertices", "problem_vertices"]:
            if ps.has_curve_network(name):
                ps.remove_curve_network(name)
            if ps.has_point_cloud(name):
                ps.remove_point_cloud(name)

    def create_colored_box_network(self, boxes_with_colors, name, radius=0.005):
        """Create a network of colored boxes

        Args:
            boxes_with_colors: List of (box_vertices, color) tuples
            name: Name for the network in Polyscope
            radius: Radius for the box edges

        Returns:
            The Polyscope network object
        """
        if not boxes_with_colors:
            return None

        all_vertices = []
        all_indices = []
        box_color_array = []

        # Process each box
        for box_vertices, color in boxes_with_colors:
            # Add box vertices
            start_idx = len(all_vertices)
            all_vertices.extend(box_vertices)

            # Add box edges
            for i in range(4):
                all_indices.append([start_idx + i, start_idx + (i + 1) % 4])

            # Add colors for each vertex of this box
            for _ in range(4):
                box_color_array.append(color)

        # Create the curve network
        if all_vertices:
            network = ps.register_curve_network(
                name,
                np.array(all_vertices),
                np.array(all_indices)
            )

            # Add colors as a quantity
            network.add_color_quantity(
                f"{name}_colors",
                np.array(box_color_array),
                enabled=True
            )
            network.set_radius(radius)

            return network

        return None

    def color_vertices_by_boxes(self, boxes_with_colors, name="colored_vertices", radius=0.012):
        """Color vertices based on which box they belong to

        Args:
            boxes_with_colors: List of ((min_point, max_point), color) tuples
            name: Name for the point cloud in Polyscope
            radius: Radius for the vertices

        Returns:
            The Polyscope point cloud object
        """
        # Initialize vertex colors (default: gray)
        vertex_colors = np.full((len(self.vertices), 3),
                                fill_value=[0.5, 0.5, 0.5])
        vertex_assigned = np.zeros(len(self.vertices), dtype=bool)

        # Assign colors based on containing box
        for (box_min, box_max), color in boxes_with_colors:
            for i, vertex in enumerate(self.vertices):
                if not vertex_assigned[i] and is_point_in_box(vertex, box_min, box_max):
                    vertex_colors[i] = color
                    vertex_assigned[i] = True

        # Create the point cloud
        point_cloud = ps.register_point_cloud(name, self.vertices)
        point_cloud.add_color_quantity(
            "box_membership", vertex_colors, enabled=True)
        point_cloud.set_radius(radius)

        return point_cloud

    def visualize_level(self, level):
        """Visualize all nodes at a specific level"""
        self.clear_visualizations()

        # No boxes to show if level is out of range
        if level < 0 or level > self.max_level:
            return

        # Generate distinct colors for each box at this level
        num_boxes = len(self.levels[level])
        box_colors = generate_distinct_colors(num_boxes)

        # Prepare boxes with their colors
        boxes_with_colors = []
        box_bounds_with_colors = []

        for box_idx, node_idx in enumerate(self.levels[level]):
            node = self.nodes[node_idx]
            p_min, p_max = node.box.p_min, node.box.p_max

            # Create box vertices
            box_vertices = create_box_vertices(p_min, p_max)
            color = box_colors[box_idx]

            boxes_with_colors.append((box_vertices, color))
            box_bounds_with_colors.append(((p_min, p_max), color))

        # Create the colored box network
        self.create_colored_box_network(
            boxes_with_colors, f"bvh_level_{level}")

        # Color vertices by containing box
        self.color_vertices_by_boxes(box_bounds_with_colors)

    def visualize_leaf_nodes(self):
        """Visualize all leaf nodes in the BVH"""
        self.clear_visualizations()

        # Collect all leaf nodes
        leaf_nodes = [node for node in self.nodes if node.n_primitives > 0]

        # Generate distinct colors for each leaf node
        num_leaf_nodes = len(leaf_nodes)
        leaf_colors = generate_distinct_colors(num_leaf_nodes)

        # Prepare boxes with their colors
        boxes_with_colors = []
        box_bounds_with_colors = []

        for box_idx, node in enumerate(leaf_nodes):
            p_min, p_max = node.box.p_min, node.box.p_max

            # Create box vertices
            box_vertices = create_box_vertices(p_min, p_max)
            color = leaf_colors[box_idx]

            boxes_with_colors.append((box_vertices, color))
            box_bounds_with_colors.append(((p_min, p_max), color))

        # Create the colored box network (slightly thicker lines for leaf nodes)
        self.create_colored_box_network(
            boxes_with_colors, "leaf_nodes", radius=0.007)

        # Color vertices by containing box
        self.color_vertices_by_boxes(box_bounds_with_colors)

    def validate_vertices_in_leaf_nodes(self):
        """Check if each vertex is contained in exactly one leaf node"""
        # Get all leaf nodes
        leaf_nodes = [node for node in self.nodes if node.n_primitives > 0]

        if not leaf_nodes:
            msg = "⚠️ No leaf nodes found in the BVH. Validation not possible."
            print(msg)
            ps.warning(msg)
            return

        # For each vertex, track which leaf nodes contain it
        vertex_to_nodes = [[] for _ in range(len(self.vertices))]

        # Record node indices for visualization and reporting
        leaf_node_indices = []

        for node_idx, node in enumerate(self.nodes):
            if node.n_primitives > 0:  # Leaf node
                leaf_node_indices.append(node_idx)
                box_min = node.box.p_min
                box_max = node.box.p_max

                for i, vertex in enumerate(self.vertices):
                    if is_point_in_box(vertex, box_min, box_max):
                        vertex_to_nodes[i].append(node_idx)

        # Compute counts for quick access
        vertex_counts = np.array([len(nodes) for nodes in vertex_to_nodes])

        # Check results
        vertices_in_no_node = np.where(vertex_counts == 0)[0]
        vertices_in_multiple_nodes = np.where(vertex_counts > 1)[0]

        # Display results
        if len(vertices_in_no_node) == 0 and len(vertices_in_multiple_nodes) == 0:
            msg = "✅ Validation successful: All vertices are in exactly one leaf node."
            print(msg)
            ps.info(msg)
            return

        # Error reporting
        error_msg = ["❌ Validation failed:"]

        # Report vertices in no node
        if len(vertices_in_no_node) > 0:
            examples = vertices_in_no_node[:5]
            error_msg.append(
                f"- {len(vertices_in_no_node)} vertices are not in any leaf node ({(len(vertices_in_no_node) / len(self.vertices)) * 100:.1f}% of total)")
            for i in examples:
                error_msg.append(f"  Vertex {i}: {self.vertices[i]}")
            if len(vertices_in_no_node) > 5:
                error_msg.append(
                    f"  ... and {len(vertices_in_no_node) - 5} more")

        # Report vertices in multiple nodes
        if len(vertices_in_multiple_nodes) > 0:
            examples = vertices_in_multiple_nodes[:5]
            error_msg.append(
                f"- {len(vertices_in_multiple_nodes)} vertices are in multiple leaf nodes ({(len(vertices_in_multiple_nodes) / len(self.vertices)) * 100:.1f}% of total)")

            # Add detailed information for a few examples
            for i in examples:
                containing_nodes = vertex_to_nodes[i]
                error_msg.append(
                    f"  Vertex {i}: {self.vertices[i]} is in {len(containing_nodes)} nodes:")
                # Show node details for each containing node
                # Limit to first 3 containing nodes
                for node_idx in containing_nodes[:3]:
                    node = self.nodes[node_idx]
                    error_msg.append(
                        f"    - Node {node_idx}: primitives={node.n_primitives}, box=[{node.box.p_min}, {node.box.p_max}]")
                if len(containing_nodes) > 3:
                    error_msg.append(
                        f"    - ... and {len(containing_nodes) - 3} more nodes")

            if len(vertices_in_multiple_nodes) > 5:
                error_msg.append(
                    f"  ... and {len(vertices_in_multiple_nodes) - 5} more vertices")

        print("\n".join(error_msg))
        ps.error("\n".join(error_msg))

        # Highlight problematic vertices
        if ps.has_point_cloud("problem_vertices"):
            ps.remove_point_cloud("problem_vertices")

        # Safe concatenation - handle empty arrays
        problem_indices = []
        if len(vertices_in_no_node) > 0:
            problem_indices.extend(vertices_in_no_node)
        if len(vertices_in_multiple_nodes) > 0:
            problem_indices.extend(vertices_in_multiple_nodes)

        if problem_indices:
            # Show problematic vertices in red
            problem_vertices = self.vertices[problem_indices]
            ps_problem = ps.register_point_cloud(
                "problem_vertices", problem_vertices)
            ps_problem.set_color((1.0, 0.0, 0.0))  # Red for problem vertices
            ps_problem.set_radius(0.015)  # Larger radius to highlight

            # Visualize problematic leaf nodes if there are vertices in multiple nodes
            if len(vertices_in_multiple_nodes) > 0:
                # Get unique leaf nodes that contain multiple vertices
                problematic_nodes = set()
                for i in vertices_in_multiple_nodes:
                    problematic_nodes.update(vertex_to_nodes[i])

                # Create boxes for problematic leaf nodes
                problem_boxes = []
                problem_box_colors = []

                for node_idx in problematic_nodes:
                    node = self.nodes[node_idx]
                    box_vertices = create_box_vertices(
                        node.box.p_min, node.box.p_max)
                    problem_boxes.append(box_vertices)
                    # Orange for problematic boxes
                    problem_box_colors.append((1.0, 0.5, 0.0))

                # Create visualization of problematic boxes
                if problem_boxes:
                    boxes_with_colors = list(
                        zip(problem_boxes, problem_box_colors))
                    self.create_colored_box_network(
                        boxes_with_colors,
                        "problem_leaf_nodes",
                        radius=0.01  # Thicker lines for better visibility
                    )

                    # Log the number of problematic nodes
                    print(
                        f"Visualizing {len(problematic_nodes)} problematic leaf nodes that contain multiple vertices.")


def is_point_in_box(point, box_min, box_max):
    """Check if a point is inside a bounding box"""
    return (box_min[0] <= point[0] <= box_max[0] and
            box_min[1] <= point[1] <= box_max[1])


def create_box_vertices(p_min, p_max):
    """Create vertices for a box from min/max points"""
    return np.array([
        [p_min[0], p_min[1]],
        [p_max[0], p_min[1]],
        [p_max[0], p_max[1]],
        [p_min[0], p_max[1]]
    ])


def generate_distinct_colors(n):
    """Generate n visually distinct colors using a HSV-based approach"""
    colors = []
    for i in range(n):
        # Use golden ratio to create well-distributed hues
        h = (i * 0.618033988749895) % 1.0
        # Fixed saturation and value for good visibility
        s = 0.7 + 0.3 * ((i % 3) / 2.0)  # Vary saturation slightly
        v = 0.85 + 0.15 * ((i % 2) / 1.0)  # Vary value slightly

        # Convert HSV to RGB
        h_i = int(h * 6)
        f = h * 6 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        if h_i == 0:
            r, g, b = v, t, p
        elif h_i == 1:
            r, g, b = q, v, p
        elif h_i == 2:
            r, g, b = p, v, t
        elif h_i == 3:
            r, g, b = p, q, v
        elif h_i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q

        colors.append((r, g, b))

    return colors


def create_random_line_segments(num_segments=20, seed=42):
    """Create random line segments for testing"""
    np.random.seed(seed)

    # Generate random points in 2D space
    points = np.random.rand(num_segments * 2, 2) * 2 - 1  # Range from -1 to 1

    # Create line segments from consecutive pairs of points
    vertices = np.array(points, dtype=np.float32)
    indices = np.array([[i*2, i*2+1]
                       for i in range(num_segments)], dtype=np.int32)

    return vertices, indices


def get_default_shape():
    """Return a default square shape for testing"""
    vertices = np.array([
        [-1, -1], [1, -1], [1, 1], [-1, 1]
    ], dtype=np.float32)
    indices = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0]
    ], dtype=np.int32)
    return vertices, indices


def load_obj_file(obj_path):
    """Load geometry from an OBJ file with error handling"""
    try:
        if not os.path.exists(obj_path):
            print(f"Error: OBJ file not found: {obj_path}")
            print("Available files in data directory:")
            data_dir = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "..", "data")
            if os.path.exists(data_dir):
                for file in os.listdir(data_dir):
                    if file.endswith(".obj"):
                        print(f"  - {file}")
            print("\nUsing default square shape instead.")
            return get_default_shape()

        vertices, indices = load_obj_2d(obj_path)

        # Check if we have valid geometry
        if len(vertices) == 0 or len(indices) == 0:
            print(f"Error: OBJ file contains no valid 2D geometry: {obj_path}")
            print("Using default square shape instead.")
            return get_default_shape()

        return vertices, indices

    except Exception as e:
        print(f"Error loading OBJ file: {e}")
        print("Using default square shape instead.")
        return get_default_shape()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a 2D BVH structure using Polyscope")
    parser.add_argument("--obj", type=str,
                        help="Path to 2D obj file (optional)", default="/Users/jarvis/Projects/gQuery/data/workpiece.obj")
    parser.add_argument("--random", action="store_true",
                        help="Use random line segments")
    parser.add_argument("--num_segments", type=int,
                        default=20, help="Number of random segments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Determine which geometry to use
    if args.obj:
        vertices, indices = load_obj_file(args.obj)
    elif args.random:
        vertices, indices = create_random_line_segments(
            args.num_segments, args.seed)
    else:
        vertices, indices = get_default_shape()

    # Create the BVH
    bvh = gq.BVH(vertices, indices)

    # Create the visualizer and run it
    visualizer = BVHVisualizer(bvh, vertices, indices)
    visualizer.visualize()


if __name__ == "__main__":
    main()
