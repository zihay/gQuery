import numpy as np
import pytest
import gquery.gquery_ext as gq
from gquery.shapes.bvh import BVHNode
from gquery.shapes.bvh3d import BVHNode3D
import drjit as dr
import os
from gquery.util.obj_loader import load_obj_2d, load_obj_3d

class TestBVH2D:
    def test_bvh_construction(self):
        """Test basic BVH construction with simple line segments using C++ BVH."""
        # Create a simple set of vertices and indices
        vertices_np = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ], dtype=np.float32)
        
        indices_np = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0]
        ], dtype=np.int32)
        
        # Construct BVH using the C++ implementation
        bvh = gq.BVH(vertices_np, indices_np)
        
        # Basic assertions
        assert bvh is not None
        assert len(bvh.nodes) > 0
        assert len(bvh.primitives) == 4
        
    def test_bvh_node_properties(self):
        """Test BVH node properties using C++ BVH."""
        # Create a simple set of vertices and indices
        vertices_np = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ], dtype=np.float32)
        
        indices_np = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0]
        ], dtype=np.int32)
        
        # Construct BVH using the C++ implementation
        bvh = gq.BVH(vertices_np, indices_np)
        
        # Get the root node
        root_node = bvh.nodes[0]
        
        # Check bounding box of the root node
        assert root_node.box is not None
        assert root_node.box.p_min is not None
        assert root_node.box.p_max is not None
        
        # The bounding box should encompass all vertices
        assert np.allclose(root_node.box.p_min, [0.0, 0.0], atol=1e-5)
        assert np.allclose(root_node.box.p_max, [1.0, 1.0], atol=1e-5)
        
        # Check node properties
        assert hasattr(root_node, 'n_primitives')
        assert hasattr(root_node, 'primitives_offset')
        assert hasattr(root_node, 'second_child_offset')
        assert hasattr(root_node, 'axis')
    
    def test_bvh_with_2d_obj_file(self):
        """Test BVH construction with a 2D shape loaded from OBJ file using C++ BVH."""
        # Path to the 2D OBJ file
        obj_path = os.path.join("data", "shape2d.obj")
        
        # Ensure the file exists
        assert os.path.exists(obj_path), f"File {obj_path} not found"
        
        # Load the 2D shape
        vertices_np, edges_np = load_obj_2d(obj_path)
        
        # Construct BVH using the C++ implementation
        bvh = gq.gquery_ext.BVH(vertices_np, edges_np)
        
        # Basic assertions
        assert bvh is not None
        assert len(bvh.nodes) > 0
        assert len(bvh.primitives) == len(edges_np)
        
        # Get the root node
        root_node = bvh.nodes[0]
        
        # Check that the bounding box is reasonable
        assert root_node.box.p_min[0] < root_node.box.p_max[0]
        assert root_node.box.p_min[1] < root_node.box.p_max[1]

class TestBVH3D:
    def test_bvh3d_construction(self):
        """Test basic BVH3D construction with simple triangles."""
        # Note: We're still using the Python BVH3D here since there seems to be no direct C++ binding for 3D BVH
        # Create a simple set of vertices and indices for a cube
        vertices = gq.Array3(np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0]   # 7
        ], dtype=np.float32))
        
        # Define triangles for the cube (12 triangles for 6 faces)
        indices = gq.Array3i(np.array([
            # Front face
            [0, 1, 2],
            [0, 2, 3],
            # Back face
            [4, 6, 5],
            [4, 7, 6],
            # Left face
            [0, 3, 7],
            [0, 7, 4],
            # Right face
            [1, 5, 6],
            [1, 6, 2],
            # Bottom face
            [0, 4, 5],
            [0, 5, 1],
            # Top face
            [3, 2, 6],
            [3, 6, 7]
        ], dtype=np.int32))
        
        # Create types array (all triangles are the same type in this test)
        types = gq.Int(np.zeros(12, dtype=np.int32))
        
        # Construct BVH3D
        bvh3d = gq.shapes.bvh3d.BVH3D(vertices, indices, types)
        
        # Basic assertions
        assert bvh3d is not None
        assert bvh3d.flat_tree is not None
        assert bvh3d.primitives is not None
        
        # Check primitives count
        assert len(bvh3d.primitives.a) == 12  # 12 triangles
        assert len(bvh3d.primitives.b) == 12
        assert len(bvh3d.primitives.c) == 12
    
    def test_bvh3d_closest_point(self):
        """Test BVH3D closest point functionality."""
        # Create a simple set of vertices and indices for a cube
        vertices = gq.Array3(np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0]   # 7
        ], dtype=np.float32))
        
        indices = gq.Array3i(np.array([
            # Front face
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=np.int32))  # Only using two triangles for simplicity
        
        types = gq.Int(np.zeros(2, dtype=np.int32))
        
        # Construct BVH3D
        bvh3d = gq.shapes.bvh3d.BVH3D(vertices, indices, types)
        
        # Test point exactly on a vertex
        query_point = gq.Array3(np.array([[0.0, 0.0, 0.0]]).T)
        result = bvh3d.closest_point(query_point)
        
        # The distance should be 0 as the point is exactly on a vertex
        assert np.isclose(dr.detach(result.d), 0.0, atol=1e-5)
        
        # Test point outside the cube
        query_point = gq.Array3(np.array([[2.0, 2.0, 2.0]]).T)
        result = bvh3d.closest_point(query_point)
        
        # The point should be on the triangle and the distance should be positive
        assert dr.detach(result.d) > 0.0
        
        # Check if node_visited is incremented
        assert dr.detach(bvh3d.node_visited) > 0
    
    def test_bvh3d_with_3d_obj_file(self):
        """Test BVH3D construction with a 3D model loaded from OBJ file."""
        # Path to the 3D OBJ file (using the smaller bunny model)
        obj_path = os.path.join("data", "bunny.obj")
        
        # Ensure the file exists
        assert os.path.exists(obj_path), f"File {obj_path} not found"
        
        # Load the 3D model
        vertices_np, faces_np = load_obj_3d(obj_path)
        
        # Convert faces to triangles if needed
        triangles = []
        for face in faces_np:
            if len(face) == 3:
                triangles.append(face)  # Already a triangle
            elif len(face) > 3:
                # Triangulate the face (simple fan triangulation)
                for i in range(1, len(face) - 1):
                    triangles.append([face[0], face[i], face[i+1]])
        
        triangles_np = np.array(triangles if triangles else faces_np)
        
        # Convert to gquery arrays for the Python BVH3D implementation
        vertices = gq.Array3(vertices_np.T)
        indices = gq.Array3i(triangles_np.T)
        types = gq.Int(np.zeros(len(triangles_np), dtype=np.int32))
        
        # Create BVH3D
        bvh3d = gq.shapes.bvh3d.BVH3D(vertices, indices, types)
        
        # Basic assertions
        assert bvh3d is not None
        assert bvh3d.flat_tree is not None
        assert bvh3d.primitives is not None
        
        # Check that primitives were created
        assert len(bvh3d.primitives.a) > 0
        assert len(bvh3d.primitives.b) > 0
        assert len(bvh3d.primitives.c) > 0
        
        # Check that the bounding box is reasonable
        p_min = dr.detach(bvh3d.flat_tree.box.p_min)
        p_max = dr.detach(bvh3d.flat_tree.box.p_max)
        
        # The bounding box should have a reasonable size in all dimensions
        assert p_min[0] < p_max[0]
        assert p_min[1] < p_max[1]
        assert p_min[2] < p_max[2]
        
        # Test closest point query
        query_point = gq.Array3(np.array([[0.0, 0.0, 0.0]]).T)
        result = bvh3d.closest_point(query_point)
        
        # We should get a valid result
        assert result is not None
        assert dr.detach(result.d) >= 0.0
        
if __name__ == "__main__":
    # Run tests manually
    test_2d = TestBVH2D()
    test_2d.test_bvh_construction()
    test_2d.test_bvh_node_properties()
    test_2d.test_bvh_with_2d_obj_file()
    
    test_3d = TestBVH3D()
    test_3d.test_bvh3d_construction()
    test_3d.test_bvh3d_closest_point()
    test_3d.test_bvh3d_with_3d_obj_file()
    
    print("All tests passed!")
