import numpy as np


def load_obj_2d(file_path, normalize=False):
    """
    Load a 2D shape from an OBJ file.

    Args:
        file_path: Path to the OBJ file

    Returns:
        vertices: numpy array of shape (N, 2) containing vertex coordinates
        edges: numpy array of shape (M, 2) containing edge indices
    """
    vertices = []
    edges = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Parse vertex (using first 2 coordinates)
                parts = line.split()
                # Get x and y coordinates
                x, y = float(parts[1]), float(parts[2])
                vertices.append((x, y))
            elif line.startswith('f '):
                # Parse face (assuming triangle or quad)
                parts = line.split()
                # OBJ indices start from 1, so subtract 1
                indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]

                # Create edges for the face boundary
                for i in range(len(indices)):
                    edges.append((indices[i], indices[(i+1) % len(indices)]))
            elif line.startswith('l '):
                # Parse line segment
                parts = line.split()
                # OBJ indices start from 1, so subtract 1
                if len(parts) >= 3:  # Need at least 2 vertices for a line
                    indices = [int(p) - 1 for p in parts[1:]]

                    # Create edges for the line segments
                    for i in range(len(indices) - 1):
                        edges.append((indices[i], indices[i+1]))

    # Convert to numpy arrays
    vertices = np.array(vertices)
    edges = np.array(edges)

    # Normalize vertices
    if normalize and len(vertices) > 0:
        center = np.mean(vertices, axis=0)
        vertices -= center
        scale = np.max(np.abs(vertices)) if np.max(
            np.abs(vertices)) > 0 else 1.0
        vertices /= scale

    # Remove duplicate edges
    # if len(edges) > 0:
    #     edges = np.array(edges)
    #     edges = np.sort(edges, axis=1)  # Sort edge vertices
    #     edges = np.unique(edges, axis=0)  # Remove duplicates

    return vertices, edges


def load_obj_3d(file_path, normalize=False):
    """
    Load a 3D mesh from an OBJ file.

    Args:
        file_path: Path to the OBJ file
        normalize: Whether to normalize the vertices to be centered and scaled

    Returns:
        vertices: numpy array of shape (N, 3) containing vertex coordinates
        faces: numpy array of shape (M, 3+) containing face indices
    """
    vertices = []
    faces = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Parse 3D vertex
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append((x, y, z))
            elif line.startswith('f '):
                # Parse face - can be triangle or quad
                parts = line.split()
                # Extract vertex indices (OBJ indices start from 1)
                indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(indices)

    # Convert to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Normalize the model
    if normalize and len(vertices) > 0:
        center = np.mean(vertices, axis=0)
        vertices -= center
        scale = np.max(np.abs(vertices)) if np.max(
            np.abs(vertices)) > 0 else 1.0
        vertices /= scale

    # # Remove duplicate face indices
    # if len(faces) > 0:
    #     faces = np.array(faces)
    #     faces = np.sort(faces, axis=1)  # Sort face vertices
    #     faces = np.unique(faces, axis=0)  # Remove duplicates

    return vertices, faces
