import numpy as np
import drjit as dr
from gquery.core.fwd import Float, Array3, Array3i
from gquery.shapes.mesh import Mesh

# Create a 3D mesh (a simple tetrahedron)
vertices = np.array([
    [0.0, 0.0, 0.0],   # 0
    [1.0, 0.0, 0.0],   # 1
    [0.5, 1.0, 0.0],   # 2
    [0.5, 0.5, 1.0]    # 3
])

# Define faces for the tetrahedron (triangles)
faces = np.array([
    [0, 1, 2],  # bottom face
    [0, 1, 3],  # side face 1
    [1, 2, 3],  # side face 2
    [0, 2, 3]   # side face 3
])

# Create a mesh
mesh = Mesh(Array3(vertices.T), Array3i(faces.T))

# Perform closest point query
# DrJit array for batch processing
query_point = Array3(np.array([[0.6, 0.6, 0.6]]).T)

closest = mesh.closest_point_bvh(query_point)

# Extract results
print('📍 Closest Point Query:')
print(f'  Query point: {query_point}')
print(f'  Closest point: {closest.p}')
print(f'  Distance: {closest.d}')


# Perform ray intersection query
direction = Array3(np.array([[1.0, 1.0, 1.0]]).T)
r_max = Float(dr.inf)
intersection = mesh.intersect_bvh(query_point, direction, r_max)

print('\n📌 Ray Intersection:')
print(f'  Origin: {query_point}')
print(f'  Direction: {direction}')
print(f'  Intersection Point: {intersection.p}')
print(f'  Intersection Distance: {intersection.d}')


# Perform silhouette query
query_point = Array3(np.array([[1.5, 0.5, 0.5]]).T)
silhouette = mesh.closest_silhouette_snch(query_point)

print('\n📍 Silhouette Query:')
print(f'  Query point: {query_point}')
print(f'  Silhouette point: {silhouette.p}')
print(f'  Distance: {silhouette.d}')