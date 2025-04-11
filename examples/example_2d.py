import numpy as np
import drjit as dr
from gquery.core.fwd import Float, Array2, Array2i
from gquery.shapes.polyline import Polyline

# Create a 2D polyline (a simple pentagon)
vertices = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.5, 1.0],
    [0.5, 1.5],
    [-0.5, 1.0]
])
edges = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 0]
])

# Create a polyline
polyline = Polyline(Array2(vertices.T), Array2i(edges.T))

# Perform closest point query
# DrJit array for batch processing
query_point = Array2(np.array([[0.8, 0.8]]).T)

closest = polyline.closest_point_bvh(query_point)

# Extract results
print('📍 Closest Point Query:')
print(f'  Query point: {query_point}')
print(f'  Closest point: {closest.p}')
print(f'  Distance: {closest.d}')


# Perform ray intersection query
direction = Array2(np.array([[1.0, 1.0]]).T)
r_max = Float(dr.inf)
intersection = polyline.intersect_bvh(query_point, direction, r_max)

print('📌 Ray Intersection:')
print(f'  Origin: {query_point}')
print(f'  Direction: {direction}')
print(f'  Intersection Point: {intersection.p}')
print(f'  Intersection Distance: {intersection.d}')


# Perform silhouette query
query_point = Array2(np.array([[2., 0.5]]).T)
silhouette = polyline.closest_silhouette_snch(query_point)

print('📍 Silhouette Query:')
print(f'  Query point: {query_point}')
print(f'  Silhouette point: {silhouette.p}')
print(f'  Distance: {silhouette.d}')
