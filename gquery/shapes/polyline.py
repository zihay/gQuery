import numpy as np
from gquery.core.fwd import *
from gquery.core.math import ray_intersection
from gquery.shapes.bvh import BVH
from gquery.shapes.line_segment import LineSegment
from gquery.shapes.primitive import ClosestPointRecord, ClosestSilhouettePointRecord, Intersection
from gquery.shapes.silhouette_vertex import SilhouetteVertex, SilhouetteVertices
from gquery.shapes.snch import SNCH


class Polyline:
    _vertices: Float  # flattened array of vertices
    _indices: Int  # flattened array of indices

    bvh: BVH
    snch: SNCH

    silhouettes: SilhouetteVertices

    def __init__(self, vertices: Array2, indices: Array2i):
        self.bvh = BVH(vertices, indices)
        self.snch = SNCH(vertices, indices)

        self._vertices = vertices
        self._indices = indices

        self._vertices = dr.zeros(Float, 2 * dr.width(vertices))
        self._indices = dr.zeros(Int, 2 * dr.width(indices))

        dr.scatter(self._vertices, vertices,
                   dr.arange(Int, dr.width(vertices)))
        dr.scatter(self._indices, indices, dr.arange(Int, dr.width(indices)))

        self.configure_silhouette()

    @property
    def vertices(self):
        return dr.gather(Array2, self._vertices, dr.arange(Int, dr.width(self._vertices) // 2))

    @property
    def indices(self):
        return dr.gather(Array2i, self._indices, dr.arange(Int, dr.width(self._indices) // 2))

    def configure_silhouette(self):
        """
        Configure silhouette vertices for the polyline.

        A silhouette vertex is a vertex that connects exactly two line segments.
        This method builds data structures needed for silhouette detection and sampling.
        """
        # Convert DrJit arrays to Python lists for processing
        vertices = self.vertices.numpy().T
        indices = self.indices.numpy().T

        # Track vertex connections and neighboring vertices
        vertex_to_neighbors = {}

        # Process each line segment
        for edge_idx, edge in enumerate(indices):
            a, b = edge[0], edge[1]

            # Initialize each vertex in the dictionary if not already present
            if a not in vertex_to_neighbors:
                vertex_to_neighbors[a] = [-1, a, -1]  # [prev, self, next]
            if b not in vertex_to_neighbors:
                vertex_to_neighbors[b] = [-1, b, -1]  # [prev, self, next]

            # Update neighbor information
            vertex_to_neighbors[a][2] = b  # Set 'next' for vertex a
            vertex_to_neighbors[b][0] = a  # Set 'prev' for vertex b

        # Collect silhouette vertex data for vertices with exactly two connected edges
        silhouette_vertices_a = []  # Previous neighbor vertices
        silhouette_vertices_b = []  # Silhouette vertices themselves
        silhouette_vertices_c = []  # Next neighbor vertices
        silhouette_indices = []     # Vertex indices for each silhouette vertex
        index_list = []             # Original vertex indices

        for i, v in enumerate(vertex_to_neighbors):
            # Check if vertex has both previous and next neighbors (is a silhouette)
            if vertex_to_neighbors[v][0] != -1 and vertex_to_neighbors[v][2] != -1:
                prev_idx = vertex_to_neighbors[v][0]
                curr_idx = vertex_to_neighbors[v][1]
                next_idx = vertex_to_neighbors[v][2]

                # Get actual 2D coordinates of vertices
                p0 = vertices[prev_idx]
                p1 = vertices[curr_idx]
                p2 = vertices[next_idx]

                # Store all data for this silhouette vertex
                silhouette_vertices_a.append(p0)
                silhouette_vertices_b.append(p1)
                silhouette_vertices_c.append(p2)
                silhouette_indices.append([prev_idx, curr_idx, next_idx])
                index_list.append(i)

        # Create DrJit array for silhouette vertices if any exist
        if len(silhouette_vertices_a) > 0 and len(silhouette_vertices_b) > 0 and len(silhouette_vertices_c) > 0:
            silhouettes = SilhouetteVertex(
                a=Array2(np.array(silhouette_vertices_a).T),
                b=Array2(np.array(silhouette_vertices_b).T),
                c=Array2(np.array(silhouette_vertices_c).T),
                indices=Array3i(np.array(silhouette_indices).T),
                index=Int(np.array(index_list))
            )

            # Filter silhouettes where the two connected edges are nearly collinear
            # Calculate tangent directions for the two connected edges
            edge1 = dr.normalize(silhouettes.b -
                                 silhouettes.a)  # prev->current
            edge2 = dr.normalize(silhouettes.c -
                                 silhouettes.b)  # current->next

            # Compute how collinear the edges are (absolute dot product close to 1 means collinear)
            collinearity = dr.abs(dr.dot(edge1, edge2))

            # Filter out vertices where the edges are nearly collinear (keep only meaningful silhouettes)
            silhouettes = dr.gather(
                SilhouetteVertex, silhouettes, dr.compress(
                    collinearity < (1.0 - 1e-5)))
            self.silhouettes = SilhouetteVertices.from_soa(silhouettes)

    @dr.syntax
    def intersect(self, p: Array2, v: Array2,
                  n: Array2 = Array2(0., 0.),
                  on_boundary: Bool = Bool(False),
                  r_max: Float = Float(dr.inf)):
        p = dr.select(on_boundary, p - 1e-5 * n, p)
        return self.intersect_baseline(p, v, r_max)

    @dr.syntax
    def intersect_bvh(self, p: Array2, v: Array2,
                      r_max: Float = Float(dr.inf)):
        return self.bvh.intersect(p, v, r_max)

    @dr.syntax
    def intersect_baseline(self, p: Array2, v: Array2,
                           r_max: Float = Float(dr.inf)):
        its = dr.zeros(Intersection)
        d_min = Float(r_max)
        idx = Int(-1)
        is_hit = Bool(False)
        i = Int(0)
        while i < dr.width(self.indices):
            f = dr.gather(Array2i, self._indices, i)
            a = dr.gather(Array2, self._vertices, f.x)
            b = dr.gather(Array2, self._vertices, f.y)
            d = ray_intersection(p, v, a, b)
            if d < d_min:
                d_min = d
                idx = i
                is_hit = Bool(True)
            i += 1

        if is_hit:
            f = dr.gather(Array2i, self._indices, idx)
            a = dr.gather(Array2, self._vertices, f.x)
            b = dr.gather(Array2, self._vertices, f.y)
            ab = dr.normalize(b - a)
            n = Array2(ab[1], -ab[0])
            its.valid = Bool(True)
            its.p = p + v * d_min
            its.n = n
            its.t = Float(-1.)
            its.d = d_min
            its.prim_id = idx
            its.on_boundary = Bool(True)
        return its

    @dr.syntax
    def closest_point_bvh(self, p: Array2) -> ClosestPointRecord:
        return self.bvh.closest_point(p)

    @dr.syntax
    def closest_point_baseline(self, p: Array2) -> ClosestPointRecord:
        d_min = Float(dr.inf)
        idx = Int(-1)
        i = Int(0)
        while i < (dr.width(self._indices) // 2):
            f = dr.gather(Array2i, self._indices, i)
            a = dr.gather(Array2, self._vertices, f.x)
            b = dr.gather(Array2, self._vertices, f.y)
            pa = p - a
            ba = b - a
            h = dr.clamp(dr.dot(pa, ba) / dr.dot(ba, ba), 0., 1.)
            # distance to the current primitive
            d = dr.norm(pa - ba * h)
            if d < d_min:
                d_min = d
                idx = i
            i += 1
        c_rec = dr.zeros(ClosestPointRecord)
        if idx != -1:
            f = dr.gather(Array2i, self._indices, idx)
            a = dr.gather(Array2, self._vertices, f.x)
            b = dr.gather(Array2, self._vertices, f.y)
            pa = p - a
            ba = b - a
            h = dr.clip(dr.dot(pa, ba) / dr.dot(ba, ba), 0., 1.)
            n = dr.normalize(Array2(ba.y, -ba.x))
            c_rec = ClosestPointRecord(
                valid=Bool(True),
                p=dr.lerp(a, b, h),
                n=n,
                t=h,
                d=d_min,
                prim_id=idx)
        return c_rec

    @dr.syntax
    def closest_silhouette_baseline(self, x: Array2, r_max: Float = Float(dr.inf)):
        c_rec = dr.zeros(ClosestSilhouettePointRecord)
        d_min = Float(r_max)
        i = Int(0)
        while (i < self.silhouettes.size()):
            silhouette = self.silhouettes[i]
            _c_rec = silhouette.closest_silhouette(x, r_max)
            if _c_rec.valid & (_c_rec.d < d_min):
                d_min = _c_rec.d
                c_rec = _c_rec
            i += 1
        return c_rec

    @dr.syntax
    def closest_silhouette_snch(self, x: Array2, r_max: Float = Float(dr.inf)):
        return self.snch.closest_silhouette(x, r_max)
