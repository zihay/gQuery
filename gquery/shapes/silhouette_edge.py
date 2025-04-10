from dataclasses import dataclass
from gquery.core.fwd import *
from gquery.core.math import closest_point_line_segment
from gquery.shapes.primitive import ClosestSilhouettePointRecord3D


def is_silhouette_edge(x: Array3, a: Array3, b: Array3, n0: Array3, n1: Array3) -> Bool:
    d, pt, t = closest_point_line_segment(x, a, b)
    view_dir = pt - x
    dot0 = dr.dot(view_dir, n0)
    dot1 = dr.dot(view_dir, n1)
    return dot0 * dot1 < 0.


@dataclass
class SilhouetteEdge:
    a: Array3  # first vertex of the edge
    b: Array3  # second vertex of the edge

    c: Array3  # first vertex of the triangle, vertex of abc
    d: Array3  # second vertex of the triangle, vertex of adb
    # indices: Array4i
    indices: Array4i  # vertex indices
    index: Int  # edge index
    face_indices: Array2i = Array2i(-1, -1)  # indices of the two faces
    prim_id: Int = Int(-1)  # prim_id in the new silhouette array

    def n0(self):
        return dr.normalize(dr.cross(self.b - self.a, self.c - self.a))

    def n1(self):
        return dr.normalize(dr.cross(self.d - self.a, self.b - self.a))

    @dr.syntax
    def star_radius(self, p: Array3, r_max: Float = Float(dr.inf)):
        d_min = Float(r_max)
        d, pt, t = closest_point_line_segment(p, self.a, self.b)
        if d < r_max:
            _is_silhouette = (self.indices[0] == -1) | (self.indices[3] == -1)
            if ~_is_silhouette:
                _is_silhouette = is_silhouette_edge(p, self.a, self.b,
                                                    self.n0(), self.n1())
            if _is_silhouette:
                d_min = d
        return d_min

    @dr.syntax
    def closest_silhouette(self, x: Array3, r_max: Float = Float(dr.inf)):
        c_rec = dr.zeros(ClosestSilhouettePointRecord3D)
        d, pt, t = closest_point_line_segment(x, self.a, self.b)
        if d < r_max:
            _is_silhouette = (self.indices[0] == -1) | (self.indices[3] == -1)
            if ~_is_silhouette:
                _is_silhouette = is_silhouette_edge(x, self.a, self.b,
                                                    self.n0(), self.n1())
            if _is_silhouette:
                c_rec = ClosestSilhouettePointRecord3D(
                    valid=Bool(True),
                    p=pt,
                    d=d,
                    prim_id=self.prim_id)
        return c_rec


class SilhouetteEdges:
    data: Float

    def __init__(self, data: Float):
        self.data = data

    def __getitem__(self, index: Int):
        return SilhouetteEdge(
            a=Array3(dr.gather(Float, self.data, 19 * index + 0),
                     dr.gather(Float, self.data, 19 * index + 1),
                     dr.gather(Float, self.data, 19 * index + 2)),
            b=Array3(dr.gather(Float, self.data, 19 * index + 3),
                     dr.gather(Float, self.data, 19 * index + 4),
                     dr.gather(Float, self.data, 19 * index + 5)),
            c=Array3(dr.gather(Float, self.data, 19 * index + 6),
                     dr.gather(Float, self.data, 19 * index + 7),
                     dr.gather(Float, self.data, 19 * index + 8)),
            d=Array3(dr.gather(Float, self.data, 19 * index + 9),
                     dr.gather(Float, self.data, 19 * index + 10),
                     dr.gather(Float, self.data, 19 * index + 11)),
            indices=Array4i(Int(dr.gather(Float, self.data, 19 * index + 12)),
                            Int(dr.gather(Float, self.data, 19 * index + 13)),
                            Int(dr.gather(Float, self.data, 19 * index + 14)),
                            Int(dr.gather(Float, self.data, 19 * index + 15))),
            face_indices=Array2i(Int(dr.gather(Float, self.data, 19 * index + 16)),
                                 Int(dr.gather(Float, self.data, 19 * index + 17))),
            index=Int(dr.gather(Float, self.data, 19 * index + 18)),
            prim_id=Int(dr.gather(Float, self.data, 19 * index + 18)))
