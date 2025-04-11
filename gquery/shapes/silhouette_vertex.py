from dataclasses import dataclass
from gquery.core.fwd import *
from gquery.core.math import closest_point_line_segment, cross
from gquery.shapes.primitive import ClosestSilhouettePointRecord


@dataclass
class SilhouetteVertex:
    a: Array2
    b: Array2
    c: Array2
    indices: Array3i
    index: Int
    prim_id: Int = Int(-1)  # prim_id in the silhouette array

    def n0(self):
        t = self.b - self.a
        n = Array2(t[1], -t[0])
        return dr.normalize(n)

    def n1(self):
        t = self.c - self.b
        n = Array2(t[1], -t[0])
        return dr.normalize(n)

    @dr.syntax
    def is_silhouette_vertex(self, x: Array2):
        return cross(self.b - self.a, x - self.a) * cross(self.c - self.b, x - self.b) < 0.

    @dr.syntax
    def star_radius(self, p: Array2, r_min: Float = dr.inf):
        d_min = r_min
        d = dr.norm(p - self.b)
        if d < r_min:
            if self.is_silhouette_vertex(p):
                d_min = d
        return d_min

    @dr.syntax
    def closest_silhouette(self, x: Array2, r_max: Float = Float(dr.inf)):
        c_rec = dr.zeros(ClosestSilhouettePointRecord)

        p = self.b
        view_dir = x - p
        d = dr.norm(view_dir)
        if (self.indices[0] != -1) & (self.indices[2] != -1):
            if d < r_max:
                if self.is_silhouette_vertex(x):
                    c_rec.valid = Bool(True)
                    c_rec.p = p
                    c_rec.d = d
                    c_rec.prim_id = self.prim_id
        return c_rec


class SilhouetteVertices:
    data: Float

    def __init__(self, data: Float):
        self.data = data

    @staticmethod
    def from_soa(silhouettes: SilhouetteVertex):
        index = dr.arange(Int, dr.width(silhouettes))
        data = dr.zeros(Float, 10 * dr.width(silhouettes))
        dr.scatter(data, silhouettes.a.x, index * 10 + 0)
        dr.scatter(data, silhouettes.a.y, index * 10 + 1)
        dr.scatter(data, silhouettes.b.x, index * 10 + 2)
        dr.scatter(data, silhouettes.b.y, index * 10 + 3)
        dr.scatter(data, silhouettes.c.x, index * 10 + 4)
        dr.scatter(data, silhouettes.c.y, index * 10 + 5)
        dr.scatter(data, Float(silhouettes.indices.x), index * 10 + 6)
        dr.scatter(data, Float(silhouettes.indices.y), index * 10 + 7)
        dr.scatter(data, Float(silhouettes.indices.z), index * 10 + 8)
        dr.scatter(data, Float(silhouettes.index), index * 10 + 9)
        return SilhouetteVertices(data)

    def __getitem__(self, index: Int):
        return SilhouetteVertex(
            a=Array2(dr.gather(Float, self.data, 10 * index + 0),
                     dr.gather(Float, self.data, 10 * index + 1)),
            b=Array2(dr.gather(Float, self.data, 10 * index + 2),
                     dr.gather(Float, self.data, 10 * index + 3)),
            c=Array2(dr.gather(Float, self.data, 10 * index + 4),
                     dr.gather(Float, self.data, 10 * index + 5)),
            indices=Array3i(Int(dr.gather(Float, self.data, 10 * index + 6)),
                            Int(dr.gather(Float, self.data, 10 * index + 7)),
                            Int(dr.gather(Float, self.data, 10 * index + 8))),
            index=Int(dr.gather(Float, self.data, 10 * index + 9)),
            prim_id=Int(dr.gather(Float, self.data, 10 * index + 9)))

    def size(self):
        return dr.width(self.data) // 10
