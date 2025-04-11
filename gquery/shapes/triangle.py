from dataclasses import dataclass
from gquery.core.fwd import *
from gquery.core.math import closest_point_triangle
from gquery.shapes.primitive import BoundarySamplingRecord3D, ClosestPointRecord3D, Intersection3D


@dataclass
class Triangle:
    a: Array3
    b: Array3
    c: Array3
    index: Int

    @dr.syntax
    def centroid(self):
        return (self.a + self.b + self.c) / 3.

    @dr.syntax
    def normal(self):
        return dr.normalize(dr.cross(self.b - self.a, self.c - self.a))

    @dr.syntax
    def surface_area(self):
        return dr.norm(dr.cross(self.b - self.a, self.c - self.a)) / 2.

    @dr.syntax
    def sample_point(self, sampler: PCG32):
        u1 = dr.sqrt(sampler.next_float32())
        u2 = sampler.next_float32()
        u = 1.0 - u1
        v = u2*u1
        w = 1.0 - u - v
        uv = Array2(u, v)
        p = self.a*u + self.b*v + self.c*w
        n = dr.cross(self.b - self.a, self.c - self.a)
        pdf = 2. / dr.norm(n)
        n = dr.normalize(n)
        return BoundarySamplingRecord3D(
            p=p, n=n, uv=uv, pdf=pdf,
            prim_id=self.sorted_index,
            type=self.type)

    @dr.syntax
    def sphere_intersect(self, x: Array3, R: Float):
        pt, uv, d = closest_point_triangle(x, self.a, self.b, self.c)
        return d <= R

    @dr.syntax
    def ray_intersect(self, x: Array3, d: Array3, r_max: Float):
        its = dr.zeros(Intersection3D)
        v1 = self.b - self.a
        v2 = self.c - self.a
        p = dr.cross(d, v2)
        det = dr.dot(v1, p)
        if dr.abs(det) > dr.epsilon(Float):
            inv_det = 1. / det
            s = x - self.a
            v = dr.dot(s, p) * inv_det
            if (v >= 0) & (v <= 1):
                q = dr.cross(s, v1)
                w = dr.dot(d, q) * inv_det
                if (w >= 0) & (v + w <= 1):
                    t = dr.dot(v2, q) * inv_det
                    if (t >= 0) & (t <= r_max):
                        its = Intersection3D(
                            valid=Bool(True),
                            p=self.a+v1*v+v2*w,
                            n=dr.normalize(dr.cross(v1, v2)),
                            uv=Array2(1. - v - w, v),
                            d=t,
                            prim_id=self.index,
                            on_boundary=Bool(True))
        return its

    @dr.syntax
    def closest_point(self, p: Array3):
        pt, uv, d = closest_point_triangle(p, self.a, self.b, self.c)
        return ClosestPointRecord3D(
            valid=Bool(True),
            p=pt, n=self.normal(), uv=uv, d=d,
            prim_id=self.index)

    @dr.syntax
    def is_inside_circle(self, c: Array3, r: Float):
        return (dr.norm(self.a - c) < r) & (dr.norm(self.b - c) < r) & (dr.norm(self.c - c) < r)


class Triangles:  # AoS
    data: Float

    def __init__(self, data: Float):
        self.data = data

    def __getitem__(self, index: Int):
        return Triangle(
            a=Array3(dr.gather(Float, self.data, 10 * index + 0),
                     dr.gather(Float, self.data, 10 * index + 1),
                     dr.gather(Float, self.data, 10 * index + 2)),
            b=Array3(dr.gather(Float, self.data, 10 * index + 3),
                     dr.gather(Float, self.data, 10 * index + 4),
                     dr.gather(Float, self.data, 10 * index + 5)),
            c=Array3(dr.gather(Float, self.data, 10 * index + 6),
                     dr.gather(Float, self.data, 10 * index + 7),
                     dr.gather(Float, self.data, 10 * index + 8)),
            index=Int(dr.gather(Float, self.data, 10 * index + 9)))
