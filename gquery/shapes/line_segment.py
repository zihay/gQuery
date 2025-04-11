from dataclasses import dataclass
from gquery.core.fwd import *
from gquery.core.math import closest_point_line_segment
from gquery.shapes.primitive import ClosestPointRecord, Intersection


@dataclass
class LineSegment:
    a: Array2
    b: Array2
    index: Int  # index in the original array

    @dr.syntax
    def normal(self):
        t = self.b - self.a
        return dr.normalize(Array2(t.y, -t.x))

    @dr.syntax
    def length(self):
        return dr.norm(self.b - self.a)

    @dr.syntax
    def surface_area(self):
        return dr.norm(self.b - self.a)

    @dr.syntax
    def sample_point(self, sampler: PCG32):
        t = sampler.next_float32()
        p = self.a + (self.b - self.a) * t
        pdf = 1. / dr.norm(self.b - self.a)
        return p, pdf, t

    @dr.syntax
    def sphere_intersect(self, x: Array2, R: Float):
        d, _, _ = closest_point_line_segment(x, self.a, self.b)
        return d <= R

    @dr.syntax
    def ray_intersect(self, x: Array2, d: Array2, r_max: Float):
        its = dr.zeros(Intersection)
        u = self.a - x
        v = self.b - self.a
        dv = d[0] * v[1] - d[1] * v[0]
        if dr.abs(dv) > dr.epsilon(Float):
            # not parallel
            ud = u[0] * d[1] - u[1] * d[0]
            s = ud / dv
            if (s >= 0.) & (s <= 1.):
                uv = u[0] * v[1] - u[1] * v[0]
                t = uv / dv

                if (t >= 0.) & (t <= r_max):
                    its = Intersection(
                        valid=Bool(True),
                        p=x + d * t,
                        n=self.normal(),
                        t=s,
                        d=dr.abs(t),
                        prim_id=self.index,
                        on_boundary=Bool(True))
        return its

    @dr.syntax
    def closest_point(self, p: Array2) -> ClosestPointRecord:
        pa = p - self.a
        ba = self.b - self.a
        t = dr.normalize(ba)
        n = Array2(t.y, -t.x)
        h = dr.clamp(dr.dot(pa, ba) / dr.dot(ba, ba), 0., 1.)
        i = self.a + ba * h
        d = dr.norm(i - p)
        return ClosestPointRecord(
            valid=Bool(True),
            p=i,
            n=n,
            t=h,
            d=d,
            prim_id=self.index)


class LineSegments:  # AoS
    data: Float

    def __init__(self, data: Float):
        self.data = data
        index = dr.arange(Int, dr.width(self.data) // 5)
        self.SoA = LineSegment(
            a=Array2(dr.gather(Float, self.data, 5 * index + 0),
                     dr.gather(Float, self.data, 5 * index + 1)),
            b=Array2(dr.gather(Float, self.data, 5 * index + 2),
                     dr.gather(Float, self.data, 5 * index + 3)),
            index=Int(dr.gather(Float, self.data, 5 * index + 4)))

    def __getitem__(self, index):
        # return dr.gather(LineSegment, self.SoA, index)
        return LineSegment(
            a=Array2(dr.gather(Float, self.data, 5 * index + 0),
                     dr.gather(Float, self.data, 5 * index + 1)),
            b=Array2(dr.gather(Float, self.data, 5 * index + 2),
                     dr.gather(Float, self.data, 5 * index + 3)),
            index=Int(dr.gather(Float, self.data, 5 * index + 4)))
