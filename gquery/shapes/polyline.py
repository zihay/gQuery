from gquery.core.fwd import *
from gquery.core.math import ray_intersection
from gquery.shapes.bvh import BVH
from gquery.shapes.line_segment import LineSegment
from gquery.shapes.primitive import ClosestPointRecord, Intersection
from gquery.shapes.snch import SNCH


class Polyline:
    _vertices: Float  # flattened array of vertices
    _indices: Int  # flattened array of indices

    bvh: BVH
    snch: SNCH

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

    @property
    def vertices(self):
        return dr.gather(Array2, self._vertices, dr.arange(Int, dr.width(self._vertices) // 2))

    @property
    def indices(self):
        return dr.gather(Array2i, self._indices, dr.arange(Int, dr.width(self._indices) // 2))

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
        f = dr.gather(Array2i, self._indices, idx)
        a = dr.gather(Array2, self._vertices, f.x)
        b = dr.gather(Array2, self._vertices, f.y)
        pa = p - a
        ba = b - a
        h = dr.clip(dr.dot(pa, ba) / dr.dot(ba, ba), 0., 1.)
        n = dr.normalize(Array2(ba.y, -ba.x))
        return ClosestPointRecord(
            valid=Bool(True),
            p=dr.lerp(a, b, h),
            n=n,
            t=h,
            d=d_min,
            prim_id=idx)
