from core.fwd import *
from shapes.bvh import BVH
from shapes.line_segment import LineSegment
from shapes.primitive import Intersection
from shapes.snch import SNCH


class Polyline:
    vertices: Array2
    indices: Array2i

    bvh: BVH
    snch: SNCH

    def init(self, vertices: Array2, indices: Array2i):
        self.vertices = vertices
        self.indices = indices

        self.bvh = BVH(self.vertices, self.indices)
        self.snch = SNCH(self.vertices, self.indices)

    @dr.syntax
    def intersect(self, p: Array2, v: Array2,
                  n: Array2 = Array2(0., 0.),
                  on_boundary: Bool = Bool(False),
                  r_max: Float = Float(dr.inf)):
        return self.intersect_baseline(p, v, n, on_boundary, r_max)

    @dr.syntax
    def intersect_bvh(self, p: Array2, v: Array2,
                      n: Array2 = Array2(0., 0.),
                      on_boundary: Bool = Bool(False),
                      r_max: Float = Float(dr.inf)):
        pass

    @dr.syntax
    def intersect_baseline(self, p: Array2, v: Array2,
                           n: Array2 = Array2(0., 0.),
                           on_boundary: Bool = Bool(False),
                           r_max: Float = Float(dr.inf)):
        # brute force
        its = dr.zeros(Intersection)
        d_min = Float(r_max)
        idx = Int(-1)
        i = Int(0)
        while i < dr.width(self.indices):
            f = dr.gather(Array2, self.vertices, i)
            a = dr.gather(Array2, self.vertices, f.x)
            b = dr.gather(Array2, self.vertices, f.y)
            _its = LineSegment(a, b, i).ray_intersect(p, v, r_max)
            if _its.valid & (_its.d < d_min):
                idx = i
                d_min = _its.d
            i += 1

        if idx != -1:
            f = dr.gather(Array2, self.vertices, idx)
            a = dr.gather(Array2, self.vertices, f.x)
            b = dr.gather(Array2, self.vertices, f.y)
            its = LineSegment(a, b, idx).ray_intersect(p, v, r_max)
        return its