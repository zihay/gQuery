from core.fwd import *
from shapes.bvh3d import BVH3D, SNCH3D
from shapes.primitive import Intersection3D
from shapes.triangle import Triangle


class Mesh:
    vertices: Array3
    indices: Array3i

    bvh: BVH3D
    snch: SNCH3D

    def init(self, vertices: Array3, indices: Array3i):
        self.vertices = vertices
        self.indices = indices

        # self.bvh = BVH3D(self.vertices, self.indices)
        # self.snch = SNCH3D(self.vertices, self.indices)

    @dr.syntax
    def intersect(self, p: Array3, v: Array3,
                  n: Array3 = Array3(0., 0., 0.),
                  on_boundary: Bool = Bool(False),
                  r_max: Float = Float(dr.inf)):
        return self.intersect_baseline(p, v, n, on_boundary, r_max)

    @dr.syntax
    def intersect_baseline(self, p: Array3, v: Array3,
                           n: Array3 = Array3(0., 0., 0.),
                           on_boundary: Bool = Bool(False),
                           r_max: Float = Float(dr.inf)):
        its = dr.zeros(Intersection3D)
        d_min = Float(r_max)
        idx = Int(-1)
        i = Int(0)
        while i < dr.width(self.indices):
            f = dr.gather(Array3i, self.indices, i)
            a = dr.gather(Array3, self.vertices, f.x)
            b = dr.gather(Array3, self.vertices, f.y)
            c = dr.gather(Array3, self.vertices, f.z)
            _its = Triangle(a, b, c, i).ray_intersect(p, v, r_max)
            if _its.valid & (_its.d < d_min):
                idx = i
                d_min = _its.d
            i += 1

        if idx != -1:
            f = dr.gather(Array3i, self.indices, idx)
            a = dr.gather(Array3, self.vertices, f.x)
            b = dr.gather(Array3, self.vertices, f.y)
            c = dr.gather(Array3, self.vertices, f.z)
            its = Triangle(a, b, c, idx).ray_intersect(p, v, r_max)
        return its

    @dr.syntax
    def intersect_bvh(self, p: Array3, v: Array3,
                      n: Array3 = Array3(0., 0., 0.),
                      on_boundary: Bool = Bool(False),
                      r_max: Float = Float(dr.inf)):
        pass
