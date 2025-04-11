from gquery.core.fwd import *
from gquery.core.math import closest_point_triangle
from gquery.shapes.snch3d import SNCH3D
from gquery.shapes.bvh3d import BVH3D
from gquery.shapes.primitive import ClosestPointRecord3D, Intersection3D
from gquery.shapes.triangle import Triangle


class Mesh:
    vertices: Array3
    indices: Array3i

    bvh: BVH3D
    snch: SNCH3D

    def __init__(self, vertices: Array3, indices: Array3i):
        self.vertices = vertices
        self.indices = indices

        self.bvh = BVH3D(self.vertices, self.indices)
        # self.snch = SNCH3D(self.vertices, self.indices)

        self.configure()

    def configure(self):
        self.configure_normal()

    def configure_normal(self):
        # polulate normals
        a = dr.gather(Array3, self.vertices, self.indices.x)
        b = dr.gather(Array3, self.vertices, self.indices.y)
        c = dr.gather(Array3, self.vertices, self.indices.z)
        u = b - a
        v = c - a
        self.face_normals = dr.normalize(dr.cross(u, v))
        self.normals = dr.zeros(Array3, shape=dr.width(self.vertices))
        dr.scatter_reduce(dr.ReduceOp.Add, self.normals,
                          self.face_normals, self.indices.x)
        dr.scatter_reduce(dr.ReduceOp.Add, self.normals,
                          self.face_normals, self.indices.y)
        dr.scatter_reduce(dr.ReduceOp.Add, self.normals,
                          self.face_normals, self.indices.z)
        self.normals = dr.normalize(self.normals)

    @dr.syntax
    def intersect(self, p: Array3, v: Array3,
                  n: Array3 = Array3(0., 0., 0.),
                  on_boundary: Bool = Bool(False),
                  r_max: Float = Float(dr.inf)):
        p = dr.select(on_boundary, p - 1e-5 * n, p)
        return self.intersect_baseline(p, v, n, on_boundary, r_max)

    @dr.syntax
    def intersect_baseline(self, p: Array3, v: Array3,
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
                      r_max: Float = Float(dr.inf)):
        return self.bvh.intersect(p, v, r_max)

    @dr.syntax
    def closest_point_baseline(self, x: Array3) -> ClosestPointRecord3D:
        d_min = Float(dr.inf)
        idx = Int(-1)
        i = Int(0)
        while i < dr.width(self.indices):
            f = dr.gather(Array3i, self.indices, i)
            a = dr.gather(Array3, self.vertices, f.x)
            b = dr.gather(Array3, self.vertices, f.y)
            c = dr.gather(Array3, self.vertices, f.z)
            pt, uv, d = closest_point_triangle(x, a, b, c)
            if d < d_min:
                d_min = d
                idx = i
            i += 1
        f = dr.gather(Array3i, self.indices, idx)
        a = dr.gather(Array3, self.vertices, f.x)
        b = dr.gather(Array3, self.vertices, f.y)
        c = dr.gather(Array3, self.vertices, f.z)
        pt, uv, d = closest_point_triangle(x, a, b, c)
        fn = dr.gather(Array3, self.face_normals, idx)
        return ClosestPointRecord3D(valid=Bool(True),
                                    p=pt,
                                    n=fn,
                                    uv=uv,
                                    d=d_min,
                                    prim_id=idx)

    @dr.syntax
    def closest_point_bvh(self, x: Array3) -> ClosestPointRecord3D:
        return self.bvh.closest_point(x)
