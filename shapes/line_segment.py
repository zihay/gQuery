from dataclasses import dataclass
from core.fwd import *
from scenes.primitive import BoundaryType
from wost.math_utils import closest_point_line_segment
from wost.scene import ClosestPointRecord, Intersection


@dataclass
class LineSegment:
    a: Array2
    b: Array2
    index: Int  # index in the original array
    sorted_index: Int  # index in the sorted array
    type: Int = Int(-1)

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
                        on_boundary=Bool(True),
                        type=self.type)
        return its

    @dr.syntax
    def closest_point(self, p: Array2):
        pa = p - self.a
        ba = self.b - self.a
        t = dr.normalize(ba)
        n = Array2(t.y, -t.x)
        h = dr.clamp(dr.dot(pa, ba) / dr.dot(ba, ba), 0., 1.)
        i = self.a + ba * h
        d = dr.norm(i - p)
        return ClosestPointRecord(
            p=i,
            n=n,
            t=h,
            d=d,
            prim_id=self.index,
            type=self.type)

    @dr.syntax
    def bounding_point(self, x: Array2, e: Array2,
                       rho_max: Float = Float(1.), r_max: Float = Float(dr.inf)):
        c_rec = dr.zeros(ClosestPointRecord)
        R = Float(r_max)
        T = self.type
        a = self.a
        b = self.b
        tangent = dr.normalize(b - a)
        normal = Array2(tangent.y, -tangent.x)
        dir = dr.normalize(a - x)
        if dr.dot(dir, normal) > 0.:
            en = dr.dot(e, normal)
            et = dr.dot(e, tangent)
            d = dr.dot(a - x, normal)
            t1 = Float(-dr.inf)
            t2 = Float(dr.inf)
            pt = x + d * normal
            if T == BoundaryType.Dirichlet.value:
                if dr.abs(et) > 1e-4:
                    t1 = -d / et * (rho_max + en)
                    t2 = d / et * (rho_max - en)
            elif T == BoundaryType.Neumann.value:
                if dr.abs(en) > 1e-4:
                    t1 = d / en * (rho_max + et)
                    t2 = -d / en * (rho_max - et)
            ta = dr.dot(a - x, tangent)
            tb = dr.dot(b - x, tangent)
            tmin = dr.minimum(ta, tb)
            tmax = dr.maximum(ta, tb)
            if (t1 > tmin) & (t1 < tmax):
                dist = dr.norm(pt + t1 * tangent - x)
                if dist < R:
                    c_rec.valid = Bool(True)
                    c_rec.p = pt + t1 * tangent
                    c_rec.t = t1
                    c_rec.d = dist
                    R = dist
            if (t2 > tmin) & (t2 < tmax):
                dist = dr.norm(pt + t2 * tangent - x)
                if dist < R:
                    c_rec.valid = Bool(True)
                    c_rec.p = pt + t2 * tangent
                    c_rec.t = t2
                    c_rec.d = dist
                    R = dist
            if ((t1 > tmax) & (t2 > tmax)) | ((t1 < tmin) & (t2 < tmin)):
                dist = dr.norm(a - x)
                if dist < R:
                    c_rec.valid = Bool(True)
                    c_rec.p = a
                    c_rec.t = Float(0.)
                    c_rec.d = dist
                    R = dist
                dist = dr.norm(b - x)
                if dist < R:
                    c_rec.valid = Bool(True)
                    c_rec.p = b
                    c_rec.t = Float(1.)
                    c_rec.d = dist
                    R = dist
        else:
            dist = dr.norm(a - x)
            if dist < R:
                c_rec.valid = Bool(True)
                c_rec.p = a
                c_rec.t = Float(0.)
                c_rec.d = dist
                R = dist
            dist = dr.norm(b - x)
            if dist < R:
                c_rec.valid = Bool(True)
                c_rec.p = b
                c_rec.t = Float(1.)
                c_rec.d = dist
                R = dist
        c_rec.prim_id = self.sorted_index
        c_rec.type = self.type
        return c_rec

    @dr.syntax
    def star_radius_2(self, x: Array2, e: Array2, rho_max: Float = Float(1.), r_max: Float = Float(dr.inf)):
        R = Float(r_max)
        T = self.type
        a = self.a
        b = self.b
        tangent = dr.normalize(b - a)
        normal = Array2(tangent.y, -tangent.x)
        dir = dr.normalize(a - x)
        if dr.dot(dir, normal) > 0.:
            en = dr.dot(e, normal)
            et = dr.dot(e, tangent)
            d = dr.dot(a - x, normal)
            t1 = Float(-dr.inf)
            t2 = Float(dr.inf)
            pt = x + d * normal
            if T == BoundaryType.Dirichlet.value:
                if dr.abs(et) > 1e-4:
                    t1 = -d / et * (rho_max + en)
                    t2 = d / et * (rho_max - en)
            elif T == BoundaryType.Neumann.value:
                if dr.abs(en) > 1e-4:
                    t1 = d / en * (rho_max + et)
                    t2 = -d / en * (rho_max - et)
            ta = dr.dot(a - x, tangent)
            tb = dr.dot(b - x, tangent)
            tmin = dr.minimum(ta, tb)
            tmax = dr.maximum(ta, tb)
            if (t1 > tmin) & (t1 < tmax):
                dist = dr.norm(pt + t1 * tangent - x)
                R = dr.minimum(R, dist)
            if (t2 > tmin) & (t2 < tmax):
                dist = dr.norm(pt + t2 * tangent - x)
                R = dr.minimum(R, dist)
            if ((t1 > tmax) & (t2 > tmax)) | ((t1 < tmin) & (t2 < tmin)):
                R = dr.minimum(R, dr.norm(a - x))
                R = dr.minimum(R, dr.norm(b - x))
        else:
            R = dr.minimum(R, dr.norm(a - x))
            R = dr.minimum(R, dr.norm(b - x))
        return R
