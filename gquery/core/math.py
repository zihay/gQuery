from core.fwd import *


@dr.syntax
def in_range(x, a, b):
    return (x >= a) & (x <= b)


@dr.syntax
def cross(a: Array2, b: Array2) -> Float:
    return a[0] * b[1] - a[1] * b[0]


@dr.syntax
def closest_point_line_segment(x: Array2, a: Array2, b: Array2):
    u = b - a
    v = x - a
    t = dr.clamp(dr.dot(v, u) / dr.dot(u, u), 0., 1.)
    p = dr.lerp(a, b, t)
    d = dr.norm(p - x)
    return d, p, t


@dr.syntax
def project_to_plane(n: Array2, e: Array2):
    b = Array2(-n[1], n[0])
    r = dr.dot(e, dr.abs(b))
    return dr.abs(r)
