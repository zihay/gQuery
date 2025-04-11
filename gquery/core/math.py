from gquery.core.fwd import *


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


@dr.syntax
def ray_intersection(x: Array2, v: Array2, a: Array2, b: Array2) -> Float:
    u = b - a
    w = x - a
    d = cross(v, u)
    s = cross(v, w) / d
    t = cross(u, w) / d
    valid = (t > 0.) & (s >= 0.) & (s <= 1.)
    return dr.select(valid, t, dr.inf)


def closest_point_triangle(p: Array3, a: Array3, b: Array3, c: Array3):
    pt = Array3(0, 0, 0)
    uv = Array2(0, 0)
    d = dr.inf
    ab = b - a
    ac = c - a
    active = Bool(True)
    # check if p is in the vertex region outside a
    ax = p - a
    d1 = dr.dot(ab, ax)
    d2 = dr.dot(ac, ax)
    cond = (d1 <= 0) & (d2 <= 0)
    pt = dr.select(cond, a, pt)
    uv = dr.select(cond, Array2(1, 0), uv)
    d = dr.select(cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the vertex region outside b
    bx = p - b
    d3 = dr.dot(ab, bx)
    d4 = dr.dot(ac, bx)
    cond = (d3 >= 0) & (d4 <= d3)
    pt = dr.select(active & cond, b, pt)
    uv = dr.select(active & cond, Array2(0, 1), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the vertex region outside c
    cx = p - c
    d5 = dr.dot(ab, cx)
    d6 = dr.dot(ac, cx)
    cond = (d6 >= 0) & (d5 <= d6)
    pt = dr.select(active & cond, c, pt)
    uv = dr.select(active & cond, Array2(0, 0), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the edge region of ab, if so return projection of p onto ab
    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    cond = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    pt = dr.select(active & cond, a + ab * v, pt)
    uv = dr.select(active & cond, Array2(1 - v, v), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the edge region of ac, if so return projection of p onto ac
    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    cond = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    pt = dr.select(active & cond, a + ac * w, pt)
    uv = dr.select(active & cond, Array2(1 - w, 0), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the edge region of bc, if so return projection of p onto bc
    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    cond = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    pt = dr.select(active & cond, b + (c - b) * w, pt)
    uv = dr.select(active & cond, Array2(0, 1 - w), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is inside face region
    denom = 1. / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    pt = dr.select(active, a + ab * v + ac * w, pt)
    uv = dr.select(active, Array2(1 - v - w, v), uv)
    d = dr.select(active, dr.norm(p - pt), d)
    return pt, Array2(uv[1], 1. - uv[0] - uv[1]), d


@dr.syntax
def ray_triangle_intersect(x: Array3, d: Array3, a: Array3, b: Array3, c: Array3,
                           r_max: Float = Float(dr.inf)):
    v1 = b - a
    v2 = c - a
    p = dr.cross(d, v2)
    det = dr.dot(v1, p)
    if dr.abs(det) > dr.epsilon(Float):
        inv_det = 1. / det
        s = x - a
        v = dr.dot(s, p) * inv_det
        if (v >= 0) & (v <= 1):
            q = dr.cross(s, v1)
            w = dr.dot(d, q) * inv_det
            if (w >= 0) & (v + w <= 1):
                t = dr.dot(v2, q) * inv_det
                if (t >= 0) & (t <= r_max):
                    r_max = t
    return r_max