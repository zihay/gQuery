from gquery.core.fwd import *


@dataclass
class Primitive:
    pass


@dataclass
class Intersection:
    valid: Bool  # whether the intersection is valid
    p: Array2  # intersection point
    n: Array2  # normal at the intersection point
    t: Float  # coordinate on the primitive
    d: Float  # distance to the intersection point
    prim_id: Int  # id of the primitive
    on_boundary: Bool  # whether the intersection is on the boundary


@dataclass
class Intersection3D:
    valid: Bool  # whether the intersection is valid
    p: Array3  # intersection point
    n: Array3  # normal at the intersection point
    uv: Array2  # uv coordinates of the intersection point
    d: Float  # distance to the intersection point
    prim_id: Int  # id of the primitive
    on_boundary: Bool  # whether the intersection is on the boundary


@dataclass
class ClosestPointRecord:
    valid: Bool  # whether the closest point is valid
    p: Array2  # closest point on the primitive
    n: Array2  # normal at the closest point
    t: Float  # coordinate on the primitive
    d: Float = Float(dr.inf)  # distance to the closest point
    prim_id: Int = Int(-1)  # id of the primitive


@dataclass
class ClosestPointRecord3D:
    valid: Bool  # whether the closest point is valid
    p: Array3  # closest point on the primitive
    n: Array3  # normal at the closest point
    uv: Array2  # uv coordinates of the closest point
    d: Float = Float(dr.inf)  # distance to the closest point
    prim_id: Int = Int(-1)  # id of the primitive


@dataclass
class BoundarySamplingRecord:
    p: Array2  # sample point
    n: Array2  # normal at the sample point
    t: Float  # coordinate on the primitive
    pdf: Float  # pdf of the sample point
    prim_id: Int = Int(-1)  # id of the primitive
    pmf: Float = Float(1.)  # probability of the sample primitive


@dataclass
class BoundarySamplingRecord3D:
    p: Array3  # sample point
    n: Array3  # normal at the sample point
    uv: Array2  # uv coordinates of the sample point
    pdf: Float  # pdf of the sample point
    prim_id: Int = Int(-1)  # id of the primitive
    pmf: Float = Float(1.)  # probability of the sample primitive
