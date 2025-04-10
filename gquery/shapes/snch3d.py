from dataclasses import dataclass
from gquery.core.fwd import *
from gquery.core.math import in_range
from gquery.shapes.bvh import TraversalStack
from gquery.shapes.bvh3d import BVHNode3D, BoundingBox3D
from gquery.shapes.primitive import ClosestSilhouettePointRecord3D
from gquery.shapes.silhouette_edge import SilhouetteEdge, SilhouetteEdges
from gquery.shapes.triangle import Triangles


@dr.syntax
def compute_orthonomal_basis(n):
    sign = dr.copysign(1., n[2])
    a = -1. / (sign + n[2])
    b = n[0] * n[1] * a
    b1 = Array3(1. + sign * n[0] * n[0] * a,
                sign * b,
                -sign * n[0])
    b2 = Array3(b,
                sign + n[1] * n[1] * a,
                -n[1])
    return b1, b2


@dr.syntax
def project_to_plane_3D(n, e):
    b1, b2 = compute_orthonomal_basis(n)
    r1 = dr.dot(e, dr.abs(b1))
    r2 = dr.dot(e, dr.abs(b2))
    return dr.sqrt(r1 * r1 + r2 * r2)


@dataclass
class BoundingCone3D:
    axis: Array3
    half_angle: Float
    radius: Float

    @dr.syntax
    def is_valid(self):
        return (self.half_angle >= 0.)


@dataclass
class SNCHNode3D(BVHNode3D):
    cone: BoundingCone3D
    silhouette_reference_offset: int
    n_silhouette_references: int

    @dr.syntax
    def overlaps(self, x: Array3, r_max: Float = Float(dr.inf)):
        min_angle = Float(0.)
        max_angle = Float(dr.pi / 2)
        is_overlap = Bool(False)
        d_min = Float(dr.inf)
        is_overlap, d_min, d_max = self.box.overlaps(x, r_max)
        if self.cone.is_valid() & is_overlap:
            # prune if the box is not hit
            is_overlap = (self.cone.half_angle > (dr.pi / 2)) \
                & (d_min < dr.epsilon(Float))
            if ~is_overlap:
                c = self.box.centroid()
                view_cone_axis = c - x
                l = dr.norm(view_cone_axis)
                view_cone_axis = view_cone_axis / l
                d_axis_angle = dr.acos(
                    dr.clamp(dr.dot(self.cone.axis, view_cone_axis), -1., 1.))
                is_overlap = in_range(dr.pi / 2.,
                                      d_axis_angle - self.cone.half_angle,
                                      d_axis_angle + self.cone.half_angle)
                if ~is_overlap:
                    if l > self.cone.radius:
                        # NOTE: most of the pruning is done here
                        # outside the bounding sphere
                        view_cone_half_angle = dr.asin(
                            self.cone.radius / l)
                        half_angle_sum = self.cone.half_angle + view_cone_half_angle
                        min_angle = d_axis_angle - half_angle_sum
                        max_angle = d_axis_angle + half_angle_sum
                        is_overlap = (half_angle_sum > (dr.pi / 2)) \
                            | in_range(dr.pi / 2., min_angle, max_angle)
                    else:
                        e = self.box.p_max - c
                        d = dr.dot(e, dr.abs(view_cone_axis))
                        s = l - d
                        is_overlap = s < 0.
                        if ~is_overlap:
                            d = project_to_plane_3D(view_cone_axis, e)
                            view_cone_half_angle = dr.atan2(d, s)
                            half_angle_sum = self.cone.half_angle + view_cone_half_angle
                            min_angle = d_axis_angle - half_angle_sum
                            max_angle = d_axis_angle + half_angle_sum
                            is_overlap = (half_angle_sum > (dr.pi / 2)) \
                                | in_range(dr.pi / 2., min_angle, max_angle)
        return is_overlap, d_min


class SNCHNode3DAoS:
    data: Float

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return SNCHNode3D(
            box=BoundingBox3D(
                p_min=Array3(dr.gather(Float, self.data, 16 * index + 0),
                             dr.gather(Float, self.data, 16 * index + 1),
                             dr.gather(Float, self.data, 16 * index + 2)),
                p_max=Array3(dr.gather(Float, self.data, 16 * index + 3),
                             dr.gather(Float, self.data, 16 * index + 4),
                             dr.gather(Float, self.data, 16 * index + 5))),
            cone=BoundingCone3D(
                axis=Array3(dr.gather(Float, self.data, 16 * index + 6),
                            dr.gather(Float, self.data, 16 * index + 7),
                            dr.gather(Float, self.data, 16 * index + 8)),
                half_angle=dr.gather(Float, self.data, 16 * index + 9),
                radius=dr.gather(Float, self.data, 16 * index + 10)),
            reference_offset=Int(dr.gather(Float, self.data, 16 * index + 11)),
            second_child_offset=Int(
                dr.gather(Float, self.data, 16 * index + 12)),
            n_references=Int(dr.gather(Float, self.data, 16 * index + 13)),
            silhouette_reference_offset=Int(
                dr.gather(Float, self.data, 16 * index + 14)),
            n_silhouette_references=Int(
                dr.gather(Float, self.data, 16 * index + 15)))


@dataclass
class SNCH3D:
    flat_tree: SNCHNode3D
    primitives: Triangles
    silhouettes: SilhouetteEdges

    def __init__(self, vertices: Array3, indices: Array3i):
        import gquery.gquery_ext as gq
        self.c_snch = gq.SNCH3D(vertices.numpy().T, indices.numpy().T)
        self.flat_tree = SNCHNode3DAoS(Float(self.c_snch.node_data()))
        self.primitives = Triangles(Float(self.c_snch.primitive_data()))
        self.silhouettes = SilhouetteEdges(
            Float(self.c_snch.silhouette_data()))

    @dr.syntax
    def closest_silhouette(self, x: Array3, r_max: Float = Float(dr.inf)):
        r_max = Float(r_max)
        c_rec = dr.zeros(ClosestSilhouettePointRecord3D)
        root_node = self.flat_tree[0]
        overlap, d_min, d_max = root_node.box.overlaps(x, r_max)
        if overlap:
            stack = dr.alloc_local(TraversalStack, 64)
            stack_ptr = dr.zeros(Int, dr.width(x))
            stack[0] = TraversalStack(index=Int(0), distance=d_min)
            while stack_ptr >= 0:
                stack_node = stack[UInt(stack_ptr)]
                node_index = stack_node.index
                curr_dist = stack_node.distance
                stack_ptr -= 1
                if curr_dist <= r_max:
                    # prune curr_dist > r_max
                    node = self.flat_tree[node_index]
                    if node.is_leaf():  # leaf node
                        j = Int(0)
                        while j < node.n_silhouette_references:
                            reference_index = node.silhouette_reference_offset + j
                            silhouette = self.silhouettes[reference_index]
                            _c_rec = silhouette.closest_silhouette(
                                x, r_max)
                            if _c_rec.valid & (_c_rec.d < r_max):
                                r_max = dr.minimum(r_max, _c_rec.d)
                                c_rec = _c_rec
                            j += 1

                    else:  # non-leaf node
                        left = self.flat_tree[node_index + 1]
                        right = self.flat_tree[node_index +
                                               node.second_child_offset]

                        hit0, d_min0 = left.overlaps(x, r_max)
                        hit1, d_min1 = right.overlaps(x, r_max)

                        if hit0 & hit1:
                            closer = node_index + 1
                            other = node_index + node.second_child_offset
                            if d_min1 < d_min0:
                                closer, other = other, closer
                                d_min0, d_min1 = d_min1, d_min0
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=other, distance=d_min1)
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=closer, distance=d_min0)
                        elif hit0:
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=node_index + 1, distance=d_min0)
                        elif hit1:
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=node_index + node.second_child_offset,
                                distance=d_min1)

        return c_rec

    @dr.syntax
    def star_radius(self, x: Array3, r_max: Float = Float(dr.inf)):
        c_rec = self.closest_silhouette(x, r_max)
        return c_rec.d
