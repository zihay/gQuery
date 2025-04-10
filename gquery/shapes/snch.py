from core.fwd import *
from core.math import in_range, project_to_plane
from gquery.shapes.line_segment import LineSegment, LineSegments
from gquery.shapes.primitive import ClosestSilhouettePointRecord
from gquery.shapes.silhouette_vertex import SilhouetteVertex, SilhouetteVertices
from shapes.bvh import BVHNode, BoundingBox, BoundingCone, TraversalStack


@dataclass
class SNCHNode(BVHNode):
    cone: BoundingCone
    silhouette_reference_offset: Int
    n_silhouette_references: Int

    @dr.syntax
    def overlaps(self, x: Array2, r_max: Float = Float(dr.inf)):
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
                            d = project_to_plane(view_cone_axis, e)
                            view_cone_half_angle = dr.atan2(d, s)
                            half_angle_sum = self.cone.half_angle + view_cone_half_angle
                            min_angle = d_axis_angle - half_angle_sum
                            max_angle = d_axis_angle + half_angle_sum
                            is_overlap = (half_angle_sum > (dr.pi / 2)) \
                                | in_range(dr.pi / 2., min_angle, max_angle)
        return is_overlap, d_min


class SNCHNodes:
    data: Float

    def __init__(self, data: Float):
        self.data = data

    def __getitem__(self, index):
        return SNCHNode(
            box=BoundingBox(
                p_min=Array2(dr.gather(Float, self.data, 13 * index + 0),
                             dr.gather(Float, self.data, 13 * index + 1)),
                p_max=Array2(dr.gather(Float, self.data, 13 * index + 2),
                             dr.gather(Float, self.data, 13 * index + 3))),
            cone=BoundingCone(
                axis=Array2(dr.gather(Float, self.data, 13 * index + 4),
                            dr.gather(Float, self.data, 13 * index + 5)),
                half_angle=Float(dr.gather(Float, self.data, 13 * index + 6)),
                radius=Float(dr.gather(Float, self.data, 13 * index + 7))),
            reference_offset=Int(dr.gather(Float, self.data, 13 * index + 8)),
            second_child_offset=Int(
                dr.gather(Float, self.data, 13 * index + 9)),
            n_references=Int(dr.gather(Float, self.data, 13 * index + 10)),
            silhouette_reference_offset=Int(
                dr.gather(Float, self.data, 13 * index + 11)),
            n_silhouette_references=Int(
                dr.gather(Float, self.data, 13 * index + 12))
        )


class SNCH:
    flat_tree: SNCHNodes
    primitives: LineSegments
    silhouettes: SilhouetteVertices

    def __init__(self, vertices: Array2, indices: Array2i, types: Int):
        import gquery.gquery_ext as gq
        self.c_snch = gq.SNCH(vertices.numpy().T, indices.numpy().T)
        self.flat_tree = SNCHNodes(Float(self.c_snch.node_data()))
        self.primitives = LineSegments(Float(self.c_snch.primitive_data()))
        self.silhouettes = SilhouetteVertices(
            Float(self.c_snch.silhouette_data()))

    @dr.syntax
    def closest_silhouette(self, x: Array2, r_max: Float = Float(dr.inf)):
        r_max = Float(r_max)
        c_rec = dr.zeros(ClosestSilhouettePointRecord)
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
