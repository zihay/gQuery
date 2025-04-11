import numpy as np
from examples.visualizer import add_rectangle, clear_rectangles
from gquery.core.fwd import *
from dataclasses import dataclass
from gquery.core.math import closest_point_line_segment, ray_intersection
from gquery.shapes.line_segment import LineSegment, LineSegments
from gquery.shapes.primitive import BoundarySamplingRecord, ClosestPointRecord, Intersection


@dataclass
class BoundingSphere:
    center: Array2
    radius: Float


@dataclass
class BoundingBox:
    p_min: Array2
    p_max: Array2

    def centroid(self):
        return (self.p_min + self.p_max) / 2

    def distance(self, p: Array2):
        u = self.p_min - p
        v = p - self.p_max
        d_min = dr.norm(dr.maximum(dr.maximum(u, v), 0.))
        d_max = dr.norm(dr.minimum(u, v))
        return d_min, d_max

    def overlaps(self, c, r_max):
        d_min, d_max = self.distance(c)
        return d_min <= r_max, d_min, d_max

    @dr.syntax
    def intersect(self, x: Array2, v: Array2, r_max: Float = Float(dr.inf)):
        t0 = (self.p_min - x) / v
        t1 = (self.p_max - x) / v
        t_near = dr.minimum(t0, t1)
        t_far = dr.maximum(t0, t1)
        hit = Bool(False)
        t_near_max = dr.maximum(dr.max(t_near), 0.)
        t_far_min = dr.minimum(dr.min(t_far), r_max)
        t_min = Float(dr.inf)
        t_max = Float(dr.inf)
        if t_near_max > t_far_min:
            hit = Bool(False)
        else:
            t_min = t_near_max
            t_max = t_far_min
            hit = Bool(True)
        return hit, t_min, t_max


@dataclass
class BoundingCone:
    axis: Array2
    half_angle: Float
    radius: Float

    @dr.syntax
    def is_valid(self):
        return (self.half_angle >= 0.)


@dataclass
class BVHNode:
    box: BoundingBox
    reference_offset: Int
    second_child_offset: Int
    n_references: Int

    @dr.syntax
    def is_leaf(self):
        return self.n_references > 0


class BVHNodes:
    # flattened array of BVH nodes
    data: Float

    def __init__(self, data: Float):
        self.data = data

        index = dr.arange(Int, dr.width(self.data) // 7)
        self.SoA = BVHNode(
            box=BoundingBox(p_min=Array2(dr.gather(Float, self.data, 7 * index + 0),
                                         dr.gather(Float, self.data, 7 * index + 1)),
                            p_max=Array2(dr.gather(Float, self.data, 7 * index + 2),
                                         dr.gather(Float, self.data, 7 * index + 3))),
            reference_offset=Int(dr.gather(Float, self.data, 7 * index + 4)),
            second_child_offset=Int(
                dr.gather(Float, self.data, 7 * index + 5)),
            n_references=Int(dr.gather(Float, self.data, 7 * index + 6)))

    def __getitem__(self, index):
        # return dr.gather(BVHNode, self.SoA, index)
        return BVHNode(
            box=BoundingBox(p_min=Array2(dr.gather(Float, self.data, 7 * index + 0),
                                         dr.gather(Float, self.data, 7 * index + 1)),
                            p_max=Array2(dr.gather(Float, self.data, 7 * index + 2),
                                         dr.gather(Float, self.data, 7 * index + 3))),
            reference_offset=Int(dr.gather(Float, self.data, 7 * index + 4)),
            second_child_offset=Int(
                dr.gather(Float, self.data, 7 * index + 5)),
            n_references=Int(dr.gather(Float, self.data, 7 * index + 6)))

    def size(self):
        return dr.width(self.data) // 7


@dataclass
class TraversalStack:
    index: Int
    distance: Float


# @dataclass
class BVH:
    flat_tree: BVHNodes
    primitives: LineSegments  # ordered primitives

    def __init__(self, vertices: Array2, indices: Array2i):
        import gquery.gquery_ext as gq
        self.c_bvh = gq.BVH(vertices.numpy().T, indices.numpy().T)
        self.primitives = LineSegments(Float(self.c_bvh.primitive_data()))
        self.flat_tree = BVHNodes(Float(self.c_bvh.node_data()))

    @dr.syntax
    def closest_point(self, p: Array2):
        r_max = Float(dr.inf)

        stack_ptr = dr.zeros(Int, dr.width(p))
        stack = dr.alloc_local(TraversalStack, 64)
        stack[0] = TraversalStack(index=Int(0), distance=Float(r_max))

        idx = Int(-1)

        while stack_ptr >= 0:
            # pop off the next node to work on
            stack_node = stack[UInt(stack_ptr)]
            node_index = stack_node.index
            current_distance = stack_node.distance
            stack_ptr -= 1
            if current_distance <= r_max:
                node = self.flat_tree[node_index]
                if node.n_references > 0:
                    #     # leaf node
                    j = Int(0)
                    while j < node.n_references:
                        reference_index = node.reference_offset + j
                        prim = self.primitives[reference_index]
                        _d, _p, _t = closest_point_line_segment(
                            p, prim.a, prim.b)
                        if _d < r_max:
                            r_max = _d
                            idx = reference_index
                        j += 1
                else:
                    # non-leaf node
                    node_left = self.flat_tree[node_index + 1]
                    node_right = self.flat_tree[node_index +
                                                node.second_child_offset]
                    hit_left, d_min_left, d_max_left = node_left.box.overlaps(
                        p, r_max)
                    r_max = dr.minimum(r_max, d_max_left)
                    hit_right, d_min_right, d_max_right = node_right.box.overlaps(
                        p, r_max)
                    r_max = dr.minimum(r_max, d_max_right)

                    if hit_left & hit_right:
                        closer = node_index + 1
                        other = node_index + node.second_child_offset
                        if (d_min_left == 0.) & (d_min_right == 0.):
                            if d_max_right < d_max_right:
                                closer, other = other, closer
                        if d_min_right < d_min_left:
                            closer, other = other, closer
                            d_min_left, d_min_right = d_min_right, d_min_left

                        stack_ptr += 1
                        stack[UInt(stack_ptr)] = TraversalStack(index=other,
                                                                distance=d_min_right)
                        stack_ptr += 1
                        stack[UInt(stack_ptr)] = TraversalStack(index=closer,
                                                                distance=d_min_left)
                    elif hit_left:
                        stack_ptr += 1
                        stack[UInt(stack_ptr)] = TraversalStack(index=node_index + 1,
                                                                distance=d_min_left)
                    elif hit_right:
                        stack_ptr += 1
                        stack[UInt(stack_ptr)] = TraversalStack(index=node_index + node.second_child_offset,
                                                                distance=d_min_right)

        c_rec = dr.zeros(ClosestPointRecord)
        if idx != -1:
            primitive = self.primitives[idx]
            c_rec = primitive.closest_point(p)
        return c_rec

    @dr.syntax
    def intersect(self, x: Array2, v: Array2, r_max: Float = Float(dr.inf)):
        its = dr.zeros(Intersection)
        root_node = self.flat_tree[0]
        hit, t_min, t_max = root_node.box.intersect(x, v, r_max)

        d_min = Float(r_max)
        idx = Int(-1)
        is_hit = Bool(False)

        if hit:
            stack = dr.alloc_local(TraversalStack, 64)
            stack_ptr = dr.zeros(Int, dr.width(x))
            stack[0] = TraversalStack(index=Int(0),
                                      distance=Float(t_min))
            while stack_ptr >= 0:
                stack_node = stack[UInt(stack_ptr)]
                node_index = stack_node.index
                curr_dist = stack_node.distance
                stack_ptr -= 1

                if curr_dist <= r_max:
                    node = self.flat_tree[node_index]
                    if node.is_leaf():
                        j = Int(0)
                        while j < node.n_references:
                            reference_index = node.reference_offset + j
                            prim = self.primitives[reference_index]
                            d = ray_intersection(x, v, prim.a, prim.b)
                            if d < d_min:
                                d_min = d
                                idx = reference_index
                                is_hit = Bool(True)
                            j += 1
                    else:
                        left_box = self.flat_tree[node_index + 1].box
                        right_box = self.flat_tree[node_index +
                                                   node.second_child_offset].box

                        hit0, t_min0, t_max0 = left_box.intersect(x, v, r_max)
                        hit1, t_min1, t_max1 = right_box.intersect(x, v, r_max)

                        if hit0 & hit1:
                            closer = node_index + 1
                            other = node_index + node.second_child_offset

                            if t_min1 < t_min0:  # swap to make sure left is closer
                                closer, other = other, closer
                                t_min0, t_min1 = t_min1, t_min0
                                t_max0, t_max1 = t_max1, t_max0

                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=other, distance=t_min1)
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=closer, distance=t_min0)
                        elif hit0:
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=node_index + 1, distance=t_min0)
                        elif hit1:
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=node_index + node.second_child_offset,
                                distance=t_min1)

        if is_hit:
            its.valid = Bool(True)
            its.p = x + v * d_min
            its.n = self.primitives[idx].normal()
            its.t = Float(-1.)
            its.d = d_min
            its.prim_id = self.primitives[idx].index
            its.on_boundary = Bool(True)

        return its
