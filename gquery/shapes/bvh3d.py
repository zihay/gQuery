
from dataclasses import dataclass
from typing import Tuple
from gquery.core.fwd import *
from gquery.shapes.bvh import TraversalStack
from gquery.shapes.primitive import BoundarySamplingRecord3D, ClosestPointRecord3D, Intersection3D
from gquery.shapes.triangle import Triangle, Triangles


@dataclass
class BoundingBox3D:
    p_min: Array3
    p_max: Array3

    def centroid(self):
        return (self.p_min + self.p_max) / 2

    def distance(self, p: Array3):
        u = self.p_min - p
        v = p - self.p_max
        d_min = dr.norm(dr.maximum(dr.maximum(u, v), 0.))
        d_max = dr.norm(dr.minimum(u, v))
        return d_min, d_max

    def overlaps(self, c, r_max):
        d_min, d_max = self.distance(c)
        return d_min <= r_max, d_min, d_max

    @dr.syntax
    def intersect(self, x: Array3, v: Array3, r_max: Float = Float(dr.inf)):
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
class BVHNode3D:
    box: BoundingBox3D
    reference_offset: int
    second_child_offset: int
    n_references: int

    @dr.syntax
    def is_leaf(self):
        return self.n_references > 0


class BVHNode3DAoS:
    data: Float

    def __init__(self, data: Float):
        self.data = data

    def __getitem__(self, index: int):
        return BVHNode3D(
            box=BoundingBox3D(
                p_min=Array3(dr.gather(Float, self.data, 9 * index + 0),
                             dr.gather(Float, self.data, 9 * index + 1),
                             dr.gather(Float, self.data, 9 * index + 2)),
                p_max=Array3(dr.gather(Float, self.data, 9 * index + 3),
                             dr.gather(Float, self.data, 9 * index + 4),
                             dr.gather(Float, self.data, 9 * index + 5))),
            reference_offset=Int(dr.gather(Float, self.data, 9 * index + 6)),
            second_child_offset=Int(
                dr.gather(Float, self.data, 9 * index + 7)),
            n_references=Int(dr.gather(Float, self.data, 9 * index + 8))
        )


@dataclass
class BVH3D:
    flat_tree: BVHNode3DAoS
    primitives: Triangles

    def __init__(self, vertices: Array3, indices: Array3i):
        import gquery.gquery_ext as gq
        self.c_bvh = gq.BVH3D(vertices.numpy().T, indices.numpy().T)
        self.primitives = Triangles(Float(self.c_bvh.primitive_data()))
        self.flat_tree = BVHNode3DAoS(Float(self.c_bvh.node_data()))

    @dr.syntax
    def closest_point(self, p: Array3):
        p = Array3(p)
        r_max = Float(dr.inf)
        its = ClosestPointRecord3D(p=Array3(0, 0, 0),
                                   n=Array3(0, 0, 0),
                                   uv=Array2(0, 0),
                                   d=Float(dr.inf),
                                   prim_id=Int(-1))
        stack_ptr = dr.zeros(Int, dr.width(p))
        stack = dr.alloc_local(TraversalStack, 64)
        stack[0] = TraversalStack(index=Int(0), distance=Float(r_max))
        while stack_ptr >= 0:
            stack_node = stack[UInt(stack_ptr)]
            node_index = stack_node.index
            current_distance = stack_node.distance
            stack_ptr -= 1
            if current_distance <= r_max:
                node = self.flat_tree[node_index]
                if node.is_leaf():
                    # leaf node
                    j = Int(0)
                    while j < node.n_references:
                        reference_index = node.reference_offset + j
                        prim = self.primitives[reference_index]
                        _its = prim.closest_point(p)
                        if _its.d < its.d:
                            its = _its
                            r_max = dr.minimum(r_max, its.d)
                        j += 1
                else:
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
        return its

    @dr.syntax
    def intersect(self, x: Array3, v: Array3, r_max: Float = Float(dr.inf)):
        its = dr.zeros(Intersection3D)
        root_node = self.flat_tree[0]
        hit, t_min, t_max = root_node.box.intersect(x, v, r_max)
        if hit:
            stack = dr.alloc_local(TraversalStack, 64)
            stack_ptr = dr.zeros(Int, dr.width(x))
            stack[0] = TraversalStack(index=Int(0), distance=Float(t_min))
            while stack_ptr >= 0:
                stack_node = stack[UInt(stack_ptr)]
                node_index = stack_node.index
                curr_dist = stack_node.distance
                stack_ptr -= 1
                if curr_dist <= r_max:
                    # prune curr_dist > r_max
                    node = self.flat_tree[node_index]
                    if node.is_leaf():
                        j = Int(0)
                        while j < node.n_references:
                            reference_index = node.reference_offset + j
                            prim = self.primitives[reference_index]
                            _its = prim.ray_intersect(x, v, r_max)
                            if _its.valid & (_its.d < r_max):
                                r_max = its.d
                                its = _its
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
                                index=node_index + node.second_child_offset, distance=t_min1)
        return its

    @dr.syntax
    def branch_traversal_weight(self, x: Array3, R: Float,
                                p: Array3):
        return Float(1.)

    @dr.syntax
    def sample_boundary(self, x: Array3, R: Float, sampler: PCG32) -> Tuple[Bool, BoundarySamplingRecord3D]:
        b_rec = dr.zeros(BoundarySamplingRecord3D)
        hit = Bool(False)
        pdf = Float(1.)
        sorted_prim_id = Int(-1)
        root_node = self.flat_tree[0]
        overlap, d_min, d_max = root_node.box.overlaps(x, R)
        if overlap:
            hit, sorted_prim_id, pdf = self._sample_boundary(x, R, sampler)
        if hit:
            prim = self.primitives[sorted_prim_id]
            s_rec = prim.sample_point(sampler)
            b_rec = BoundarySamplingRecord3D(p=s_rec.p,
                                             n=s_rec.n,
                                             uv=s_rec.uv,
                                             pdf=pdf * s_rec.pdf,
                                             pmf=pdf,
                                             prim_id=prim.index,
                                             type=prim.type)
        return hit, b_rec

    @dr.syntax
    def _sample_boundary(self, x: Array3, R: Float,
                         sampler: PCG32, r_max: Float = Float(dr.inf)):
        # a single root to leaf traversal
        d_max0 = Float(dr.inf)
        d_max1 = Float(dr.inf)

        hit = Bool(False)
        pdf = Float(1.)
        prim_id = Int(-1)
        sorted_prim_id = Int(-1)

        stack_ptr = dr.zeros(Int, dr.width(x))
        node_index = Int(0)
        while stack_ptr >= 0:
            node = self.flat_tree[node_index]
            stack_ptr -= 1

            if node.is_leaf():
                # terminate at leaf node
                total_primitive_weight = Float(0.)
                # traverse all primitives in the leaf node
                j = Int(0)
                while j < node.n_references:
                    reference_index = node.reference_offset + j
                    prim = self.primitives[reference_index]
                    surface_area = prim.surface_area()
                    if (r_max <= R) | prim.sphere_intersect(x, R):  # all primitives are visible or hit
                        hit = Bool(True)
                        total_primitive_weight += surface_area
                        selection_prob = surface_area / total_primitive_weight
                        u = sampler.next_float32()
                        if u < selection_prob:  # select primitive
                            u = u / selection_prob  # rescale u to [0, 1]
                            prim_id = prim.index
                            sorted_prim_id = prim.sorted_index
                    j += 1
                if total_primitive_weight > 0:
                    prim = self.primitives[sorted_prim_id]
                    surface_area = prim.surface_area()
                    pdf *= surface_area / total_primitive_weight
            else:
                box0 = self.flat_tree[node_index + 1].box
                overlap0, d_min0, d_max0 = box0.overlaps(x, R)
                weight0 = dr.select(overlap0, 1., 0.)

                box1 = self.flat_tree[node_index +
                                      node.second_child_offset].box
                overlap1, d_min1, d_max1 = box1.overlaps(x, R)
                weight1 = dr.select(overlap1, 1., 0.)

                if weight0 > 0:
                    weight0 *= self.branch_traversal_weight(
                        x, R, box0.centroid())

                if weight1 > 0:
                    weight1 *= self.branch_traversal_weight(
                        x, R, box1.centroid())

                total_weight = weight0 + weight1
                if total_weight > 0:
                    stack_ptr += 1  # push a node
                    traversal_prob0 = weight0 / total_weight
                    traversal_prob1 = 1. - traversal_prob0
                    u = sampler.next_float32()
                    if u < traversal_prob0:
                        # choose left child
                        u = u / traversal_prob0  # rescale u to [0, 1]
                        node_index = node_index + 1  # jump to left child
                        pdf *= traversal_prob0
                        r_max = d_max0
                    else:
                        # choose right child
                        u = (u - traversal_prob0) / \
                            traversal_prob1  # rescale u to [0, 1]
                        node_index = node_index + node.second_child_offset  # jump to right child
                        pdf *= traversal_prob1
                        r_max = d_max1

        return hit, sorted_prim_id, pdf
