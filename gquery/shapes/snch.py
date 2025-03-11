from core.fwd import *
from core.math import in_range
from shapes.bvh import BVHNode, BoundingCone


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

    @dr.syntax
    def overlaps_3(self, x: Array2, e: Array2,
                   rho_bound: Float = Float(1.),
                   r_max: Float = Float(dr.inf),
                   type: Int = Int(0)):
        # used by star radius 2
        is_overlap = Bool(False)
        d_min = Float(dr.inf)
        is_overlap, d_min, d_max = self.box.overlaps(x, r_max)
        if self.cone.is_valid() & is_overlap:
            is_overlap = (self.cone.half_angle > (dr.pi / 2)) \
                & (d_min < dr.epsilon(Float))
            if ~is_overlap:
                c = self.box.centroid()
                view_cone_axis = c - x
                l = dr.norm(view_cone_axis)
                view_cone_axis = view_cone_axis / l

                is_overlap = l < self.cone.radius
                if ~is_overlap:
                    theta = dr.acos(
                        dr.clamp(dr.dot(self.cone.axis, e), -1., 1.))
                    theta_min = theta - self.cone.half_angle
                    theta_max = theta + self.cone.half_angle

                    cos_min = dr.minimum(dr.cos(theta_min), dr.cos(theta_max))
                    cos_max = dr.maximum(dr.cos(theta_min), dr.cos(theta_max))
                    if in_range(0., theta_min, theta_max):
                        cos_max = Float(1.)

                    # TODO: sin could be negative
                    sin_min = dr.minimum(dr.sin(theta_min), dr.sin(theta_max))
                    if in_range(-dr.pi / 2., theta_min, theta_max):
                        sin_min = Float(-1.)

                    sin_max = dr.maximum(dr.sin(theta_min), dr.sin(theta_max))
                    if in_range(dr.pi / 2., theta_min, theta_max):
                        sin_max = Float(1.)

                    axis_angle = dr.acos(
                        dr.clamp(dr.dot(self.cone.axis, view_cone_axis), -1., 1.))
                    view_cone_half_angle = dr.asin(
                        self.cone.radius / l)
                    half_angle_sum = self.cone.half_angle + view_cone_half_angle
                    alpha_min = axis_angle - half_angle_sum
                    alpha_max = axis_angle + half_angle_sum

                    is_overlap = (half_angle_sum > (dr.pi / 2)) \
                        | in_range(dr.pi / 2., alpha_min, alpha_max)

                    if ~is_overlap:
                        # assume alpha_min > -pi/2 and alpha_max < pi/2
                        tan_min = dr.tan(alpha_min)
                        tan_max = dr.tan(alpha_max)
                        if type == BoundaryType.Dirichlet.value:
                            # reflectance bound
                            rho_min = cos_min + sin_min * tan_min
                            rho_max = cos_max + sin_max * tan_max
                            is_overlap = (rho_min < -rho_bound) \
                                | (rho_max > rho_bound)
                        elif type == BoundaryType.Neumann.value:
                            # reflectance bound
                            _min = cos_min * tan_min
                            _min = dr.minimum(_min, cos_max * tan_max)
                            _min = dr.minimum(_min, cos_min * tan_max)
                            _min = dr.minimum(_min, cos_max * tan_min)
                            _max = cos_min * tan_min
                            _max = dr.maximum(_max, cos_max * tan_max)
                            _max = dr.maximum(_max, cos_min * tan_max)
                            _max = dr.maximum(_max, cos_max * tan_min)
                            rho_min = sin_min - _max
                            rho_max = sin_max - _min
                            dr.assert_true(rho_min <= rho_max)
                            is_overlap = (rho_min <= -rho_bound) \
                                | (rho_max >= rho_bound)

        return is_overlap, d_min

    @dr.syntax
    def overlaps_2(self, x: Array2, e: Array2,
                   rho_bound: Float = Float(1.),
                   r_max: Float = Float(dr.inf),
                   type: Int = Int(0)):
        # used by star radius 2
        is_overlap = Bool(False)
        d_min = Float(dr.inf)
        is_overlap, d_min, d_max = self.box.overlaps(x, r_max)
        if self.cone.is_valid() & is_overlap:
            is_overlap = (self.cone.half_angle > (dr.pi / 2)) \
                & (d_min < dr.epsilon(Float))
            if ~is_overlap:
                c = self.box.centroid()
                view_cone_axis = c - x
                l = dr.norm(view_cone_axis)
                view_cone_axis = view_cone_axis / l

                is_overlap = l < self.cone.radius
                if ~is_overlap:
                    axis_angle = dr.acos(
                        dr.clamp(dr.dot(self.cone.axis, view_cone_axis), -1., 1.))
                    view_cone_half_angle = dr.asin(
                        self.cone.radius / l)
                    half_angle_sum = self.cone.half_angle + view_cone_half_angle
                    alpha_min = axis_angle - half_angle_sum
                    alpha_max = axis_angle + half_angle_sum

                    is_overlap = (half_angle_sum > (dr.pi / 2)) \
                        | in_range(dr.pi / 2., alpha_min, alpha_max)

                    if ~is_overlap:
                        # alpha_min > -pi/2 and alpha_max < pi/2
                        dn_min = dr.minimum(
                            dr.cos(alpha_min), dr.cos(alpha_max))
                        dn_max = dr.maximum(
                            dr.cos(alpha_min), dr.cos(alpha_max))

                        ed_axis_angle = dr.acos(
                            dr.clamp(dr.dot(e, view_cone_axis), -1., 1.))
                        ed_angle_min = ed_axis_angle - self.cone.half_angle
                        ed_angle_max = ed_axis_angle + self.cone.half_angle
                        ed_min = dr.minimum(
                            dr.cos(ed_angle_min), dr.cos(ed_angle_max))
                        ed_max = dr.maximum(
                            dr.cos(ed_angle_min), dr.cos(ed_angle_max))

                        if type == BoundaryType.Dirichlet.value:
                            ed_en_max = ed_max / dn_min
                            ed_en_min = Float(0.)
                            if ed_min < 0.:
                                ed_en_min = ed_min / dn_min
                            else:
                                ed_en_min = ed_min / dn_max
                            is_overlap = (ed_en_max > rho_bound) \
                                | (ed_en_min < -rho_bound)
                        elif type == BoundaryType.Neumann.value:
                            en_axis_angle = dr.acos(
                                dr.clamp(dr.dot(e, self.cone.axis), -1., 1.))
                            en_angle_min = en_axis_angle - self.cone.half_angle
                            en_angle_max = en_axis_angle + self.cone.half_angle
                            en_min = dr.minimum(
                                dr.cos(en_angle_min), dr.cos(en_angle_max))
                            en_max = dr.maximum(
                                dr.cos(en_angle_min), dr.cos(en_angle_max))

                            en_dn_max = en_max / dn_min
                            en_dn_min = Float(0.)
                            if en_min < 0.:
                                en_dn_min = en_min / dn_min
                            else:
                                en_dn_min = en_min / dn_max

                            def rho(ed, en_dn):
                                return 1. - 2. * ed * en_dn + dr.sqr(en_dn)

                            rho_max = rho(ed_min, en_dn_min)
                            rho_max = dr.maximum(rho_max,
                                                 rho(ed_min, en_dn_max))
                            rho_max = dr.maximum(rho_max,
                                                 rho(ed_max, en_dn_min))
                            rho_max = dr.maximum(rho_max,
                                                 rho(ed_max, en_dn_max))

                            is_overlap = (rho_max > rho_bound * rho_bound)

        return is_overlap, d_min

    @dr.syntax
    def visit(self, x: Array2, v: Array2,
              rho_max: Float = Float(1.),
              r_max: Float = Float(dr.inf)):
        is_overlap = Bool(False)
        d_min = Float(dr.inf)
        # check box inside the query sphere
        is_overlap, d_min, d_max = self.box.overlaps(x, r_max)
        if self.cone.is_valid() & is_overlap:
            is_overlap = (self.cone.half_angle > (dr.pi / 2)) \
                & (d_min < dr.epsilon(Float))
            if ~is_overlap:
                c = self.box.centroid()
                view_cone_axis = c - x
                l = dr.norm(view_cone_axis)
                view_cone_axis = view_cone_axis / l
                is_overlap = l < self.cone.radius  # point inside the bounding sphere
                if ~is_overlap:
                    # angle between view_cone_axis and self.cone.axis
                    axis_angle = dr.acos(
                        dr.clamp(dr.dot(self.cone.axis, view_cone_axis), -1., 1.))
                    view_cone_half_angle = dr.asin(
                        self.cone.radius / l)
                    half_angle_sum = self.cone.half_angle + view_cone_half_angle
                    alpha_min = axis_angle - half_angle_sum
                    alpha_max = axis_angle + half_angle_sum

                    # probably has silhouettes
                    is_overlap = (half_angle_sum > (dr.pi / 2)) \
                        | in_range(dr.pi / 2., alpha_min, alpha_max)

                    if ~is_overlap:
                        # alpha_min > -pi/2 and alpha_max < pi/2
                        cos_min_alpha = dr.minimum(
                            dr.cos(alpha_min), dr.cos(alpha_max))
                        cos_max_alpha = dr.maximum(
                            dr.cos(alpha_min), dr.cos(alpha_max))
                        if (alpha_min < 0.) & (alpha_max > 0.):
                            cos_max_alpha = dr.maximum(cos_max_alpha, 1.)

                        beta = dr.acos(
                            dr.clamp(dr.dot(self.cone.axis, v), -1., 1.))
                        beta_min = beta - self.cone.half_angle
                        beta_max = beta + self.cone.half_angle
                        cos_min_beta = dr.minimum(
                            dr.cos(beta_min), dr.cos(beta_max))
                        cos_max_beta = dr.maximum(
                            dr.cos(beta_min), dr.cos(beta_max))
                        if (beta_min < 0.) & (beta_max > 0.):
                            cos_max_beta = dr.maximum(cos_max_beta, 1.)

                        gamma = dr.acos(
                            dr.clamp(dr.dot(v, view_cone_axis), -1., 1.))
                        gamma_min = gamma - view_cone_half_angle
                        gamma_max = gamma + view_cone_half_angle
                        cos_min_gamma = dr.minimum(
                            dr.cos(gamma_min), dr.cos(gamma_max))
                        cos_max_gamma = dr.maximum(
                            dr.cos(gamma_min), dr.cos(gamma_max))
                        if (gamma_min < 0.) & (gamma_max > 0.):
                            cos_max_gamma = dr.maximum(cos_max_gamma, 1.)

                        a_min = cos_min_beta / cos_max_alpha
                        a_max = cos_max_beta / cos_min_alpha

                        b_min = cos_min_gamma
                        b_max = cos_max_gamma

                        rho2max = 1 - dr.minimum(dr.minimum(a_min * b_min, a_min * b_max),
                                                 dr.minimum(a_max * b_min, a_max * b_max)) + dr.maximum(a_min * a_min, a_max * a_max)
                        is_overlap = (rho2max > rho_max * rho_max)

        return is_overlap, d_min


class SNCH:
    flat_tree: SNCHNode
    primitives: LineSegment
    silhouettes: SilhouetteVertex

    def __init__(self, vertices: Array2, indices: Array2i, types: Int):
        import diff_wost as dw
        c_scene = dw.Polyline.from_neumann(vertices.numpy().T, indices.numpy().T,
                                           build_bvh=True)
        snch = c_scene.reflecting_boundary_handler.scene.getSceneData().aggregate
        primitives = dw.convert_line_segments(snch.primitives)
        self.primitives = LineSegment(
            a=Array2(np.array(primitives.a).T),
            b=Array2(np.array(primitives.b).T),
            index=Int(np.array(primitives.index)),
            sorted_index=dr.arange(Int, len(primitives.index)),
            type=dr.gather(Int, types, np.array(primitives.index)))
        _silhouettes = dw.convert_silhouette_vertices(snch.silhouetteRefs)
        self.silhouettes = SilhouetteVertex(
            indices=Array3i(np.array(_silhouettes.indices).T),
            index=Int(np.array(_silhouettes.pIndex)),
            a=Array2(np.array(_silhouettes.a).T),
            b=Array2(np.array(_silhouettes.b).T),
            c=Array2(np.array(_silhouettes.c).T),
            prim_id=dr.arange(Int, len(_silhouettes.pIndex))
        )
        _flat_tree: dw.SNCHNodeSoA = dw.convert_snch_nodes(snch.flatTree)
        box = BoundingBox(p_min=Array2(np.array(_flat_tree.box.pMin).T),
                          p_max=Array2(np.array(_flat_tree.box.pMax).T))
        cone = BoundingCone(axis=Array2(np.array(_flat_tree.cone.axis).T),
                            half_angle=Float(
                                np.array(_flat_tree.cone.halfAngle)),
                            radius=Float(np.array(_flat_tree.cone.radius)))
        self.flat_tree = SNCHNode(
            box=box,
            reference_offset=Int(np.array(_flat_tree.referenceOffset)),
            second_child_offset=Int(np.array(_flat_tree.secondChildOffset)),
            n_references=Int(np.array(_flat_tree.nReferences)),
            silhouette_reference_offset=Int(
                np.array(_flat_tree.silhouetteReferenceOffset)),
            n_silhouette_references=Int(
                np.array(_flat_tree.nSilhouetteReferences)),
            cone=cone)

    @dr.syntax
    def closest_silhouette(self, x: Array2, r_max: Float = Float(dr.inf)):
        r_max = Float(r_max)
        c_rec = dr.zeros(ClosestSilhouettePointRecord)
        root_node = dr.gather(SNCHNode, self.flat_tree, 0)
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
                    node = dr.gather(SNCHNode, self.flat_tree, node_index)
                    if node.is_leaf():  # leaf node
                        j = Int(0)
                        while j < node.n_silhouette_references:
                            reference_index = node.silhouette_reference_offset + j
                            silhouette = dr.gather(SilhouetteVertex, self.silhouettes,
                                                   reference_index)
                            _c_rec = silhouette.closest_silhouette(
                                x, r_max)
                            if _c_rec.valid & (_c_rec.d < r_max):
                                r_max = dr.minimum(r_max, _c_rec.d)
                                c_rec = _c_rec
                            j += 1
                    else:  # non-leaf node
                        left = dr.gather(SNCHNode, self.flat_tree,
                                         node_index + 1)
                        right = dr.gather(SNCHNode, self.flat_tree,
                                          node_index + node.second_child_offset)

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
    def star_radius_2(self, x: Array2, e: Array2,
                      rho_max: Float = Float(1.),
                      r_max: Float = Float(dr.inf),
                      type: Int = Int(0)):
        node_visited = Int(0)
        r_max = Float(r_max)
        root_node = dr.gather(SNCHNode, self.flat_tree, 0)
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
                    node = dr.gather(SNCHNode, self.flat_tree, node_index)
                    if node.is_leaf():  # leaf node
                        j = Int(0)
                        while j < node.n_references:
                            reference_index = node.reference_offset + j
                            prim = dr.gather(LineSegment, self.primitives,
                                             reference_index)
                            _r_max = prim.star_radius_2(x, e,
                                                        rho_max, r_max)
                            if _r_max < r_max:
                                r_max = _r_max
                            node_visited += 1
                            j += 1
                    else:  # non-leaf node
                        left = dr.gather(SNCHNode, self.flat_tree,
                                         node_index + 1)
                        right = dr.gather(SNCHNode, self.flat_tree,
                                          node_index + node.second_child_offset)

                        hit0, d_min0 = left.overlaps_2(x, e,
                                                       rho_max, r_max, type)
                        hit1, d_min1 = right.overlaps_2(x, e,
                                                        rho_max, r_max, type)

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
                        node_visited += 1
        return r_max

    @dr.syntax
    def star_radius_3(self, x: Array2, v: Array2,
                      rho_max: Float = Float(1.),
                      r_max: Float = Float(dr.inf)):
        self.node_visited = Int(0)
        pt = dr.zeros(Array3)
        r_max = Float(r_max)
        root_node = dr.gather(SNCHNode, self.flat_tree, 0)
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
                    node = dr.gather(SNCHNode, self.flat_tree, node_index)
                    if node.is_leaf():  # leaf node
                        j = Int(0)
                        while j < node.n_references:
                            reference_index = node.reference_offset + j
                            prim = dr.gather(LineSegment, self.primitives,
                                             reference_index)
                            _r_max, _pt = prim.star_radius_2(x, v,
                                                             rho_max, r_max)
                            if _r_max < r_max:
                                r_max = _r_max
                                pt = _pt
                            j += 1
                        self.node_visited += 1
                    else:  # non-leaf node
                        left = dr.gather(SNCHNode, self.flat_tree,
                                         node_index + 1)
                        right = dr.gather(SNCHNode, self.flat_tree,
                                          node_index + node.second_child_offset)

                        hit0, d_min0 = left.visit(x, v,
                                                  rho_max, r_max)

                        hit1, d_min1 = right.visit(x, v,
                                                   rho_max, r_max)

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

                        self.node_visited += 1
        return r_max, pt
