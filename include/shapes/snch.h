#pragma once

#include <core/fwd.h>
#include <core/parallel.h>
#include <shapes/bounding_box.h>
#include <shapes/bounding_cone.h>
#include <shapes/bvh.h>
#include <shapes/silhouette_edge.h>
#include <shapes/silhouette_vertex.h>

#include <atomic>
#include <memory_resource>
#include <span>
#include <vector>

namespace gquery {

template <size_t DIM>
struct SNCHBuildNode {
    BoundingBox<DIM>  box;
    BoundingCone<DIM> cone;
    SNCHBuildNode    *left;
    SNCHBuildNode    *right;
    int               split_axis;        ///< Axis to split on
    int               first_prim_offset; ///< Starting index of primitives in this node
    int               n_primitives;      ///< Number of primitives (0 for interior nodes)
};

template <size_t DIM>
struct SNCHNode {
    BoundingBox<DIM>  box;
    BoundingCone<DIM> cone;

    size_t primitives_offset;   // leaf node
    size_t second_child_offset; // interior node
    size_t n_primitives;        // number of primitives

    size_t silhouette_offset; // 0 if interior node
    size_t n_silhouettes;     // number of silhouettes

    bool is_leaf() const { return n_primitives > 0; }

    ArrayX flatten() const {
        ArrayX ret(DIM * 2 + DIM + 2 + 3 + 2);
        ret << box.p_min, box.p_max, cone.axis, cone.half_angle, cone.radius,
            Float(primitives_offset), Float(second_child_offset), Float(n_primitives),
            Float(silhouette_offset), Float(n_silhouettes);
        return ret;
    }
};

template <size_t DIM>
class SNCH : public BVH<DIM> {
public:
    using SilhouetteType = std::conditional_t<DIM == 2, SilhouetteVertex, SilhouetteEdge>;

    SNCH(const std::vector<Vector<DIM>> &vertices, const std::vector<Vectori<DIM>> &indices,
         int max_prims_in_node = 10, SplitMethod split_method = SplitMethod::SAH);

    void build();

    void build_silhouettes(const std::vector<Vector<DIM>>  &vertices,
                           const std::vector<Vectori<DIM>> &indices);

    void build_recursive(size_t start, size_t end);

    ArrayX primitive_data() const;
    ArrayX silhouette_data() const;
    ArrayX node_data() const;

public:
    std::vector<SilhouetteType> m_silhouettes;
    std::vector<SNCHNode<DIM>>  m_nodes; ///< Flattened SNCH nodes
};

} // namespace gquery