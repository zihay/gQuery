#pragma once
#include <core/bounding_box.h>
#include <core/fwd.h>
#include <core/parallel.h>
#include <primitives/line_segment.h>

namespace gquery {

struct BVHPrimitive {
    using BoundingBox = BoundingBox<2>;

    size_t      primitive_index;
    BoundingBox bounding_box;
};
struct BVHBuildNode {
    using BoundingBox = BoundingBox<2>;

    BoundingBox   box;
    BVHBuildNode *left;
    BVHBuildNode *right;
    int           first_prim_offset;
    int           n_primitives;
};

class BVH {
public:
    enum class SplitMethod {
        SAH
    };

    BVH(const std::vector<LineSegment> &primitives, int max_prims_in_node = 10, SplitMethod split_method = SplitMethod::SAH);
    BVHBuildNode *build_recursive();

    int                      m_max_prims_in_node;
    std::vector<LineSegment> m_primitives;
    SplitMethod              m_split_method;
};

} // namespace gquery