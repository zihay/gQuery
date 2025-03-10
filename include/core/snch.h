#pragma once

#include <core/bounding_box.h>
#include <core/bounding_cone.h>
#include <core/fwd.h>
#include <core/parallel.h>
#include <primitives/line_segment.h>

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
struct alignas(32) SNCHNode {
    BoundingBox<DIM>  box;
    BoundingCone<DIM> cone;
    union {
        int primitives_offset;   // leaf node
        int second_child_offset; // interior node
    };
    uint8_t n_primitives; // number of primitives

    int     silhouette_offset; // 0 if interior node
    uint8_t n_silhouettes;     // number of silhouettes
};

class SNCH {

};

} // namespace gquery
