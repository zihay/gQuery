#pragma once
/**
 * @file bvh.h
 * @brief Bounding Volume Hierarchy implementation for spatial partitioning.
 *
 * BVH (Bounding Volume Hierarchy) is a tree structure that organizes geometric primitives
 * for efficient spatial queries like ray tracing, collision detection, and nearest neighbor searches.
 * This implementation supports 2D primitives and uses surface area heuristics for optimization.
 */
#include <core/bounding_box.h>
#include <core/fwd.h>
#include <core/parallel.h>
#include <primitives/line_segment.h>

#include <atomic>
#include <memory_resource>
#include <span>
#include <vector>

namespace gquery {

/** @brief Allocator used for memory management during BVH construction */
using Allocator = std::pmr::polymorphic_allocator<std::byte>;

/**
 * @brief Wraps a primitive with its bounding box and original index for BVH construction.
 *
 * This structure maintains the mapping between primitives in the BVH and their original collection.
 */
struct BVHPrimitive {
    using BoundingBox = BoundingBox<2>;

    size_t      primitive_index; ///< Index in the original primitive collection
    BoundingBox bounding_box;    ///< Axis-aligned bounding box of the primitive
};

struct alignas(32) BVHNode {
    BoundingBox<2> box;
    union {
        int primitivesOffset;  // leaf node
        int secondChildOffset; // interior node
    };

    uint16_t n_primitives; // 0 if interior node
    uint8_t  axis;         // for interior node, axis to split on
};

/**
 * @brief A node in the BVH tree structure.
 *
 * Can be either:
 * - An interior node with child pointers (n_primitives = 0)
 * - A leaf node with primitives (left = right = nullptr)
 */
struct BVHBuildNode {
    using BoundingBox = BoundingBox<2>;

    /**
     * @brief Initializes a leaf node with the given parameters.
     *
     * @param first The starting index of the primitives in the original collection
     * @param n The number of primitives in this node
     * @param box The bounding box of the primitives in this node
     */
    void init_leaf(int first, int n, const BoundingBox &box) {
        first_prim_offset = first;
        n_primitives      = n;
        this->box         = box;
        left              = nullptr;
        right             = nullptr;
    }

    void init_interior(int axis, BVHBuildNode *left, BVHBuildNode *right) {
        this->split_axis = axis;
        this->left       = left;
        this->right      = right;
        n_primitives     = 0;
        box              = BoundingBox();
        box.expand(left->box);
        box.expand(right->box);
    }

    BoundingBox   box;               ///< Axis-aligned bounding box containing all primitives in this node
    BVHBuildNode *left;              ///< Left child (nullptr for leaf nodes)
    BVHBuildNode *right;             ///< Right child (nullptr for leaf nodes)
    int           split_axis;        ///< Axis to split on
    int           first_prim_offset; ///< Starting index of primitives in this node
    int           n_primitives;      ///< Number of primitives (0 for interior nodes)
};

/**
 * @brief Main BVH class that builds and contains the acceleration structure.
 *
 * Provides methods for constructing and traversing a Bounding Volume Hierarchy
 * optimized for spatial queries on 2D line segments.
 */
class BVH {
public:
    /**
     * @brief Methods for splitting nodes during BVH construction.
     */
    enum class SplitMethod {
        SAH ///< Surface Area Heuristic - balances construction cost vs traversal efficiency
    };

    /**
     * @brief Builds a BVH from a collection of line segments.
     *
     * @param primitives The collection of line segments to organize
     * @param max_prims_in_node Maximum number of primitives in a leaf node
     * @param split_method Method used to determine node splitting
     */
    BVH(const std::vector<LineSegment> &primitives, int max_prims_in_node = 10, SplitMethod split_method = SplitMethod::SAH);

    /**
     * @brief Core recursive function that builds the BVH tree structure.
     *
     * @param thread_allocators Thread-local allocators for parallel construction
     * @param primitives Span of primitives to organize in this subtree
     * @param total_nodes Counter for tracking the total number of nodes created
     * @param ordered_prims_offset Current offset in the ordered primitives array
     * @param ordered_prims Vector of primitives being reordered for cache-friendly traversal
     * @return Pointer to the root of the constructed subtree
     *
     * This function:
     * - Creates leaf nodes when the primitive count is small enough
     * - Splits primitives using the selected method and creates child nodes
     * - Uses thread-local allocators for better performance
     * - Reorders primitives for more cache-friendly traversal
     */
    BVHBuildNode *build_recursive(ThreadLocal<Allocator>   &thread_allocators,
                                  std::span<BVHPrimitive>   primitives,
                                  std::atomic<int>         &total_nodes,
                                  std::atomic<int>         &ordered_prims_offset,
                                  std::vector<LineSegment> &ordered_prims);

    int flatten_bvh(BVHBuildNode *node, int *offset);

    int                        m_max_prims_in_node; ///< Maximum primitives in a leaf node
    std::vector<LineSegment>   m_primitives;        ///< Original primitives
    std::vector<LineSegment>   m_ordered_prims;     ///< Ordered primitives
    SplitMethod                m_split_method;      ///< Method used for node splitting
    std::vector<BVHNode> m_nodes;             ///< Flattened BVH nodes
};

template <size_t DIM>
struct SoABVHNode {
    SoABoundingBox<DIM> box;
    std::vector<int>    reference_offset;
    std::vector<int>    n_references;
    std::vector<int>    second_child_offset;
};

template <size_t DIM>
struct SoABVH {
    SoABVH() = default;

    SoABVH(const BVH &bvh) {
        for (const auto &node : bvh.m_nodes) {
            flat_tree.box.p_min.push_back(node.box.p_min);
            flat_tree.box.p_max.push_back(node.box.p_max);

            flat_tree.reference_offset.push_back(node.primitivesOffset);
            flat_tree.n_references.push_back(node.n_primitives);
            flat_tree.second_child_offset.push_back(node.secondChildOffset);
        }

        for (const auto &primitive : bvh.m_primitives) {
            primitives.a.push_back(primitive.a);
            primitives.b.push_back(primitive.b);
        }

        for (const auto &primitive : bvh.m_ordered_prims) {
            sorted_primitives.a.push_back(primitive.a);
            sorted_primitives.b.push_back(primitive.b);
        }
    }

    SoABVHNode<DIM> flat_tree;
    SoALineSegment  primitives;
    SoALineSegment  sorted_primitives;
};

} // namespace gquery