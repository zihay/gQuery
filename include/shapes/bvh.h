#pragma once
/**
 * @file bvh.h
 * @brief Bounding Volume Hierarchy implementation for spatial partitioning.
 *
 * BVH (Bounding Volume Hierarchy) is a tree structure that organizes geometric primitives
 * for efficient spatial queries like ray tracing, collision detection, and nearest neighbor searches.
 * This implementation supports both 2D (line segments) and 3D (triangles) primitives and uses
 * surface area heuristics for optimization.
 */
#include <core/fwd.h>
#include <core/parallel.h>
#include <shapes/bounding_box.h>
#include <shapes/line_segment.h>
#include <shapes/triangle.h>

#include <atomic>
#include <memory_resource>
#include <span>
#include <vector>

namespace gquery {

/** @brief Allocator used for memory management during BVH construction */
using Allocator = std::pmr::polymorphic_allocator<std::byte>;

// Forward declarations
template <size_t DIM>
struct SoABVHNode;

template <size_t DIM>
struct SoABVH;

/**
 * @brief Wraps a primitive with its bounding box and original index for BVH construction.
 *
 * This structure maintains the mapping between primitives in the BVH and their original collection.
 * It is templated on dimension to support both 2D (line segments) and 3D (triangles).
 */
template <size_t DIM>
struct BVHPrimitive {
    using BoundingBox = BoundingBox<DIM>;

    size_t      primitive_index; ///< Index in the original primitive collection
    BoundingBox bounding_box;    ///< Axis-aligned bounding box of the primitive
};

template <size_t DIM>
struct alignas(32) BVHNode {
    BoundingBox<DIM> box;
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
template <size_t DIM>
struct BVHBuildNode {
    using BoundingBox = BoundingBox<DIM>;

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

    BoundingBox        box;               ///< Axis-aligned bounding box containing all primitives in this node
    BVHBuildNode<DIM> *left;              ///< Left child (nullptr for leaf nodes)
    BVHBuildNode<DIM> *right;             ///< Right child (nullptr for leaf nodes)
    int                split_axis;        ///< Axis to split on
    int                first_prim_offset; ///< Starting index of primitives in this node
    int                n_primitives;      ///< Number of primitives (0 for interior nodes)
};

// Helper type traits to select the appropriate SoA primitive type based on dimension
template <size_t DIM>
struct SoAPrimitiveTypeSelector {};

template <>
struct SoAPrimitiveTypeSelector<2> {
    using Type = SoALineSegment;
};

template <>
struct SoAPrimitiveTypeSelector<3> {
    using Type = SoATriangle;
};

/**
 * @brief Main BVH class that builds and contains the acceleration structure.
 *
 * Provides methods for constructing and traversing a Bounding Volume Hierarchy
 * optimized for spatial queries on 2D line segments or 3D triangles.
 */
template <size_t DIM>
class BVH {
public:
    using PrimitiveT       = typename gquery::PrimitiveType<DIM>::Type;
    using BoundingBoxType  = BoundingBox<DIM>;
    using VectorType       = Vector<DIM>;
    using BVHNodeType      = BVHNode<DIM>;
    using BVHBuildNodeType = BVHBuildNode<DIM>;
    using BVHPrimitiveType = BVHPrimitive<DIM>;

    /**
     * @brief Methods for splitting nodes during BVH construction.
     */
    enum class SplitMethod {
        SAH ///< Surface Area Heuristic - balances construction cost vs traversal efficiency
    };

    /**
     * @brief Builds a BVH from a collection of vertices and indices.
     *
     * @param vertices Collection of vertices
     * @param indices Collection of indices defining primitives
     * @param max_prims_in_node Maximum number of primitives in a leaf node
     * @param split_method Method used to determine node splitting
     */
    BVH(const std::vector<Vector<DIM>> &vertices, const std::vector<Vectori<DIM>> &indices,
        int max_prims_in_node = 10, SplitMethod split_method = SplitMethod::SAH);

    /**
     * @brief Builds a BVH from a collection of primitives.
     *
     * @param primitives The collection of primitives to organize
     * @param max_prims_in_node Maximum number of primitives in a leaf node
     * @param split_method Method used to determine node splitting
     */
    BVH(const std::vector<PrimitiveT> &primitives, int max_prims_in_node = 10, SplitMethod split_method = SplitMethod::SAH);

    void build();

    /**
     * @brief Core recursive function that builds the BVH tree structure.
     *
     * @param thread_allocators Thread-local allocators for parallel construction
     * @param primitives Span of primitives to organize in this subtree
     * @param total_nodes Counter for tracking the total number of nodes created
     * @param ordered_prims_offset Current offset in the ordered primitives array
     * @param ordered_prims Vector of primitives being reordered for cache-friendly traversal
     * @return Pointer to the root of the constructed subtree
     */
    BVHBuildNodeType *build_recursive(ThreadLocal<Allocator>     &thread_allocators,
                                      std::span<BVHPrimitiveType> primitives,
                                      std::atomic<int>           &total_nodes,
                                      std::atomic<int>           &ordered_prims_offset,
                                      std::vector<PrimitiveT>    &ordered_prims);

    int flatten_bvh(BVHBuildNodeType *node, int *offset);

    // convert to SoABVH
    SoABVH<DIM> to_soa_bvh() const;

    int                      m_max_prims_in_node; ///< Maximum primitives in a leaf node
    std::vector<PrimitiveT>  m_primitives;        ///< Original primitives
    std::vector<PrimitiveT>  m_ordered_prims;     ///< Ordered primitives
    SplitMethod              m_split_method;      ///< Method used for node splitting
    std::vector<BVHNodeType> m_nodes;             ///< Flattened BVH nodes
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
    using SoAPrimitiveType = typename SoAPrimitiveTypeSelector<DIM>::Type;

    SoABVH() = default;

    // constructor from BVH
    SoABVH(const BVH<DIM> &bvh) {
        for (const auto &node : bvh.m_nodes) {
            flat_tree.box.p_min.push_back(node.box.p_min);
            flat_tree.box.p_max.push_back(node.box.p_max);

            flat_tree.reference_offset.push_back(node.primitivesOffset);
            flat_tree.n_references.push_back(node.n_primitives);
            flat_tree.second_child_offset.push_back(node.secondChildOffset);
        }

        // This will need to be specialized for different primitive types
        if constexpr (DIM == 2) {
            for (const auto &primitive : bvh.m_primitives) {
                primitives.a.push_back(primitive.a);
                primitives.b.push_back(primitive.b);
                primitives.index.push_back(primitive.index);
            }

            for (const auto &primitive : bvh.m_ordered_prims) {
                sorted_primitives.a.push_back(primitive.a);
                sorted_primitives.b.push_back(primitive.b);
            }
        } else if constexpr (DIM == 3) {
            for (const auto &primitive : bvh.m_primitives) {
                primitives.a.push_back(primitive.a);
                primitives.b.push_back(primitive.b);
                primitives.c.push_back(primitive.c);
            }

            for (const auto &primitive : bvh.m_ordered_prims) {
                sorted_primitives.a.push_back(primitive.a);
                sorted_primitives.b.push_back(primitive.b);
                sorted_primitives.c.push_back(primitive.c);
            }
        }
    }

    SoABVHNode<DIM>  flat_tree;
    SoAPrimitiveType primitives;
    SoAPrimitiveType sorted_primitives;
};

// Explicit instantiations for both 2D and 3D
// Explicit instantiation for 2D (LineSegment)
template class BVH<2>;

// Explicit instantiation for 3D (Triangle)
template class BVH<3>;

} // namespace gquery