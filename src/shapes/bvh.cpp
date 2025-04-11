#include <shapes/bvh.h>
#include <util/check.h>
#include <util/span.h>

#include <iostream>
#include <memory_resource>

namespace gquery {

using Allocator = std::pmr::polymorphic_allocator<std::byte>;

template <size_t DIM>
struct BVHSplitBucket {
    BoundingBox<DIM> box;
    int              count = 0; // Initialize to 0
};

struct SplitResult {
    int   axis;
    int   split;
    Float cost;
};

template <size_t DIM>
class SAHSplitMethod {
public:
    using BVHPrimitiveType = BVHPrimitive<DIM>;

    SplitResult find_best_split(const span<BVHPrimitiveType> bvh_primitives) {
        BoundingBox<DIM> box;
        for (const auto &prim : bvh_primitives) {
            box.expand(prim.bounding_box);
        }

        BoundingBox<DIM> centroids_box;
        for (const auto &prim : bvh_primitives) {
            centroids_box.expand(prim.bounding_box.centroid());
        }
        int dim = centroids_box.max_dimension(); // split axis

        // Initialize buckets for SAH partition
        BVHSplitBucket<DIM> buckets[m_n_buckets] = {};
        for (const auto &prim : bvh_primitives) {
            int b = m_n_buckets * centroids_box.offset(prim.bounding_box.centroid())[dim];
            if (b == m_n_buckets)
                b = m_n_buckets - 1;
            // check if the bucket index is valid
            DCHECK_GE(b, 0);
            DCHECK_LT(b, m_n_buckets);

            buckets[b].count++;
            buckets[b].box.expand(prim.bounding_box);
        }

        // compute the cost of each split
        constexpr int n_splits        = m_n_buckets - 1;
        Float         costs[n_splits] = {}; // costs for splitting after each bucket

        // Partially initialize costs using a forward scan
        int              count_below = 0;
        BoundingBox<DIM> box_below;
        for (int i = 0; i < n_splits; ++i) {
            box_below.expand(buckets[i].box);
            count_below += buckets[i].count;
            costs[i] = count_below * box_below.surface_area();
        }

        // Finish initializing costs using a backward scan
        int              count_above = 0;
        BoundingBox<DIM> box_above;
        for (int i = n_splits - 1; i >= 0; --i) {
            box_above.expand(buckets[i + 1].box);
            count_above += buckets[i + 1].count;
            costs[i] += count_above * box_above.surface_area();
        }

        // Find bucket to split at that minimizes SAH metric
        int   min_cost_split = 0;
        Float min_cost       = costs[0];
        for (int i = 1; i < n_splits; ++i) {
            if (costs[i] < min_cost) {
                min_cost       = costs[i];
                min_cost_split = i;
            }
        }

        return SplitResult{ dim, min_cost_split, min_cost };
    }

public:
    constexpr static int m_n_buckets = 12;
};

// Implementation of BVH constructor from vertices and indices
template <size_t DIM>
BVH<DIM>::BVH(const std::vector<Vector<DIM>>  &vertices,
              const std::vector<Vectori<DIM>> &indices,
              int max_prims_in_node, SplitMethod split_method)
    : m_max_prims_in_node(max_prims_in_node), m_split_method(split_method) {
    m_primitives.reserve(indices.size());

    if constexpr (DIM == 2) {
        // 2D case - LineSegment construction
        for (int i = 0; i < indices.size(); ++i) {
            auto        face = indices[i];
            auto        v0   = vertices[face[0]];
            auto        v1   = vertices[face[1]];
            LineSegment line_segment;
            line_segment.index   = i;
            line_segment.a       = v0;
            line_segment.b       = v1;
            line_segment.indices = face;
            m_primitives.push_back(line_segment);
        }
    } else if constexpr (DIM == 3) {
        // 3D case - Triangle construction
        for (int i = 0; i < indices.size(); ++i) {
            auto     face = indices[i];
            auto     v0   = vertices[face[0]];
            auto     v1   = vertices[face[1]];
            auto     v2   = vertices[face[2]];
            Triangle triangle;
            triangle.index   = i;
            triangle.a       = v0;
            triangle.b       = v1;
            triangle.c       = v2;
            triangle.indices = face;
            m_primitives.push_back(triangle);
        }
    }

    build();
}

// Implementation of BVH constructor from primitives
template <size_t DIM>
BVH<DIM>::BVH(const std::vector<PrimitiveT> &primitives, int max_prims_in_node, SplitMethod split_method)
    : m_primitives(primitives), m_max_prims_in_node(max_prims_in_node), m_split_method(split_method) {
    build();
}

// Implementation of BVH::build
template <size_t DIM>
void BVH<DIM>::build() {
    DCHECK(!m_primitives.empty());
    std::vector<BVHPrimitiveType> bvh_primitives(m_primitives.size());
    for (size_t i = 0; i < m_primitives.size(); ++i) {
        bvh_primitives[i] = BVHPrimitiveType{ i, m_primitives[i].bounding_box() };
    }

    std::pmr::monotonic_buffer_resource resource;
    Allocator                           allocator(&resource);
    using Resource = std::pmr::monotonic_buffer_resource;
    std::vector<std::unique_ptr<Resource>> thread_buffer_resources;
    ThreadLocal<Allocator>                 thread_allocators([&thread_buffer_resources]() {
        thread_buffer_resources.push_back(std::make_unique<Resource>());
        return Allocator(thread_buffer_resources.back().get());
    });

    std::vector<PrimitiveT> ordered_prims(m_primitives.size());

    BVHBuildNodeType *root;
    std::atomic<int>  total_nodes{ 0 };

    std::atomic<int> ordered_prims_offset{ 0 };
    root = build_recursive(thread_allocators, span<BVHPrimitiveType>(bvh_primitives),
                           total_nodes, ordered_prims_offset, ordered_prims);
    CHECK_EQ(ordered_prims_offset.load(), ordered_prims.size());
    m_ordered_prims = std::move(ordered_prims);

    bvh_primitives.resize(0);
    bvh_primitives.shrink_to_fit();

    m_nodes.resize(total_nodes.load());
    int offset = 0;
    flatten_bvh(root, &offset);
    CHECK_EQ(total_nodes.load(), offset);
}

// Implementation of BVH::build_recursive
template <size_t DIM>
typename BVH<DIM>::BVHBuildNodeType *BVH<DIM>::build_recursive(
    ThreadLocal<Allocator>  &thread_allocators,
    span<BVHPrimitiveType>   bvh_primitives,
    std::atomic<int>        &total_nodes,
    std::atomic<int>        &ordered_prims_offset,
    std::vector<PrimitiveT> &ordered_prims) {
    DCHECK_GT(bvh_primitives.size(), 0);
    Allocator         alloc = thread_allocators.Get();
    BVHBuildNodeType *node  = alloc.new_object<BVHBuildNodeType>();
    ++total_nodes;

    // Compute bounds of all primitives in BVH node
    BoundingBox<DIM> box;
    for (const auto &prim : bvh_primitives) {
        box.expand(prim.bounding_box);
    }

    // Check for early termination cases
    if (box.surface_area() == 0 || bvh_primitives.size() == 1) {
        // early termination if the box is degenerate or there is only one primitive
        int first_prim_offset = ordered_prims_offset.fetch_add(bvh_primitives.size());
        for (size_t i = 0; i < bvh_primitives.size(); ++i) {
            int index                            = bvh_primitives[i].primitive_index;
            ordered_prims[first_prim_offset + i] = m_primitives[index]; // copy primitive from original primitives to ordered_prims
        }
        node->init_leaf(first_prim_offset, bvh_primitives.size(), box);
        return node;
    }

    // Compute bound of primitive centroids and choose split dimension
    BoundingBox<DIM> centroid_box;
    for (const auto &prim : bvh_primitives) {
        centroid_box.expand(prim.bounding_box.centroid());
    }
    int dim = centroid_box.max_dimension();

    if (centroid_box.p_max[dim] == centroid_box.p_min[dim]) {
        // early termination if the centroid box is degenerate
        int first_prim_offset = ordered_prims_offset.fetch_add(bvh_primitives.size());
        for (size_t i = 0; i < bvh_primitives.size(); ++i) {
            int index                            = bvh_primitives[i].primitive_index;
            ordered_prims[first_prim_offset + i] = m_primitives[index]; // copy primitive from original primitives to ordered_prims
        }
        node->init_leaf(first_prim_offset, bvh_primitives.size(), box);
        return node;
    }

    int mid = bvh_primitives.size() / 2;
    switch (m_split_method) {
        case SplitMethod::SAH:
        default: {
            if (bvh_primitives.size() <= 2) {
                mid = bvh_primitives.size() / 2;
                std::nth_element(bvh_primitives.begin(), bvh_primitives.begin() + mid,
                                 bvh_primitives.end(),
                                 [dim](const BVHPrimitiveType &a, const BVHPrimitiveType &b) {
                                     return a.bounding_box.centroid()[dim] < b.bounding_box.centroid()[dim];
                                 });
            } else {
                auto split_method = SAHSplitMethod<DIM>();
                auto split_result = split_method.find_best_split(bvh_primitives);

                int   n_buckets      = split_method.m_n_buckets;
                int   min_cost_split = split_result.split;
                Float min_cost       = split_result.cost;

                // compute leaf cost and SAH split cost for the chosen split
                Float leaf_cost = bvh_primitives.size();
                // normalize the cost by dividing by the surface area of the node
                min_cost = 1.f / 2.f + min_cost / box.surface_area();

                // perform SAH split if the cost is lower than the leaf cost
                if (bvh_primitives.size() > m_max_prims_in_node || min_cost < leaf_cost) {
                    auto mid_iter = std::partition(
                        bvh_primitives.begin(), bvh_primitives.end(),
                        [=](const BVHPrimitiveType &prim) {
                            int b = n_buckets * centroid_box.offset(prim.bounding_box.centroid())[dim];
                            if (b == n_buckets)
                                b = n_buckets - 1;
                            return b <= min_cost_split;
                        });
                    mid = mid_iter - bvh_primitives.begin();
                } else {
                    // create leaf node
                    int first_prim_offset = ordered_prims_offset.fetch_add(bvh_primitives.size());
                    for (size_t i = 0; i < bvh_primitives.size(); ++i) {
                        int index                            = bvh_primitives[i].primitive_index;
                        ordered_prims[first_prim_offset + i] = m_primitives[index]; // copy primitive from original primitives to ordered_prims
                    }
                    node->init_leaf(first_prim_offset, bvh_primitives.size(), box);
                    return node;
                }
            }
            break;
        }
    }

    BVHBuildNodeType *left  = nullptr;
    BVHBuildNodeType *right = nullptr;
    if (bvh_primitives.size() > 128 * 1024) {
        ParallelFor(0, 2, [&](int i) {
            if (i == 0) {
                left = build_recursive(
                    thread_allocators, bvh_primitives.subspan(0, mid),
                    total_nodes, ordered_prims_offset, ordered_prims);
            } else {
                right = build_recursive(
                    thread_allocators, bvh_primitives.subspan(mid),
                    total_nodes, ordered_prims_offset, ordered_prims);
            }
        });
    } else {
        left = build_recursive(
            thread_allocators, bvh_primitives.subspan(0, mid),
            total_nodes, ordered_prims_offset, ordered_prims);
        right = build_recursive(
            thread_allocators, bvh_primitives.subspan(mid),
            total_nodes, ordered_prims_offset, ordered_prims);
    }

    node->init_interior(dim, left, right);
    return node;
}

// Implementation of BVH::flatten_bvh
template <size_t DIM>
int BVH<DIM>::flatten_bvh(BVHBuildNodeType *node, int *offset) {
    BVHNodeType *linear_node = &m_nodes[*offset];
    linear_node->box         = node->box;
    int my_offset            = (*offset)++;

    if (node->n_primitives > 0) {
        DCHECK(!node->left && !node->right);
        linear_node->primitivesOffset = node->first_prim_offset;
        linear_node->n_primitives     = node->n_primitives;
    } else {
        // Create interior flattened BVH node
        linear_node->axis         = node->split_axis;
        linear_node->n_primitives = 0;
        flatten_bvh(node->left, offset);
        // TODO: improve this
        size_t second_child_offset     = flatten_bvh(node->right, offset);
        linear_node->secondChildOffset = second_child_offset - my_offset;
    }
    return my_offset;
}

template <size_t DIM>
ArrayX BVH<DIM>::primitive_data() const {
    if (m_ordered_prims.empty()) {
        return ArrayX();
    }
    size_t segment_size = m_ordered_prims[0].flatten().size();
    ArrayX ret(m_ordered_prims.size() * segment_size);
    for (size_t i = 0; i < m_ordered_prims.size(); ++i) {
        ret.segment(i * segment_size, segment_size) = m_ordered_prims[i].flatten();
    }
    return ret;
}

template <size_t DIM>
ArrayX BVH<DIM>::node_data() const {
    if (m_nodes.empty()) {
        return ArrayX();
    }
    size_t segment_size = m_nodes[0].flatten().size();
    ArrayX ret(m_nodes.size() * segment_size);
    for (size_t i = 0; i < m_nodes.size(); ++i) {
        ret.segment(i * segment_size, segment_size) = m_nodes[i].flatten();
    }
    return ret;
}

// Explicit instantiations for SAHSplitMethod only
template class SAHSplitMethod<2>;
template class SAHSplitMethod<3>;

// Explicit instantiations for both 2D and 3D
template class BVH<2>;
template class BVH<3>;
} // namespace gquery