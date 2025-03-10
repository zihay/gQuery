#include <core/bvh.h>
#include <util/check.h>

#include <memory_resource>
namespace gquery {

using Allocator = std::pmr::polymorphic_allocator<std::byte>;

struct BVHSplitBucket {
    BoundingBox<2> box;
    int            count;
};

BVH::BVH(const std::vector<LineSegment> &primitives, int max_prims_in_node, SplitMethod split_method)
    : m_primitives(std::move(primitives)), m_max_prims_in_node(max_prims_in_node), m_split_method(split_method) {
    CHECK(!m_primitives.empty());
    std::vector<BVHPrimitive> bvh_primitives(m_primitives.size());
    for (size_t i = 0; i < m_primitives.size(); ++i) {
        bvh_primitives[i] = BVHPrimitive{ i, m_primitives[i].bounding_box() };
    }

    std::pmr::monotonic_buffer_resource resource;
    Allocator                           allocator(&resource);
    using Resource = std::pmr::monotonic_buffer_resource;
    std::vector<std::unique_ptr<Resource>> thread_buffer_resources;
    ThreadLocal<Allocator>                 thread_allocators([&thread_buffer_resources]() {
        thread_buffer_resources.push_back(std::make_unique<Resource>());
        return Allocator(thread_buffer_resources.back().get());
    });

    std::vector<LineSegment> ordered_prims(m_primitives.size());

    BVHBuildNode    *root;
    std::atomic<int> total_nodes{ 0 };

    std::atomic<int> ordered_prims_offset{ 0 };
    root = build_recursive(thread_allocators, std::span<BVHPrimitive>(bvh_primitives),
                           total_nodes, ordered_prims_offset, ordered_prims);
    CHECK_EQ(ordered_prims_offset.load(), ordered_prims.size());
    m_primitives.swap(ordered_prims);

    bvh_primitives.resize(0);
    bvh_primitives.shrink_to_fit();
}

BVHBuildNode *BVH::build_recursive(ThreadLocal<Allocator>   &thread_allocators,
                                   std::span<BVHPrimitive>   bvh_primitives,
                                   std::atomic<int>         &total_nodes,
                                   std::atomic<int>         &ordered_prims_offset,
                                   std::vector<LineSegment> &ordered_prims) {
    DCHECK_NE(bvh_primitives.size(), 0);
    Allocator     alloc = thread_allocators.Get();
    BVHBuildNode *node  = alloc.new_object<BVHBuildNode>();
    ++total_nodes;

    BoundingBox<2> box;
    for (const auto &prim : bvh_primitives) {
        box.expand(prim.bounding_box);

        if (box.surface_area() == 0 or bvh_primitives.size() == 1) {
            // early termination if the box is degenerate or there is only one primitive
            int first_prim_offset = ordered_prims_offset.fetch_add(bvh_primitives.size());
            for (size_t i = 0; i < bvh_primitives.size(); ++i) {
                int index                            = bvh_primitives[i].primitive_index;
                ordered_prims[first_prim_offset + i] = m_primitives[index]; // copy primitive from original primitives to ordered_prims
            }
            node->init_leaf(first_prim_offset, bvh_primitives.size(), box);
            return node;
        }

        BoundingBox<2> centroid_boxs;
        for (const auto &prim : bvh_primitives) {
            centroid_boxs.expand(prim.bounding_box.centroid());
        }
        int dim = centroid_boxs.max_dimension();

        if (centroid_boxs.p_max[dim] == centroid_boxs.p_min[dim]) {
            // early termination if the box is degenerate
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
                } else {
                    constexpr int  n_buckets = 12;
                    BVHSplitBucket buckets[n_buckets];

                    // distribute primitives into buckets according to the centroid of the primitive,
                    // and count the number of primitives in each bucket
                    for (const auto &prim : bvh_primitives) {
                        int b = n_buckets * centroid_boxs.offset(prim.bounding_box.centroid())[dim];
                        if (b == n_buckets)
                            b = n_buckets - 1;
                        // check if the bucket index is valid
                        DCHECK_GE(b, 0);
                        DCHECK_LT(b, n_buckets);

                        buckets[b].count++;
                        buckets[b].box.expand(prim.bounding_box);
                    }

                    // compute the cost of each split
                    constexpr int  n_splits        = n_buckets - 1;
                    Float          costs[n_splits] = {}; // unnormalized costs
                    int            count_below     = 0;
                    BoundingBox<2> box_below;
                    // a forward pass to compute the cost of each split
                    for (int i = 0; i < n_splits; ++i) {
                        box_below.expand(buckets[i].box);
                        count_below += buckets[i].count;
                        costs[i] += count_below * box_below.surface_area();
                    }

                    // a backward pass to compute the cost of each split
                    int            count_above = 0;
                    BoundingBox<2> box_above;
                    for (int i = n_splits - 1; i >= 0; --i) {
                        box_above.expand(buckets[i + 1].box);
                        count_above += buckets[i + 1].count;
                        costs[i] += count_above * box_above.surface_area();
                    }

                    // find bucket to split at minimum cost
                    int   min_cost_split = -1;
                    Float min_cost       = FLT_MAX;
                    for (int i = 0; i < n_splits; ++i) {
                        if (costs[i] < min_cost) {
                            min_cost       = costs[i];
                            min_cost_split = i;
                        }
                    }

                    // compute leaf cost and SAH split cost for the chosen split
                    Float leaf_cost = bvh_primitives.size();
                    min_cost        = 1.f / 2.f + min_cost / box.surface_area(); // actual cost

                    if (bvh_primitives.size() > m_max_prims_in_node or min_cost < leaf_cost) {
                        auto mid_iter = std::partition(
                            bvh_primitives.begin(), bvh_primitives.end(),
                            [=](const BVHPrimitive &prim) {
                                int b = n_buckets * centroid_boxs.offset(prim.bounding_box.centroid())[dim];
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

        BVHBuildNode *left  = nullptr;
        BVHBuildNode *right = nullptr;
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
            left  = build_recursive(thread_allocators, bvh_primitives.subspan(0, mid),
                                    total_nodes, ordered_prims_offset, ordered_prims);
            right = build_recursive(thread_allocators, bvh_primitives.subspan(mid),
                                    total_nodes, ordered_prims_offset, ordered_prims);
        }

        node->init_interior(dim, left, right);
    }

    return node;
}

int BVH::flatten_bvh(BVHBuildNode *node, int *offset) {
    // use DFS to flatten the BVH
    LinearBVHNode *linear_node = &m_nodes[*offset];
    linear_node->box           = node->box;
    int node_offset            = (*offset)++;
    if (node->n_primitives > 0) { // leaf node
        CHECK(!node->left and !node->right);
        CHECK_LT(node->n_primitives, (1 << 16) - 1);
        linear_node->primitivesOffset = node->first_prim_offset;
        linear_node->n_primitives     = node->n_primitives;
    } else { // interior node
        linear_node->axis         = node->split_axis;
        linear_node->n_primitives = 0;
        flatten_bvh(node->left, offset);
        linear_node->secondChildOffset = flatten_bvh(node->right, offset);
    }
    return node_offset;
}

} // namespace gquery