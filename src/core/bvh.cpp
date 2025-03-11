#include <core/bvh.h>
#include <util/check.h>

#include <memory_resource>
namespace gquery {

using Allocator = std::pmr::polymorphic_allocator<std::byte>;

struct BVHSplitBucket {
    BoundingBox<2> box;
    int            count;
};

struct SplitResult {
    int   axis;
    int   split;
    Float cost;
};

template <size_t DIM>
class SAHSplitMethod {
public:
    using BoundingBox = BoundingBox<DIM>;

    SplitResult find_best_split(const std::span<BVHPrimitive> bvh_primitives) {
        BoundingBox box;
        for (const auto &prim : bvh_primitives) {
            box.expand(prim.bounding_box);
        }

        BoundingBox centroids_box;
        for (const auto &prim : bvh_primitives) {
            centroids_box.expand(prim.bounding_box.centroid());
        }
        int dim = centroids_box.max_dimension(); // split axis

        // populate buckets
        BVHSplitBucket buckets[m_n_buckets];
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
        Float         costs[n_splits] = {}; // unnormalized costs
        int           count_below     = 0;
        BoundingBox   box_below;
        // a forward pass to compute the cost of each split
        for (int i = 0; i < n_splits; ++i) {
            box_below.expand(buckets[i].box);
            count_below += buckets[i].count;
            costs[i] += count_below * box_below.surface_area();
        }

        // a backward pass to compute the cost of each split
        int         count_above = 0;
        BoundingBox box_above;
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

        return SplitResult{ dim, min_cost_split, min_cost / box.surface_area() };
    }

public:
    constexpr static int m_n_buckets = 12;
};

BVH::BVH(const std::vector<Vector2> &vertices, const std::vector<Vector2i> &indices,
         int max_prims_in_node, SplitMethod split_method)
    : m_max_prims_in_node(max_prims_in_node), m_split_method(split_method) {
    m_primitives.reserve(indices.size());
    for (int i = 0; i < indices.size(); ++i) {
        auto index = indices[i];
        auto v0    = vertices[index[0]];
        auto v1    = vertices[index[1]];
        m_primitives.push_back(LineSegment{ v0, v1 });
    }

    build();
}

BVH::BVH(const std::vector<LineSegment> &primitives, int max_prims_in_node, SplitMethod split_method)
    : m_primitives(std::move(primitives)), m_max_prims_in_node(max_prims_in_node), m_split_method(split_method) {
    build();
}

void BVH::build() {
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
    m_ordered_prims = std::move(ordered_prims);

    bvh_primitives.resize(0);
    bvh_primitives.shrink_to_fit();

    m_nodes.resize(total_nodes.load());
    int offset = 0;
    flatten_bvh(root, &offset);
    CHECK_EQ(total_nodes.load(), offset);
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
                    mid = bvh_primitives.size() / 2;
                    std::nth_element(bvh_primitives.begin(), bvh_primitives.begin() + mid,
                                     bvh_primitives.end(),
                                     [dim](const BVHPrimitive &a, const BVHPrimitive &b) {
                                         return a.bounding_box.centroid()[dim] < b.bounding_box.centroid()[dim];
                                     });
                } else {
                    auto split_method = SAHSplitMethod<2>();
                    auto split_result = split_method.find_best_split(bvh_primitives);

                    int   n_buckets      = split_method.m_n_buckets;
                    int   min_cost_split = split_result.split;
                    Float min_cost       = split_result.cost;

                    // compute leaf cost and SAH split cost for the chosen split
                    Float leaf_cost = bvh_primitives.size();
                    min_cost        = 1.f / 2.f + min_cost / box.surface_area(); // actual cost

                    // perform SAH split if the cost is lower than the leaf cost
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
    BVHNode *linear_node = &m_nodes[*offset];
    linear_node->box     = node->box;
    int node_offset      = (*offset)++;
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

SoABVH<2> BVH::to_soa_bvh() const {
    SoABVH<2> soa_bvh(*this);
    return soa_bvh;
}

} // namespace gquery