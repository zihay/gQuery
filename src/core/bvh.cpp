#include <core/bvh.h>

#include <memory_resource>

namespace gquery {

using Allocator = std::pmr::polymorphic_allocator<std::byte>;

BVH::BVH(const std::vector<LineSegment> &primitives, int max_prims_in_node, SplitMethod split_method)
    : m_primitives(std::move(primitives)), m_max_prims_in_node(max_prims_in_node), m_split_method(split_method) {
    std::pmr::monotonic_buffer_resource resource;
    Allocator allocator(&resource);
    
}

} // namespace gquery