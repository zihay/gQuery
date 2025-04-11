#include <shapes/snch.h>

#include <span>
#include <unordered_map>
#include <unordered_set>

namespace gquery {

template <size_t DIM>
SNCH<DIM>::SNCH(const std::vector<Vector<DIM>> &vertices, const std::vector<Vectori<DIM>> &indices,
                int max_prims_in_node, SplitMethod split_method)
    : BVH<DIM>(vertices, indices, max_prims_in_node, split_method) {
    build_silhouettes(vertices, indices);
    build();
}

template <size_t DIM>
void SNCH<DIM>::build() {
    // First, get the number of nodes from the BVH
    size_t num_nodes = BVH<DIM>::m_nodes.size();

    // Resize our SNCH nodes array to match
    m_nodes.resize(num_nodes);

    // Copy and convert each node from BVH to SNCH
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto    &bvh_node  = BVH<DIM>::m_nodes[i];
        SNCHNode<DIM> &snch_node = m_nodes[i];
        // Copy the bounding box
        snch_node.box = bvh_node.box;

        // Initialize cone (can be refined later)
        snch_node.cone = BoundingCone<DIM>();

        // Copy the number of primitives
        snch_node.n_primitives = bvh_node.n_primitives;
        if (bvh_node.n_primitives > 0) {
            // Leaf node
            snch_node.primitives_offset = bvh_node.primitivesOffset;
        } else {
            // Interior node
            snch_node.second_child_offset = bvh_node.secondChildOffset;
        }

        // handle the silhouettes
        if (bvh_node.n_primitives > 0) { // leaf node
            size_t                     start = m_ordered_silhouettes.size();
            std::unordered_set<size_t> visited_silhouettes;

            for (size_t j = 0; j < bvh_node.n_primitives; ++j) {
                size_t primitive_idx = bvh_node.primitivesOffset + j;
                if constexpr (DIM == 2) {
                    LineSegment &primitive = BVH<DIM>::m_ordered_prims[primitive_idx];
                    for (size_t k = 0; k < 2; ++k) {
                        size_t            vertex_idx = primitive.indices[k];
                        SilhouetteVertex &silhouette = m_silhouettes[vertex_idx];
                        if (visited_silhouettes.insert(vertex_idx).second) {
                            m_ordered_silhouettes.push_back(silhouette);
                        }
                    }
                } else {
                    Triangle &primitive = BVH<DIM>::m_ordered_prims[primitive_idx];
                    for (size_t k = 0; k < 3; ++k) {
                        size_t i0 = primitive.indices[k];
                        size_t i1 = primitive.indices[(k + 1) % 3];
                        if (i0 > i1)
                            std::swap(i0, i1);
                        auto            edge       = std::make_pair(i0, i1);
                        size_t          edge_idx   = m_edge_map[edge];
                        SilhouetteEdge &silhouette = m_silhouettes[edge_idx];
                        if (visited_silhouettes.insert(edge_idx).second) {
                            m_ordered_silhouettes.push_back(silhouette);
                        }
                    }
                }
            }

            size_t end = m_ordered_silhouettes.size();

            snch_node.n_silhouettes     = end - start;
            snch_node.silhouette_offset = start;
        } else {
            snch_node.n_silhouettes     = 0;
            snch_node.silhouette_offset = 0;
        }
    }

    // Build the silhouette hierarchy
    if (m_nodes.size() > 0) {
        build_recursive(0, m_nodes.size());
    }
}

template <size_t DIM>
void SNCH<DIM>::build_recursive(size_t start, size_t end) {
    // compute the bounding cone axis
    auto       &root                   = m_nodes[start];
    Vector<DIM> centroid               = root.box.centroid();
    bool        any_silhouette         = false;
    bool        all_have_adjacent_face = true;
    // iterate over all leaf nodes
    for (size_t i = start; i < end; ++i) {
        auto &node = m_nodes[i];
        // node is a leaf node
        for (size_t j = 0; j < node.n_silhouettes; ++j) {
            int   idx        = node.silhouette_offset + j;
            auto &silhouette = m_silhouettes[idx];
            root.cone.axis += silhouette.normal();
            // TODO: need to check
            root.cone.radius = std::max(root.cone.radius, (silhouette.centroid() - centroid).norm());
            all_have_adjacent_face &= silhouette.has_face_0() && silhouette.has_face_1();
            any_silhouette = true;
        }
    }
    // compute the half angle of the cone
    if (!any_silhouette) { // no silhouette
        root.cone.half_angle = -M_PI;
    } else if (!all_have_adjacent_face) { // any silhouettes are boundary edges
        root.cone.half_angle = M_PI;
    } else {
        Float axis_norm = root.cone.axis.norm();
        if (axis_norm > Epsilon) {
            root.cone.axis /= axis_norm;
            root.cone.half_angle = 0.f;
            for (size_t i = start; i < end; ++i) {
                auto &node = m_nodes[i];
                // node is a leaf node
                for (size_t j = 0; j < node.n_silhouettes; ++j) {
                    int         idx        = node.silhouette_offset + j;
                    auto       &silhouette = m_silhouettes[idx];
                    Vector<DIM> n0         = silhouette.normal_0();
                    Vector<DIM> n1         = silhouette.normal_1();
                    Float       angle0     = std::acos(std::clamp(root.cone.axis.dot(n0), -1.f, 1.f));
                    Float       angle1     = std::acos(std::clamp(root.cone.axis.dot(n1), -1.f, 1.f));
                    root.cone.half_angle   = std::max(root.cone.half_angle, std::max(angle0, angle1));
                }
            }
        }
    }

    if (!root.is_leaf()) {
        // TODO: improve this
        build_recursive(start + 1, root.second_child_offset);
        build_recursive(start + root.second_child_offset, end);
    }
}

template <>
void SNCH<2>::build_silhouettes(const std::vector<Vector<2>>  &vertices,
                                const std::vector<Vectori<2>> &indices) {
    m_silhouettes.clear();
    m_silhouettes.resize(vertices.size());
    // for each edge
    for (const auto &face : indices) {
        size_t i0 = face[0];
        size_t i1 = face[1];
        auto   v0 = vertices[i0];
        auto   v1 = vertices[i1];

        // check the first end point of the line segment
        auto &silhouette0         = m_silhouettes[i0];
        silhouette0.m_vertices[1] = v0;
        silhouette0.m_vertices[2] = v1;
        silhouette0.m_indices[1]  = i0;
        silhouette0.m_indices[2]  = i1;
        silhouette0.m_prim_index  = i0;

        // check the second end point of the line segment
        auto &silhouette1         = m_silhouettes[i1];
        silhouette1.m_vertices[0] = v0;
        silhouette1.m_vertices[1] = v1;
        silhouette1.m_indices[0]  = i0;
        silhouette1.m_indices[1]  = i1;
        silhouette1.m_prim_index  = i1;
    }
}

template <>
void SNCH<3>::build_silhouettes(const std::vector<Vector<3>>  &vertices,
                                const std::vector<Vectori<3>> &indices) {
    m_edge_map.clear();

    size_t n_edges = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            size_t i0 = indices[i][j];
            size_t i1 = indices[i][(j + 1) % 3];
            // sort the vertices
            if (i0 > i1)
                std::swap(i0, i1);
            auto edge = std::make_pair(i0, i1);
            // if the edge is not in the map, add it
            if (m_edge_map.find(edge) == m_edge_map.end()) {
                m_edge_map[edge] = n_edges++;
            }
        }
    }

    m_silhouettes.clear();
    m_silhouettes.resize(n_edges);
    // for each triangle
    for (size_t i = 0; i < indices.size(); ++i) {
        auto face = indices[i];
        // for each edge of the triangle
        for (size_t j = 0; j < 3; ++j) {
            size_t i0 = face[j];
            size_t i1 = face[(j + 1) % 3];
            size_t i2 = face[(j + 2) % 3];
            auto   v0 = vertices[i0];
            auto   v1 = vertices[i1];
            auto   v2 = vertices[i2];
            // get the edge index
            bool swapped = false;
            if (i0 > i1) {
                std::swap(i0, i1);
                swapped = true;
            }
            auto edge = std::make_pair(i0, i1);

            size_t edge_index = m_edge_map[edge];
            // update the silhouette
            SilhouetteEdge &silhouette                 = m_silhouettes[edge_index];
            silhouette.m_vertices[1]                   = v0;
            silhouette.m_vertices[2]                   = v1;
            silhouette.m_vertices[swapped ? 3 : 0]     = v2;
            silhouette.m_indices[1]                    = i0;
            silhouette.m_indices[2]                    = i1;
            silhouette.m_indices[swapped ? 3 : 0]      = i2;
            silhouette.m_face_indices[swapped ? 1 : 0] = i;
            silhouette.m_prim_index                    = edge_index;
        }
    }
}

template <size_t DIM>
ArrayX SNCH<DIM>::primitive_data() const {
    return BVH<DIM>::primitive_data();
}

template <size_t DIM>
ArrayX SNCH<DIM>::silhouette_data() const {
    if (m_ordered_silhouettes.empty()) {
        return ArrayX();
    }
    size_t segment_size = m_ordered_silhouettes[0].flatten().size();
    ArrayX ret(m_ordered_silhouettes.size() * segment_size);
    for (size_t i = 0; i < m_ordered_silhouettes.size(); ++i) {
        ret.segment(i * segment_size, segment_size) = m_ordered_silhouettes[i].flatten();
    }
    return ret;
}

template <size_t DIM>
ArrayX SNCH<DIM>::raw_silhouette_data() const {
    if (m_silhouettes.empty()) {
        return ArrayX();
    }
    size_t segment_size = m_silhouettes[0].flatten().size();
    ArrayX ret(m_silhouettes.size() * segment_size);
    for (size_t i = 0; i < m_silhouettes.size(); ++i) {
        ret.segment(i * segment_size, segment_size) = m_silhouettes[i].flatten();
    }
    return ret;
}

template <size_t DIM>
ArrayX SNCH<DIM>::node_data() const {
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

// Explicit template instantiations
template class SNCH<2>;
template class SNCH<3>;

} // namespace gquery