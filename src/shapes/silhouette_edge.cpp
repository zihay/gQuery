#include <shapes/silhouette_edge.h>

namespace gquery {

std::vector<SilhouetteEdge> build_silhouette_edges(const std::vector<Vector<3>>  &vertices,
                                                   const std::vector<Vectori<3>> &indices) {
    std::unordered_map<std::pair<size_t, size_t>, size_t, PairHash> edge_map;

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
            if (edge_map.find(edge) == edge_map.end()) {
                edge_map[edge] = n_edges++;
            }
        }
    }

    std::vector<SilhouetteEdge> silhouettes;
    silhouettes.resize(n_edges);
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

            size_t edge_index = edge_map[edge];
            // update the silhouette
            SilhouetteEdge &silhouette                 = silhouettes[edge_index];
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
    return silhouettes;
}

ArrayX build_flat_silhouette_edges(const std::vector<Vector<3>>  &vertices,
                                   const std::vector<Vectori<3>> &indices) {
    auto   silhouettes  = build_silhouette_edges(vertices, indices);
    size_t segment_size = silhouettes[0].flatten().size();
    ArrayX ret(silhouettes.size() * segment_size);
    for (size_t i = 0; i < silhouettes.size(); ++i) {
        ret.segment(i * segment_size, segment_size) = silhouettes[i].flatten();
    }
    return ret;
}
} // namespace gquery