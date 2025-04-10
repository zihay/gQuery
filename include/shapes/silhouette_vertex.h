#pragma once
#include <shapes/bounding_box.h>

namespace gquery {

struct SilhouetteVertex {
    Vector2 m_vertices[3];
    size_t  m_indices[3];
    size_t  m_prim_index;

    Vector2 centroid() const {
        return m_vertices[1];
    }

    bool has_face_0() const { return m_indices[0] != -1; }
    bool has_face_1() const { return m_indices[2] != -1; }

    // compute the normal of the edge
    Vector2 normal_0(bool normalize = true) const {
        const auto &v0 = m_vertices[0];
        const auto &v1 = m_vertices[1];
        Vector2     v  = v1 - v0;
        Vector2     n  = Vector2(v[1], -v[0]);
        return normalize ? n.normalized() : n;
    }

    // compute the normal of the edge
    Vector2 normal_1(bool normalize = true) const {
        const auto &v0 = m_vertices[1];
        const auto &v1 = m_vertices[2];
        Vector2     v  = v1 - v0;
        Vector2     n  = Vector2(v[1], -v[0]);
        return normalize ? n.normalized() : n;
    }

    // compute the normal of the vertex
    Vector2 normal() const {
        Vector2 n = Vector2::Zero();
        if (has_face_0()) {
            n += normal_0(false);
        }
        if (has_face_1()) {
            n += normal_1(false);
        }
        return n.normalized();
    }

    ArrayX flatten() const {
        ArrayX ret(2 * 3 + 3 + 1);
        ret << m_vertices[0], m_vertices[1], m_vertices[2],
            Float(m_indices[0]), Float(m_indices[1]), Float(m_indices[2]),
            Float(m_prim_index);
        return ret;
    }
};

} // namespace gquery