#pragma once
#include <core/fwd.h>

namespace gquery {

struct SilhouetteEdge {
    Vector3 m_vertices[4]; // v1 and v2 are the vertices of the edge
    size_t  m_indices[4];  // i1 and i2 are the indices of the edge
    size_t  m_face_indices[2];
    size_t  m_prim_index;

    Vector3 centroid() const {
        return 0.5 * (m_vertices[1] + m_vertices[2]);
    }

    bool has_face_0() const { return m_face_indices[0] != -1; }
    bool has_face_1() const { return m_face_indices[1] != -1; }

    Vector3 normal_0(bool normalize = true) const {
        const auto &v0 = m_vertices[1];
        const auto &v1 = m_vertices[2];
        const auto &v2 = m_vertices[0];
        Vector3     a  = v1 - v0;
        Vector3     b  = v2 - v0;
        Vector3     n  = a.cross(b);
        return normalize ? n.normalized() : n;
    }

    Vector3 normal_1(bool normalize = true) const {
        const auto &v0 = m_vertices[2];
        const auto &v1 = m_vertices[0];
        const auto &v2 = m_vertices[1];
        Vector3     a  = v1 - v0;
        Vector3     b  = v2 - v0;
        Vector3     n  = a.cross(b);
        return normalize ? n.normalized() : n;
    }

    Vector3 normal() const {
        Vector3 n = Vector3::Zero();
        if (has_face_0()) {
            n += normal_0(false);
        }
        if (has_face_1()) {
            n += normal_1(false);
        }
        return n.normalized();
    }

    ArrayX flatten() const {
        ArrayX ret(3 * 4 + 4 + 2 + 1);
        ret << m_vertices[0], m_vertices[1], m_vertices[2], m_vertices[3],
            Float(m_indices[0]), Float(m_indices[1]), Float(m_indices[2]), Float(m_indices[3]),
            Float(m_face_indices[0]), Float(m_face_indices[1]), Float(m_prim_index);
        return ret;
    }
};
} // namespace gquery