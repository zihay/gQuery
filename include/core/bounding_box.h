#pragma once

#include <core/fwd.h>

namespace gquery {

template <size_t DIM>
struct BoundingBox {
    using Vector = Eigen::Matrix<Float, DIM, 1>;

    BoundingBox() : p_min(FLT_MAX), p_max(-FLT_MAX) {}
    BoundingBox(const Vector &p_min, const Vector &p_max) : p_min(p_min), p_max(p_max) {}

    Float width() const { return p_max[0] - p_min[0]; }
    Float height() const { return p_max[1] - p_min[1]; }

    void expand(const Vector &p) {
        Vector eps = Vector::Constant(Epsilon);
        p_min      = p_min.cwiseMin(p - eps);
        p_max      = p_max.cwiseMax(p + eps);
    }

    void expand(const BoundingBox &box) {
        p_min = p_min.cwiseMin(box.p_min);
        p_max = p_max.cwiseMax(box.p_max);
    }

    Vector extent() const { return p_max - p_min; }
    Vector center() const { return (p_min + p_max) / 2; }

    Vector p_min;
    Vector p_max;
};

} // namespace gquery
