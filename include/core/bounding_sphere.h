#pragma once

#include <core/fwd.h>

namespace gquery {

template <size_t DIM>
struct BoundingSphere {
    using Vector = Eigen::Matrix<Float, DIM, 1>;

    BoundingSphere() : center(Vector::Zero()), radius(0.f) {}
    BoundingSphere(const Vector &center, Float radius) : center(center), radius(radius) {}

    Vector center;
    Float  radius;
};
} // namespace gquery