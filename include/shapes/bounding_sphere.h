#pragma once

#include <core/fwd.h>

namespace gquery {

template <size_t DIM>
struct BoundingSphere {
    BoundingSphere() : center(Vector<DIM>::Zero()), radius(0.f) {}
    BoundingSphere(const Vector<DIM> &center, Float radius) : center(center), radius(radius) {}

    Vector<DIM> center;
    Float       radius;
};

template <size_t DIM>
struct SoABoundingSphere {
    std::vector<Vector<DIM>> center;
    std::vector<Float>       radius;
};

} // namespace gquery