#pragma once

#include <core/bounding_box.h>
#include <core/fwd.h>
#include <core/parallel.h>
#include <primitives/line_segment.h>

#include <atomic>
#include <memory_resource>
#include <span>
#include <vector>

namespace gquery {

template <size_t DIM>
struct BoundingCone {
    using Vector      = Eigen::Matrix<Float, DIM, 1>;
    using BoundingBox = BoundingBox<DIM>;

    BoundingCone() : axis(Vector::Zero()), half_angle(M_PI), radius(0.f) {}
    BoundingCone(const Vector &axis, Float half_angle, Float radius)
        : axis(axis), half_angle(half_angle), radius(radius) {}

    Vector axis;
    Float  half_angle;
    Float  radius;
};

template <size_t DIM>
struct SoABoundingCone {
    std::vector<Vector<DIM>> axis;
    std::vector<Float>       half_angle;
    std::vector<Float>       radius;
};

} // namespace gquery
