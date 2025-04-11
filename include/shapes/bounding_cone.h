#pragma once

#include <core/fwd.h>
#include <shapes/bounding_box.h>

#include <atomic>
#include <memory_resource>
#include <span>
#include <vector>

namespace gquery {

template <size_t DIM>
struct BoundingCone {
    using BoundingBox = BoundingBox<DIM>;

    BoundingCone() : axis(Vector<DIM>::Zero()), half_angle(M_PI), radius(0.f) {}
    BoundingCone(const Vector<DIM> &axis, Float half_angle, Float radius)
        : axis(axis), half_angle(half_angle), radius(radius) {}

    std::string __repr__() const {
        std::stringstream ss;
        ss << "BoundingCone(axis=[" << axis.transpose() << "], half_angle=" << half_angle << ", radius=" << radius << ")";
        return ss.str();
    }

    Vector<DIM> axis;
    Float       half_angle;
    Float       radius;
};

template <size_t DIM>
struct SoABoundingCone {
    std::vector<Vector<DIM>> axis;
    std::vector<Float>       half_angle;
    std::vector<Float>       radius;
};

} // namespace gquery
