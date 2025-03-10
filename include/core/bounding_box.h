#pragma once

#include <core/fwd.h>

#include <stdexcept>

namespace gquery {

template <size_t DIM>
struct BoundingBox {
    using Vector = Eigen::Matrix<Float, DIM, 1>;

    BoundingBox() : p_min(Vector::Constant(FLT_MAX)), p_max(Vector::Constant(-FLT_MAX)) {}
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
    Vector centroid() const { return (p_min + p_max) / 2; }

    // Calculates perimeter for 2D and surface area for 3D
    Float surface_area() const {
        Vector ext = extent();
        if constexpr (DIM == 2) {
            // For 2D, return perimeter
            return 2 * (ext[0] + ext[1]);
        } else if constexpr (DIM == 3) {
            // For 3D, return surface area
            return 2 * (ext[0] * ext[1] + ext[0] * ext[2] + ext[1] * ext[2]);
        } else {
            // Higher dimensions - use static_assert instead of runtime exception
            static_assert(DIM <= 3, "Surface area calculation not supported for dimensions higher than 3");
            return 0; // Will never reach here due to static_assert, but needed for compilation
        }
    }

    int max_dimension() const {
        Vector ext = extent();
        int    dim = 0;
        for (int i = 1; i < DIM; ++i) {
            if (ext[i] > ext[dim])
                dim = i;
        }
        return dim;
    }

    Vector offset(const Vector &p) const {
        Vector ext = extent();
        ext        = ext.cwiseMax(Epsilon);
        return (p - p_min).cwiseQuotient(ext);
    }

    Vector p_min;
    Vector p_max;
};

} // namespace gquery
