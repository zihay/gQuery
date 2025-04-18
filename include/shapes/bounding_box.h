#pragma once

#include <core/fwd.h>

#include <stdexcept>

namespace gquery {

template <size_t DIM>
struct BoundingBox {
    BoundingBox() : p_min(Vector<DIM>::Constant(FLT_MAX)), p_max(Vector<DIM>::Constant(-FLT_MAX)) {}
    BoundingBox(const Vector<DIM> &p) {
        Vector<DIM> eps = Vector<DIM>::Constant(Epsilon);
        p_min           = p - eps;
        p_max           = p + eps;
    }
    BoundingBox(const Vector<DIM> &p_min, const Vector<DIM> &p_max) : p_min(p_min), p_max(p_max) {}

    Float width() const { return p_max[0] - p_min[0]; }
    Float height() const { return p_max[1] - p_min[1]; }

    void expand(const Vector<DIM> &p) {
        Vector<DIM> eps = Vector<DIM>::Constant(Epsilon);
        p_min           = p_min.cwiseMin(p - eps);
        p_max           = p_max.cwiseMax(p + eps);
    }

    void expand(const BoundingBox<DIM> &box) {
        p_min = p_min.cwiseMin(box.p_min);
        p_max = p_max.cwiseMax(box.p_max);
    }

    Vector<DIM> extent() const { return p_max - p_min; }
    Vector<DIM> centroid() const { return (p_min + p_max) * 0.5f; }

    // Calculates perimeter for 2D and surface area for 3D
    Float surface_area() const {
        Vector<DIM> ext = extent();
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
        int   index;
        Float max_length = (p_max - p_min).maxCoeff(&index);
        return index;
    }

    Vector<DIM> offset(const Vector<DIM> &p) const {
        Vector<DIM> ext = extent();
        ext             = ext.cwiseMax(Epsilon);
        return (p - p_min).cwiseQuotient(ext);
    }

    std::string __repr__() const {
        std::stringstream ss;
        ss << "BoundingBox(p_min=[" << p_min.transpose() << "], p_max=[" << p_max.transpose() << "])";
        return ss.str();
    }

    Vector<DIM> p_min;
    Vector<DIM> p_max;
};

template <size_t DIM>
struct SoABoundingBox {
    std::vector<Vector<DIM>> p_min;
    std::vector<Vector<DIM>> p_max;
};

} // namespace gquery
