#pragma once
#include <core/fwd.h>
#include <shapes/bounding_box.h>

namespace gquery {

/**
 * @brief Common interface for primitives of different dimensions.
 * 
 * This provides a CRTP-based interface for primitive types like LineSegment (2D)
 * and Triangle (3D) to ensure they have consistent functionality.
 */
template <size_t DIM, typename Derived>
struct Primitive {
    using Vector = Vector<DIM>;
    using BoundingBoxType = BoundingBox<DIM>;
    
    // Index for identifying the primitive
    int index;
    
    // Get bounding box (implemented by derived class)
    BoundingBoxType bounding_box() const {
        return static_cast<const Derived*>(this)->bounding_box();
    }
    
    // Get centroid (implemented by derived class)
    Vector centroid() const {
        return static_cast<const Derived*>(this)->centroid();
    }
};

// Forward declarations to be used in primitive type traits
struct LineSegment;
struct Triangle;

// Type trait to get the primitive type for a given dimension
template <size_t DIM>
struct PrimitiveType {
    // Default case results in a compile error
    static_assert(DIM == 2 || DIM == 3, "Only 2D and 3D primitives are supported");
};

// 2D specialization
template <>
struct PrimitiveType<2> {
    using Type = LineSegment;
};

// 3D specialization
template <>
struct PrimitiveType<3> {
    using Type = Triangle;
};

} // namespace gquery
