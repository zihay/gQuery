#pragma once
#include <core/bounding_box.h>
#include <core/fwd.h>
namespace gquery {

struct LineSegment {
    using Vector = Vector2;
    Vector a;
    Vector b;

    BoundingBox<2> bounding_box() const {
        return BoundingBox<2>(a, b);
    }
};

} // namespace gquery
