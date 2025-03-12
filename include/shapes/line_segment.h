#pragma once
#include <shapes/bounding_box.h>
#include <core/fwd.h>
namespace gquery {

struct LineSegment {
    using Vector = Vector2;
    Vector a;
    Vector b;

    BoundingBox<2> bounding_box() const {
        BoundingBox<2> box;
        box.expand(a);
        box.expand(b);
        return box;
    }
};

struct SoALineSegment {
    std::vector<Vector2> a;
    std::vector<Vector2> b;
};

} // namespace gquery
