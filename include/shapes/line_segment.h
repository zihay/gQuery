#pragma once
#include <core/fwd.h>
#include <shapes/bounding_box.h>
namespace gquery {

struct LineSegment {
    using Vector = Vector2;
    Vector a;
    Vector b;
    int    index;

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
    std::vector<int>     index;
};

} // namespace gquery
