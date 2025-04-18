#pragma once
#include <core/fwd.h>
#include <shapes/bounding_box.h>
#include <shapes/primitive.h>

namespace gquery {

struct LineSegment : public Primitive<2, LineSegment> {
    Vector2  a;
    Vector2  b;
    Vector2i indices;

    BoundingBox<2> bounding_box() const {
        BoundingBox<2> box;
        box.expand(a);
        box.expand(b);
        return box;
    }

    Vector2 centroid() const {
        return (a + b) * 0.5f;
    }

    ArrayX flatten() const {
        ArrayX ret(5);
        ret << a, b, Float(index);
        return ret;
    }
};

struct SoALineSegment {
    std::vector<Vector2> a;
    std::vector<Vector2> b;
    std::vector<int>     index;
};

} // namespace gquery
