#pragma once
#include <core/fwd.h>
#include <shapes/bounding_box.h>

namespace gquery {

struct Triangle {
    Vector3 a;
    Vector3 b;
    Vector3 c;
    int     index;

    BoundingBox<3> bounding_box() const {
        BoundingBox<3> box;
        box.expand(a);
        box.expand(b);
        box.expand(c);
        return box;
    }
};

struct SoATriangle {
    std::vector<Vector3> a;
    std::vector<Vector3> b;
    std::vector<Vector3> c;
};

} // namespace gquery