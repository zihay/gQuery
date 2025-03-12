#pragma once
#include <core/bounding_box.h>
#include <fwd.h>

namespace gquery {

struct Triangle {
    Vector3 a;
    Vector3 b;
    Vector3 c;

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