#pragma once
#include <core/fwd.h>
#include <shapes/bounding_box.h>
#include <shapes/primitive.h>

namespace gquery {

struct Triangle : public Primitive<3, Triangle> {
    Vector3 a;
    Vector3 b;
    Vector3 c;
    Vector3i indices;

    BoundingBox<3> bounding_box() const {
        BoundingBox<3> box;
        box.expand(a);
        box.expand(b);
        box.expand(c);
        return box;
    }
    
    Vector3 centroid() const {
        return (a + b + c) * (1.0f/3.0f);
    }

    ArrayX flatten() const {
        ArrayX ret(10);
        ret << a, b, c, Float(index);
        return ret;
    }
};

struct SoATriangle {
    std::vector<Vector3> a;
    std::vector<Vector3> b;
    std::vector<Vector3> c;
    std::vector<int>     index;
};

} // namespace gquery