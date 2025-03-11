#pragma once
#include <fwd.h>

namespace gquery {

struct Triangle {
    using Vector = Vector3;
    Vector a;
    Vector b;
    Vector c;
};

struct SoATriangle {
    std::vector<Vector3> a;
    std::vector<Vector3> b;
    std::vector<Vector3> c;
};

} // namespace gquery