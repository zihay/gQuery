#pragma once
#include <fwd.h>

namespace gquery {

struct Triangle {
    using Vector = Vector3;
    Vector a;
    Vector b;
    Vector c;
};

} // namespace gquery