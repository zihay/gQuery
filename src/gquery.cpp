#include <python/python.h>

NB_MODULE(gquery_ext, m) {
    m.attr("__name__")    = "gquery";
    m.attr("__version__") = "0.1.0";
}