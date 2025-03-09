#pragma once
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unique_ptr.h>

namespace nb = nanobind;
using namespace nb::literals;

#define PY_DECLARE(Name)          extern void python_export_##Name(nb::module_ &m)
#define PY_EXPORT(Name)           void python_export_##Name(nb::module_ &m)
#define PY_IMPORT(Name)           python_export_##Name(m)
#define PY_IMPORT_SUBMODULE(Name) python_export_##Name(Name)