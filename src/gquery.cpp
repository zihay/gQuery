#include <core/fwd.h>
#include <python/python.h>
#include <shapes/bounding_box.h>
#include <shapes/bvh.h>
#include <shapes/line_segment.h>
#include <shapes/triangle.h>

using namespace nanobind::literals;
namespace nb = nanobind;

NB_MODULE(gquery_ext, m) {
    m.attr("__name__")    = "gquery";
    m.attr("__version__") = "0.1.0";

    // Bind BoundingBox class
    nb::class_<gquery::BoundingBox<2>>(m, "BoundingBox")
        .def(nb::init<>())
        .def(nb::init<const Vector2 &, const Vector2 &>())
        .def_rw("p_min", &gquery::BoundingBox<2>::p_min)
        .def_rw("p_max", &gquery::BoundingBox<2>::p_max)
        .def("height", &gquery::BoundingBox<2>::height)
        .def("expand", nb::overload_cast<const Vector2 &>(&gquery::BoundingBox<2>::expand))
        .def("expand", nb::overload_cast<const gquery::BoundingBox<2> &>(&gquery::BoundingBox<2>::expand))
        .def("extent", &gquery::BoundingBox<2>::extent)
        .def("centroid", &gquery::BoundingBox<2>::centroid)
        .def("surface_area", &gquery::BoundingBox<2>::surface_area)
        .def("max_dimension", &gquery::BoundingBox<2>::max_dimension)
        .def("offset", &gquery::BoundingBox<2>::offset);

    nb::class_<gquery::BoundingBox<3>>(m, "BoundingBox3D")
        .def(nb::init<>())
        .def(nb::init<const Vector3 &, const Vector3 &>())
        .def_rw("p_min", &gquery::BoundingBox<3>::p_min)
        .def_rw("p_max", &gquery::BoundingBox<3>::p_max)
        .def("height", &gquery::BoundingBox<3>::height)
        .def("expand", nb::overload_cast<const Vector3 &>(&gquery::BoundingBox<3>::expand))
        .def("expand", nb::overload_cast<const gquery::BoundingBox<3> &>(&gquery::BoundingBox<3>::expand))
        .def("extent", &gquery::BoundingBox<3>::extent)
        .def("centroid", &gquery::BoundingBox<3>::centroid)
        .def("surface_area", &gquery::BoundingBox<3>::surface_area)
        .def("max_dimension", &gquery::BoundingBox<3>::max_dimension)
        .def("offset", &gquery::BoundingBox<3>::offset);

    // Bind LineSegment class
    nb::class_<gquery::LineSegment>(m, "LineSegment")
        .def(nb::init<>())
        .def_rw("a", &gquery::LineSegment::a)
        .def_rw("b", &gquery::LineSegment::b)
        .def_rw("index", &gquery::LineSegment::index);

    // Bind Triangle class
    nb::class_<gquery::Triangle>(m, "Triangle")
        .def(nb::init<>())
        .def_rw("a", &gquery::Triangle::a)
        .def_rw("b", &gquery::Triangle::b)
        .def_rw("c", &gquery::Triangle::c)
        .def_rw("index", &gquery::Triangle::index);

    // Bind BVHNode class
    nb::class_<gquery::BVHNode>(m, "BVHNode")
        .def(nb::init<>())
        .def_ro("box", &gquery::BVHNode::box)
        .def_ro("n_primitives", &gquery::BVHNode::n_primitives)
        .def_ro("primitives_offset", &gquery::BVHNode::primitivesOffset)
        .def_ro("second_child_offset", &gquery::BVHNode::secondChildOffset)
        .def_ro("axis", &gquery::BVHNode::axis);

    // Bind SoABoundingBox class
    nb::class_<gquery::SoABoundingBox<2>>(m, "SoABoundingBox")
        .def(nb::init<>())
        .def_rw("p_min", &gquery::SoABoundingBox<2>::p_min)
        .def_rw("p_max", &gquery::SoABoundingBox<2>::p_max);

    nb::class_<gquery::SoABoundingBox<3>>(m, "SoABoundingBox3D")
        .def(nb::init<>())
        .def_rw("p_min", &gquery::SoABoundingBox<3>::p_min)
        .def_rw("p_max", &gquery::SoABoundingBox<3>::p_max);

    // Bind SoABVHNode class
    nb::class_<gquery::SoABVHNode<2>>(m, "SoABVHNode")
        .def(nb::init<>())
        .def_rw("box", &gquery::SoABVHNode<2>::box)
        .def_rw("reference_offset", &gquery::SoABVHNode<2>::reference_offset)
        .def_rw("n_references", &gquery::SoABVHNode<2>::n_references)
        .def_rw("second_child_offset", &gquery::SoABVHNode<2>::second_child_offset);

    // Bind SoALineSegment class
    nb::class_<gquery::SoALineSegment>(m, "SoALineSegment")
        .def(nb::init<>())
        .def_rw("a", &gquery::SoALineSegment::a)
        .def_rw("b", &gquery::SoALineSegment::b)
        .def_rw("index", &gquery::SoALineSegment::index);

    // Bind SoABVH class
    nb::class_<gquery::SoABVH<2>>(m, "SoABVH")
        .def(nb::init<>())
        .def(nb::init<const gquery::BVH &>())
        .def_rw("flat_tree", &gquery::SoABVH<2>::flat_tree)
        .def_rw("primitives", &gquery::SoABVH<2>::primitives)
        .def_rw("sorted_primitives", &gquery::SoABVH<2>::sorted_primitives);

    // Enum for BVH split method
    nb::enum_<gquery::BVH::SplitMethod>(m, "BVHSplitMethod")
        .value("SAH", gquery::BVH::SplitMethod::SAH)
        .export_values();

    // Bind BVH class
    nb::class_<gquery::BVH>(m, "BVH")
        .def(nb::init<const std::vector<gquery::LineSegment> &, int, gquery::BVH::SplitMethod>(),
             "primitives"_a, "max_prims_in_node"_a = 10, "split_method"_a = gquery::BVH::SplitMethod::SAH)
        .def(nb::init<const std::vector<Vector2> &, const std::vector<Vector2i> &, int, gquery::BVH::SplitMethod>(),
             "vertices"_a, "indices"_a, "max_prims_in_node"_a = 10, "split_method"_a = gquery::BVH::SplitMethod::SAH)
        .def_ro("nodes", &gquery::BVH::m_nodes)
        .def_ro("primitives", &gquery::BVH::m_primitives)
        .def_ro("ordered_primitives", &gquery::BVH::m_ordered_prims)
        .def("to_soa", [](const gquery::BVH &bvh) {
            return gquery::SoABVH<2>(bvh);
        });
}