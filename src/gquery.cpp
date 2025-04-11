#include <core/fwd.h>
#include <python/python.h>
#include <shapes/bounding_box.h>
#include <shapes/bounding_cone.h>
#include <shapes/bvh.h>
#include <shapes/line_segment.h>
#include <shapes/snch.h>
#include <shapes/triangle.h>

#include <sstream>

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
        .def("offset", &gquery::BoundingBox<2>::offset)
        .def("__repr__", &gquery::BoundingBox<2>::__repr__);

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
        .def("offset", &gquery::BoundingBox<3>::offset)
        .def("__repr__", &gquery::BoundingBox<3>::__repr__);

    // Bind BoundingCone class
    nb::class_<gquery::BoundingCone<2>>(m, "BoundingCone")
        .def(nb::init<>())
        .def(nb::init<const Vector2 &, float, float>())
        .def_rw("axis", &gquery::BoundingCone<2>::axis)
        .def_rw("half_angle", &gquery::BoundingCone<2>::half_angle)
        .def_rw("radius", &gquery::BoundingCone<2>::radius)
        .def("__repr__", &gquery::BoundingCone<2>::__repr__);

    // Bind BoundingCone3D class
    nb::class_<gquery::BoundingCone<3>>(m, "BoundingCone3D")
        .def(nb::init<>())
        .def(nb::init<const Vector3 &, float, float>())
        .def_rw("axis", &gquery::BoundingCone<3>::axis)
        .def_rw("half_angle", &gquery::BoundingCone<3>::half_angle)
        .def_rw("radius", &gquery::BoundingCone<3>::radius)
        .def("__repr__", &gquery::BoundingCone<3>::__repr__);

    // Bind LineSegment class
    nb::class_<gquery::LineSegment>(m, "LineSegment")
        .def(nb::init<>())
        .def_rw("a", &gquery::LineSegment::a)
        .def_rw("b", &gquery::LineSegment::b)
        .def_rw("index", &gquery::LineSegment::index);

    // Bind Triangle class
    nb::class_<gquery::Triangle>(m, "Triangle")
        .def(nb::init<>())
        .def("flatten", &gquery::Triangle::flatten)
        .def_rw("a", &gquery::Triangle::a)
        .def_rw("b", &gquery::Triangle::b)
        .def_rw("c", &gquery::Triangle::c)
        .def_rw("index", &gquery::Triangle::index);

    // Bind SilhouetteVertex class
    nb::class_<gquery::SilhouetteVertex>(m, "SilhouetteVertex")
        .def(nb::init<>())
        .def_prop_ro("vertices",
                     [](const gquery::SilhouetteVertex &sv) {
                         std::vector<Vector2> vertices = { sv.m_vertices[0], sv.m_vertices[1], sv.m_vertices[2] };
                         return vertices;
                     })
        .def_prop_ro("indices",
                     [](const gquery::SilhouetteVertex &sv) {
                         std::vector<size_t> indices = { sv.m_indices[0], sv.m_indices[1], sv.m_indices[2] };
                         return indices;
                     })
        .def_rw("prim_index", &gquery::SilhouetteVertex::m_prim_index);

    // Bind SilhouetteEdge class
    nb::class_<gquery::SilhouetteEdge>(m, "SilhouetteEdge")
        .def(nb::init<>())
        .def_prop_ro("vertices",
                     [](const gquery::SilhouetteEdge &se) {
                         std::vector<Vector3> vertices = { se.m_vertices[0], se.m_vertices[1], se.m_vertices[2], se.m_vertices[3] };
                         return vertices;
                     })
        .def_prop_ro("indices",
                     [](const gquery::SilhouetteEdge &se) {
                         std::vector<size_t> indices = { se.m_indices[0], se.m_indices[1], se.m_indices[2], se.m_indices[3] };
                         return indices;
                     })
        .def_prop_ro("face_indices",
                     [](const gquery::SilhouetteEdge &se) {
                         std::vector<size_t> face_indices = { se.m_face_indices[0], se.m_face_indices[1] };
                         return face_indices;
                     })
        .def_rw("prim_index", &gquery::SilhouetteEdge::m_prim_index);

    // Bind BVHNode class (2D version)
    nb::class_<gquery::BVHNode<2>>(m, "BVHNode")
        .def(nb::init<>())
        .def("flatten", &gquery::BVHNode<2>::flatten)
        .def_ro("box", &gquery::BVHNode<2>::box)
        .def_ro("n_primitives", &gquery::BVHNode<2>::n_primitives)
        .def_ro("primitives_offset", &gquery::BVHNode<2>::primitivesOffset)
        .def_ro("second_child_offset", &gquery::BVHNode<2>::secondChildOffset)
        .def_ro("axis", &gquery::BVHNode<2>::axis)
        .def("__repr__", &gquery::BVHNode<2>::__repr__);

    // Bind BVHNode class (3D version)
    nb::class_<gquery::BVHNode<3>>(m, "BVHNode3D")
        .def(nb::init<>())
        .def("flatten", &gquery::BVHNode<3>::flatten)
        .def_ro("box", &gquery::BVHNode<3>::box)
        .def_ro("n_primitives", &gquery::BVHNode<3>::n_primitives)
        .def_ro("primitives_offset", &gquery::BVHNode<3>::primitivesOffset)
        .def_ro("second_child_offset", &gquery::BVHNode<3>::secondChildOffset)
        .def_ro("axis", &gquery::BVHNode<3>::axis)
        .def("__repr__", &gquery::BVHNode<3>::__repr__);

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
        .def(nb::init<const gquery::BVH<2> &>())
        .def_rw("flat_tree", &gquery::SoABVH<2>::flat_tree)
        .def_rw("primitives", &gquery::SoABVH<2>::primitives)
        .def_rw("sorted_primitives", &gquery::SoABVH<2>::sorted_primitives);

    // Enum for BVH split method
    nb::enum_<gquery::SplitMethod>(m, "BVHSplitMethod")
        .value("SAH", gquery::SplitMethod::SAH)
        .export_values();

    // Bind BVH class (2D version)
    nb::class_<gquery::BVH<2>>(m, "BVH")
        .def(nb::init<const std::vector<gquery::LineSegment> &, int, gquery::SplitMethod>(),
             "primitives"_a, "max_prims_in_node"_a = 10, "split_method"_a = gquery::SplitMethod::SAH)
        .def(nb::init<const std::vector<Vector2> &, const std::vector<Vector2i> &, int, gquery::SplitMethod>(),
             "vertices"_a, "indices"_a, "max_prims_in_node"_a = 10, "split_method"_a = gquery::SplitMethod::SAH)
        .def("primitive_data", &gquery::BVH<2>::primitive_data)
        .def("node_data", &gquery::BVH<2>::node_data)
        .def_ro("nodes", &gquery::BVH<2>::m_nodes)
        .def_ro("primitives", &gquery::BVH<2>::m_primitives)
        .def_ro("ordered_primitives", &gquery::BVH<2>::m_ordered_prims);

    // Bind BVH class (3D version)
    nb::class_<gquery::BVH<3>>(m, "BVH3D")
        .def(nb::init<const std::vector<gquery::Triangle> &, int, gquery::SplitMethod>(),
             "primitives"_a, "max_prims_in_node"_a = 10, "split_method"_a = gquery::SplitMethod::SAH)
        .def(nb::init<const std::vector<Vector3> &, const std::vector<Vector3i> &, int, gquery::SplitMethod>(),
             "vertices"_a, "indices"_a, "max_prims_in_node"_a = 10, "split_method"_a = gquery::SplitMethod::SAH)
        .def("primitive_data", &gquery::BVH<3>::primitive_data)
        .def("node_data", &gquery::BVH<3>::node_data)
        .def_ro("nodes", &gquery::BVH<3>::m_nodes)
        .def_ro("primitives", &gquery::BVH<3>::m_primitives)
        .def_ro("ordered_primitives", &gquery::BVH<3>::m_ordered_prims);

    // Bind SNCHNode class
    nb::class_<gquery::SNCHNode<2>>(m, "SNCHNode")
        .def(nb::init<>())
        .def_ro("box", &gquery::SNCHNode<2>::box)
        .def_ro("cone", &gquery::SNCHNode<2>::cone)
        .def_ro("primitives_offset", &gquery::SNCHNode<2>::primitives_offset)
        .def_ro("second_child_offset", &gquery::SNCHNode<2>::second_child_offset)
        .def_ro("n_primitives", &gquery::SNCHNode<2>::n_primitives)
        .def_ro("silhouette_offset", &gquery::SNCHNode<2>::silhouette_offset)
        .def_ro("n_silhouettes", &gquery::SNCHNode<2>::n_silhouettes)
        .def("__repr__", &gquery::SNCHNode<2>::__repr__);

    nb::class_<gquery::SNCHNode<3>>(m, "SNCHNode3D")
        .def(nb::init<>())
        .def_ro("box", &gquery::SNCHNode<3>::box)
        .def_ro("cone", &gquery::SNCHNode<3>::cone)
        .def_ro("primitives_offset", &gquery::SNCHNode<3>::primitives_offset)
        .def_ro("second_child_offset", &gquery::SNCHNode<3>::second_child_offset)
        .def_ro("n_primitives", &gquery::SNCHNode<3>::n_primitives)
        .def("__repr__", &gquery::SNCHNode<3>::__repr__);

    // Bind SNCH
    nb::class_<gquery::SNCH<2>>(m, "SNCH")
        .def(nb::init<const std::vector<Vector2> &, const std::vector<Vector2i> &, int, gquery::SplitMethod>(),
             "vertices"_a, "indices"_a, "max_prims_in_node"_a = 10, "split_method"_a = gquery::SplitMethod::SAH)
        .def("primitive_data", &gquery::SNCH<2>::primitive_data)
        .def("silhouette_data", &gquery::SNCH<2>::silhouette_data)
        .def("node_data", &gquery::SNCH<2>::node_data)
        .def_ro("flat_tree", &gquery::SNCH<2>::m_nodes)
        .def_ro("primitives", &gquery::SNCH<2>::m_primitives)
        .def_ro("silhouettes", &gquery::SNCH<2>::m_silhouettes);

    // Bind SNCH class (3D version)
    nb::class_<gquery::SNCH<3>>(m, "SNCH3D")
        .def(nb::init<const std::vector<Vector3> &, const std::vector<Vector3i> &, int, gquery::SplitMethod>(),
             "vertices"_a, "indices"_a, "max_prims_in_node"_a = 10, "split_method"_a = gquery::SplitMethod::SAH)
        .def("primitive_data", &gquery::SNCH<3>::primitive_data)
        .def("silhouette_data", &gquery::SNCH<3>::silhouette_data)
        .def("node_data", &gquery::SNCH<3>::node_data)
        .def_ro("flat_tree", &gquery::SNCH<3>::m_nodes)
        .def_ro("primitives", &gquery::SNCH<3>::m_primitives)
        .def_ro("silhouettes", &gquery::SNCH<3>::m_silhouettes);
}