#pragma once
#include <Eigen/Dense>
#include <memory>

// Specify calculation precision
#define DOUBLE_PRECISION

#ifdef DOUBLE_PRECISION
typedef float Float; // TODO
#define Epsilon 1e-9
#define FilterEpsilon 1e-12
#undef M_PI
#define M_PI 3.14159265358979323846
#define INV_PI 0.31830988618379067154
#define INV_TWOPI 0.15915494309189534561
#define INV_FOURPI 0.07957747154594766788
#else
typedef float Float;
#define Epsilon 1e-4f
#define M_PI 3.14159265358979323846f
#define INV_PI 0.31830988618379067154f
#define INV_TWOPI 0.15915494309189534561f
#define INV_FOURPI 0.07957747154594766788f
#endif

#undef FLT_MAX
auto const FLT_MAX = std::numeric_limits<float>::max();

auto const RAY_OFFSET = 1e-4f;

// #define AD
template <size_t DIM>
using Vector = Eigen::Matrix<Float, DIM, 1>;

typedef Eigen::Matrix<Float, 2, 1> Vector2;
typedef Eigen::Matrix<float, 2, 1> Vector2f;
typedef Eigen::Matrix<double, 2, 1> Vector2d;
typedef Eigen::Matrix<int, 2, 1> Vector2i;

typedef Eigen::Array<Float, 2, 1> Array2;
typedef Eigen::Array<float, 2, 1> Array2f;
typedef Eigen::Array<double, 2, 1> Array2d;
typedef Eigen::Array<int, 2, 1> Array2i;

typedef Eigen::Matrix<Float, 3, 1> Vector3;
typedef Eigen::Matrix<float, 3, 1> Vector3f;
typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef Eigen::Matrix<int, 3, 1> Vector3i;

typedef Eigen::Array<Float, 3, 1> Array;
typedef Eigen::Array<Float, 3, 1> Array3;
typedef Eigen::Array<float, 3, 1> Array3f;
typedef Eigen::Array<double, 3, 1> Array3d;
typedef Eigen::Array<int, 3, 1> Array3i;

typedef Eigen::Matrix<Float, 4, 1> Vector4;
typedef Eigen::Matrix<float, 4, 1> Vector4f;
typedef Eigen::Matrix<double, 4, 1> Vector4d;
typedef Eigen::Matrix<int, 4, 1> Vector4i;

typedef Eigen::Array<Float, 4, 1> Array4;
typedef Eigen::Array<float, 4, 1> Array4f;
typedef Eigen::Array<double, 4, 1> Array4d;
typedef Eigen::Array<int, 4, 1> Array4i;

typedef Eigen::Matrix<Float, 2, 2> Matrix2x2;
typedef Eigen::Matrix<Float, 3, 3> Matrix3x3;
typedef Eigen::Matrix<float, 3, 3> Matrix3x3f;
typedef Eigen::Matrix<double, 3, 3> Matrix3x3d;
typedef Eigen::Matrix<Float, 4, 4> Matrix4x4;
typedef Eigen::Matrix<float, 4, 4> Matrix4x4f;
typedef Eigen::Matrix<double, 4, 4> Matrix4x4d;
typedef Eigen::Matrix<Float, 2, 4> Matrix2x4;
typedef Eigen::Matrix<Float, -1, 3> MatrixX3;
typedef Eigen::Matrix<int, -1, 3> MatrixX3i;
template <int N>
using MatrixNxN = Eigen::Matrix<Float, N, N>;
template <int N>
using VectorN = Eigen::Matrix<Float, N, 1>;
// ArrayX
typedef Eigen::Array<Float, Eigen::Dynamic, 1> ArrayX;

template <int N>
using ArrayXN = Eigen::Array<Float, N, 1>;

using ArrayX3d = Eigen::Array<Float, Eigen::Dynamic, 3, Eigen::RowMajor>;
// VectorX2
typedef Eigen::Matrix<Float, Eigen::Dynamic, 2> VectorX2;

using MatrixX = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixX2 = Eigen::Matrix<Float, Eigen::Dynamic, 2, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXi = Eigen::MatrixXi;

using Spectrum = Eigen::Array<Float, 3, 1>;

typedef Eigen::ArrayXd ArrayXd;

// template <typename T>
// using ref = std::shared_ptr<T>;
// template <typename T>
// using ref = std::reference_wrapper<T>;

// const ref
template <typename T>
using cref = std::reference_wrapper<const T>;

// square
template <typename T>
inline T square(T x) { return x * x; }

// frac
template <typename T>
inline T frac(T x) { return x - std::floor(x); }

// ============== Math =======================
template <typename T>
inline T sqrt(T x);

// scalar
template <>
inline Float sqrt<Float>(Float x)
{
    return std::sqrt(x);
}
// array
template <typename T, int n, int m>
T sqrt(const Eigen::Array<T, n, m> &x)
{
    return x.sqrt();
}

template <typename T>
inline bool isnan(const T &x);

// scalar
template <>
inline bool isnan<Float>(const Float &x)
{
    return std::isnan(x);
}

// array
template <typename T, int n, int m>
bool isnan(const Eigen::Array<T, n, m> &x)
{
    return x.isNaN().all();
}

template <typename T>
T lerp(T a, T b, Float t)
{
    return a + t * (b - a);
}

template <typename T>
inline T sign(T x)
{
    return (x > 0) - (x < 0);
}

// cross
template <typename T>
inline Vector3 cross(const Eigen::Matrix<T, 3, 1> &a, const Eigen::Matrix<T, 3, 1> &b)
{
    return a.cross(b);
}

// cross 2d
template <typename T>
inline T cross(const Eigen::Matrix<T, 2, 1> &a, const Eigen::Matrix<T, 2, 1> &b)
{
    return a.x() * b.y() - a.y() * b.x();
}