# gQuery: Fast CPU and GPU-Accelerated Geometry Queries

[![Test Build](https://github.com/zihay/gquery/actions/workflows/test_build.yml/badge.svg)](https://github.com/zihay/gquery/actions/workflows/test_build.yml)
[![Build and Release](https://github.com/zihay/gquery/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/zihay/gquery/actions/workflows/build_wheels.yml)

**gQuery** is a high-performance Python library built with [DrJit](https://github.com/mitsuba-renderer/drjit), designed to efficiently handle geometry queries using Bounding Volume Hierarchies (BVH) and Spatialized Normal Cone Hierarchies (SNCH). It provides fast, accurate solutions tailored for computer graphics and computational geometry applications, including:

- **Closest Point Queries**
- **Ray Intersection Queries**
- **Closest Silhouette Queries**

Leveraging DrJit's parallel execution capabilities on both CPU and GPU, gQuery significantly speeds up spatial queries, enabling real-time and interactive workflows in computer graphics, simulations, and more.

---

## Key Features

- **DrJit Backend**: Efficient parallel computations seamlessly supported on CPU and GPU.
- **Efficient Hierarchy Implementation**: Optimized BVH and SNCH construction and traversal algorithms.
- **Python API**: User-friendly Python interface, perfect for rapid prototyping and integration into existing workflows.

---

## Installation

### Install from GitHub Releases (recommended)

You can install the pre-built wheels from the latest GitHub release:

```bash
# Install the latest release
pip install https://github.com/zihay/gquery/releases/download/v0.1.0/gquery-0.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Or for macOS
pip install https://github.com/zihay/gquery/releases/download/v0.1.0/gquery-0.1.0-cp310-cp310-macosx_10_14_x86_64.whl
```

Replace the URL with the specific wheel file that matches your Python version and platform.

### Install from Source

You can also install directly from the repository:

```bash
# Install the latest main branch
pip install git+https://github.com/zihay/gquery.git

# Install a specific branch
pip install git+https://github.com/zihay/gquery.git@develop

# Install a specific tag/release
pip install git+https://github.com/zihay/gquery.git@v0.1.0
```

Note that installing from source requires you to have all the build dependencies installed, including:
- C++ compiler with C++20 support
- CMake 3.15 or newer
- Eigen3 library

---

## Usage Example

Here's how you can perform a simple closest point query:

```python
import gquery

# Load geometry and build query structure
scene = gquery.Scene(device="gpu")  # or device="cpu"
scene.load_geometry("scene.obj")
scene.build()

# Perform a closest point query
query_point = [1.0, 2.0, 3.0]
result = scene.closest_point(query_point)

print(f"Closest point: {result.point}")
print(f"Distance: {result.distance}")
```

---

## Requirements

- CUDA-compatible GPU (optional, for GPU acceleration)
- CUDA Toolkit 11.0 or newer (for GPU execution)
- Python 3.7 or newer
- [DrJit](https://github.com/mitsuba-renderer/drjit)

---

## License

gQuery is distributed under the MIT license. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions and feedback are welcome! Please submit pull requests or open issues on GitHub.
