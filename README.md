# gQuery: Fast CPU and GPU-Accelerated Geometry Queries

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

Install gQuery via pip:

```bash
pip install gquery
```

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
