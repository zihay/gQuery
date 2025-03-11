# Examples

This directory contains example scripts demonstrating various features of the library.

## 2D Geometry Query Example

The `polyline_intersect_2d.py` script demonstrates how to perform various geometric queries on 2D shapes with a visual interface using Polyscope.

### Features:

- Interactive 2D visualization of polyline shapes
- Support for multiple query types:
  - Ray intersection (fully implemented)
  - Closest point query (placeholder for future implementation)
  - Closest silhouette query (placeholder for future implementation)
- Slider controls for query parameters (ray origin/direction, point position)
- Visual representation of query results
- Support for different shape types (polygon, star)
- Proper handling of drjit variables

### How to Run:

```bash
python examples/polyline_intersect_2d.py
```

### Usage:

1. Select the query type from the dropdown menu
2. Use the sliders to adjust the query parameters:
   - For ray intersection: origin point and direction
   - For closest point: point position
   - For silhouette: direction vector
3. The query result (if any) will be shown in red, with additional information displayed in the UI
4. You can switch between different shape types and customize their parameters
5. Click "Update Visualization" to manually refresh if needed

### Implementation Details:

This example demonstrates:
- A generic, extensible class structure for different query types
- Converting between numpy arrays and drjit Arrays
- Using various query methods (intersect, etc.) to analyze geometric shapes
- Extracting and visualizing query data from drjit variables
- Creating an interactive UI with imgui through Polyscope

The example is designed to be easily extended with additional query types while properly handling the conversion between drjit variables and numpy arrays for visualization purposes. 