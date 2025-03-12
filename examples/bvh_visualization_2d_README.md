# BVH Visualization Example

This example demonstrates how to visualize a 2D Bounding Volume Hierarchy (BVH) structure using Polyscope. The visualization provides an interactive way to explore the hierarchical nature of the BVH, showing the bounding boxes at different levels of the tree.

## Features

- Visualization of 2D line segments (primitives)
- Interactive exploration of the BVH levels
- Color-coded visualization of bounding boxes at each level
- Support for loading geometry from OBJ files
- Generation of random line segments for testing

## Requirements

- Python 3.6+
- polyscope
- numpy
- gquery (with nanobind bindings)

## Usage

```bash
# Basic usage (creates a simple square)
python bvh_visualization_2d.py

# Generate random line segments
python bvh_visualization_2d.py --random --num_segments 30 --seed 123

# Load geometry from an OBJ file
python bvh_visualization_2d.py --obj path/to/your/2d_model.obj
```

## Interactive Controls

When running the visualization, you can interact with the BVH using the following controls:

- **BVH Level slider**: Select which level of the BVH to display
- **Show All Levels button**: Display all levels of the BVH simultaneously with different colors
- **Clear All button**: Remove all BVH visualizations

## How It Works

1. The script constructs a BVH from line segments using the C++ implementation bound with nanobind
2. It then recursively traverses the BVH to identify the hierarchy levels
3. The bounding boxes at each level are visualized as rectangles
4. Colors are assigned based on the depth in the tree (root is green, deeper levels become more red)

## Understanding the Visualization

- **Black lines**: The original line segments
- **Colored rectangles**: The bounding boxes at the selected BVH level
- **Color gradient**: Green → Yellow → Red indicates shallow → deep levels in the BVH

## Example Output

When running the visualization, you will see the input geometry (line segments) in black, and the BVH bounding boxes in colors that correspond to their level in the hierarchy. 