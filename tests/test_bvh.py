from gquery.core.fwd import *
from gquery.shapes.bvh import *
import numpy as np


def test_bvh():
    vertices = Array2(np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]).T)
    indices = Array2i(np.array([[0, 1], [1, 2], [2, 3], [3, 0]]).T)
    bvh = BVH(vertices, indices)
    print(bvh.flat_tree)
    print(bvh.primitives)


if __name__ == "__main__":
    test_bvh()
