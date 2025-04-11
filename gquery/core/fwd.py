from dataclasses import dataclass
from pathlib import Path

from drjit.auto.ad import TensorXf as Tensor
from drjit.auto.ad import PCG32
from drjit.auto.ad import Int32 as Int
from drjit.auto.ad import UInt32 as UInt
from drjit.auto.ad import Quaternion4f as Quaternion4
from drjit.auto.ad import Matrix2f as Matrix2
from drjit.auto.ad import Matrix3f as Matrix3
from drjit.auto.ad import Array4i as Array4i
from drjit.auto.ad import Array3i as Array3i
from drjit.auto.ad import Array2i as Array2i
from drjit.auto.ad import Array2f as Array2
from drjit.auto.ad import Array3f as Array3
from drjit.auto.ad import Array4f as Array4
from drjit.auto.ad import Bool
from drjit.auto.ad import Float32 as Float
from drjit.auto.ad import TensorXf
from drjit.auto.ad import TensorXi
import drjit as dr


BASE_DIR = Path(__file__).parent.parent.parent