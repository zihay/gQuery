from core.fwd import *

@dataclass
class Mesh:
    vertices: Array3
    indices: Array3i
    
    bvh: BVH
    
    def __post_init__(self):
        pass
    