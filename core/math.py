from core.fwd import *


@dr.syntax
def in_range(x, a, b):
    return (x >= a) & (x <= b)
