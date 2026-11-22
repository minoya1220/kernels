import pathlib
import torch

def ceil_div(a: int, b: int):
    return -(-a // b) 

# bitwise implementations of next pow 2 and previous pow 2 for fun
def ceil_pow2(x: int):
    return 1 << ((x - 1).bit_length() & -(x > 0))

def floor_pow2(x: int):
    return (x > 0) << ((x.bit_length() - 1) & -(x > 0)) # neg inputs ret 0 

def get_dir(file):
    return pathlib.Path(file).parent

