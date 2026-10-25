import torch
import csv
import itertools
import pathlib
from typing import NamedTuple

class Param(NamedTuple):
    fallback: int
    options: tuple[int, ...]

class Tuner:
    def __init__(self, path, search_space):
        self.path = pathlib.Path(path)
        self.search_space = search_space
        self.configs = {}

        if not self.path.exists():
            return
            
        with self.path.open(newline="") as f:
            for row in csv.DictReader(f):
                row = {k: int(v) for }
                
        
    def get_config(dir=None):
        pass
        
    def save_config(dir=None):
        pass
    
    def gen_config(dir=None, autosave=True):
        pass