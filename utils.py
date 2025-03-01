#!/usr/bin/env python3

import torch
import pandas as pd
import os
import dotenv

dotenv.load_dotenv()

# General-purpose utility function to show the shape of anything for debugging.
# - If it's a tensor, the tensor shape
# - If it's a dict, the keys, and the shape of each value
# - If it's a list, the length, and the shape of the first element
# - If it's a tuple, the shapes of each element
# (all recursively)
# - For any other type, just say what type it is.
def show_shape(thing, prefix=""):
    if isinstance(thing, torch.Tensor):
        print(f"{prefix}torch.Tensor {thing.shape}:")
    elif isinstance(thing, pd.DataFrame):
        print(f"{prefix}pandas.DataFrame {thing.shape}, rows {len(thing.axes[0])}, cols {list(thing.axes[1])}:")
    elif isinstance(thing, pd.Series):
        print(f"{prefix}pandas.Series ({thing.name}) {thing.shape} of {thing.dtype}:")
    elif isinstance(thing, dict):
        print(f"{prefix}Dict:")
        for key, value in thing.items():
            show_shape(value, f"{prefix}[{key}] ")
    elif isinstance(thing, list):
        print(f"{prefix}List[{len(thing)}]:")
        if len(thing) > 0:
            show_shape(thing[0], f"{prefix}[0] ")
    elif isinstance(thing, tuple):
        print(f"{prefix}Tuple[{len(thing)}]:")
        for i, value in enumerate(thing[:5]):
            show_shape(value, f"{prefix}[{i}] ")
    else:
        print(f"{prefix}({type(thing)})")

def get_hf_token():
    return os.getenv("HF_TOKEN")
