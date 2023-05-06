import numpy as np


def write_numpy_file(x, filepath, overwrite=False):
    if overwrite:
        mode = "wb"
    else:
        mode = "ab"
    with open(filepath, mode) as f:
        f.write(x.tobytes())


def read_numpy_file(filepath, dtype=np.float64, shape=None, order="C"):
    with open(filepath, "rb") as f:
        tmp = f.read()
    out = np.frombuffer(tmp, dtype=dtype)
    if shape is not None:
        if order == "F":
            out = out.reshape(shape[::-1])
            out = out.T
        else:
            out = out.reshape(shape)
    return out
