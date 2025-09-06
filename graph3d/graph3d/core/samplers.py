import numpy as np

def grid_xy(xs, ys):
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return X, Y

def grid_uv(us, vs):
    U, V = np.meshgrid(us, vs, indexing="xy")
    return U, V

def lin_t(t0, t1, n):
    return np.linspace(t0, t1, n)
