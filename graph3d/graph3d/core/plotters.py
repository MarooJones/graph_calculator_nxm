import numpy as np
import plotly.graph_objects as go

def surface_from_xyz(X, Y, Z, x=None, y=None, parametric=False):
    if x is None: x = X
    if y is None: y = Y
    surf = go.Surface(x=x, y=y, z=Z, colorscale="Viridis", showscale=True, opacity=0.96)
    fig = go.Figure(surf)
    return fig

def _wireframe_lines(X, Y, Z):
    lines = []
    for i in range(X.shape[0]):
        lines.append(go.Scatter3d(x=X[i,:], y=Y[i,:], z=Z[i,:], mode="lines", line=dict(width=1)))
    for j in range(X.shape[1]):
        lines.append(go.Scatter3d(x=X[:,j], y=Y[:,j], z=Z[:,j], mode="lines", line=dict(width=1)))
    return lines

def wireframe_from_xyz(X, Y, Z, as_points=False):
    if as_points:
        return go.Figure(go.Scatter3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), mode="markers", marker=dict(size=2)))
    if X.ndim == 1 or Y.ndim == 1 or Z.ndim == 1:
        return go.Figure(go.Scatter3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), mode="markers", marker=dict(size=2)))
    lines = _wireframe_lines(X, Y, Z)
    fig = go.Figure(lines)
    return fig

def line_from_xyz(X, Y, Z):
    return go.Figure(go.Scatter3d(x=X, y=Y, z=Z, mode="lines"))

def _arrow_lines(xs, ys, zs, us, vs, ws, scale=0.2):
    xs = np.asarray(xs); ys = np.asarray(ys); zs = np.asarray(zs)
    us = np.asarray(us); vs = np.asarray(vs); ws = np.asarray(ws)
    xe = xs + scale * us; ye = ys + scale * vs; ze = zs + scale * ws
    X = np.empty(xs.size * 3); Y = np.empty_like(X); Z = np.empty_like(X)
    X[0::3] = xs.flatten(); X[1::3] = xe.flatten(); X[2::3] = np.nan
    Y[0::3] = ys.flatten(); Y[1::3] = ye.flatten(); Y[2::3] = np.nan
    Z[0::3] = zs.flatten(); Z[1::3] = ze.flatten(); Z[2::3] = np.nan
    return go.Scatter3d(x=X, y=Y, z=Z, mode="lines", line=dict(width=2))

def quiver2d(X, Y, U, V, scale=0.2):
    return go.Figure(_arrow_lines(X, Y, np.zeros_like(X), U, V, np.zeros_like(U), scale=scale))

def quiver3d(X, Y, Z, U, V, W, scale=0.2):
    return go.Figure(_arrow_lines(X, Y, Z, U, V, W, scale=scale))
