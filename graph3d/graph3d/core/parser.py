from __future__ import annotations
import numpy as np
import sympy as sp

SUPPORTED_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
    "floor": sp.floor, "ceil": sp.ceiling, "ceiling": sp.ceiling,
    "pi": sp.pi, "E": sp.E,
    "Piecewise": sp.Piecewise, "Max": sp.Max, "Min": sp.Min,
}

def _sympify(expr: str, symbols: dict[str, sp.Symbol]):
    if expr is None or str(expr).strip() == "":
        raise ValueError("Empty expression.")
    return sp.sympify(expr, locals={**SUPPORTED_FUNCS, **symbols})

def _lambdify(expr, vars):
    return sp.lambdify(vars, expr, modules=["numpy"])

# ℝ²→ℝ: f(x, y)
def make_func_xy(expr: str):
    x, y = sp.symbols("x y")
    e = _sympify(expr, {"x": x, "y": y})
    f = _lambdify(e, (x, y))
    return lambda X, Y: np.asarray(f(X, Y), dtype=float)

def make_mask_xy(expr: str):
    x, y = sp.symbols("x y")
    e = _sympify(expr, {"x": x, "y": y})
    f = _lambdify(e, (x, y))
    return lambda X, Y: np.asarray(f(X, Y), dtype=bool)

# ℝ²→ℝ: z(u, v)
def make_func_uv(expr: str):
    u, v = sp.symbols("u v")
    e = _sympify(expr, {"u": u, "v": v})
    f = _lambdify(e, (u, v))
    return lambda U, V: np.asarray(f(U, V), dtype=float)

def make_mask_uv(expr: str):
    u, v = sp.symbols("u v")
    e = _sympify(expr, {"u": u, "v": v})
    f = _lambdify(e, (u, v))
    return lambda U, V: np.asarray(f(U, V), dtype=bool)

# ℝ²→ℝ³: (x(u,v), y(u,v), z(u,v))
def make_funcs_embed_uv(ex, ey, ez):
    u, v = sp.symbols("u v")
    xe = _sympify(ex, {"u": u, "v": v})
    ye = _sympify(ey, {"u": u, "v": v})
    ze = _sympify(ez, {"u": u, "v": v})
    fx = _lambdify(xe, (u, v))
    fy = _lambdify(ye, (u, v))
    fz = _lambdify(ze, (u, v))
    return (
        lambda U, V: np.asarray(fx(U, V), dtype=float),
        lambda U, V: np.asarray(fy(U, V), dtype=float),
        lambda U, V: np.asarray(fz(U, V), dtype=float),
    )

# ℝ→ℝ³: (x(t), y(t), z(t))
def make_funcs_curve_t(ex, ey, ez):
    t = sp.symbols("t")
    xe = _sympify(ex, {"t": t})
    ye = _sympify(ey, {"t": t})
    ze = _sympify(ez, {"t": t})
    fx = _lambdify(xe, (t,))
    fy = _lambdify(ye, (t,))
    fz = _lambdify(ze, (t,))
    return (
        lambda T: np.asarray(fx(T), dtype=float),
        lambda T: np.asarray(fy(T), dtype=float),
        lambda T: np.asarray(fz(T), dtype=float),
    )

# General ℝⁿ → ℝᵐ helpers
def make_funcs_general(exprs: str, n: int, m: int):
    xs = sp.symbols(" ".join([f"x{i}" for i in range(1, n+1)]))
    if isinstance(xs, sp.Symbol):
        xs = (xs,)
    parts = [p.strip() for p in str(exprs).split(",") if p.strip()]
    if len(parts) != m:
        raise ValueError(f"Expected {m} expression(s), got {len(parts)}.")
    env = {**SUPPORTED_FUNCS, **{f"x{i}": xs[i-1] for i in range(1, n+1)}}
    expr_objs = [sp.sympify(p, locals=env) for p in parts]
    fs = [sp.lambdify(xs, e, modules=["numpy"]) for e in expr_objs]
    return [lambda *args, f=f: np.asarray(f(*args), dtype=float) for f in fs]

def make_mask_general(expr: str, n: int):
    xs = sp.symbols(" ".join([f"x{i}" for i in range(1, n+1)]))
    if isinstance(xs, sp.Symbol):
        xs = (xs,)
    env = {**SUPPORTED_FUNCS, **{f"x{i}": xs[i-1] for i in range(1, n+1)}}
    e = sp.sympify(expr, locals=env)
    f = sp.lambdify(xs, e, modules=["numpy"])
    return lambda *args: np.asarray(f(*args), dtype=bool)
