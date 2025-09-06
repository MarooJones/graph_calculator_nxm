
import numpy as np
import math
from dash import Dash, dcc, html, Input, Output, State, callback_context as ctx
import plotly.graph_objects as go

# ----------------------------
# Helpers
# ----------------------------
SAFE = {
    # numpy aliases
    "np": np,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
    "min": np.minimum, "max": np.maximum,
    "pi": np.pi, "e": np.e,
}

def eval_expr(expr, **vars):
    """Safe-ish eval: expose only numpy ufuncs + provided vars."""
    env = dict(SAFE)
    env.update(vars)
    return eval(expr, {"__builtins__": {}}, env)

def samples_from_density(a, b, density_per_unit, min_s=10, max_s=200):
    """Return integer sample count based on axis span and a density (samples per unit)."""
    span = max(1e-9, float(b) - float(a))
    s = int(round(span * float(density_per_unit)))
    return max(min_s, min(max_s, s))

def default_ranges(n):
    # (-3, 3) for up to 3 variables
    return {f"x{i}": [-3.0, 3.0] for i in range(1, n+1)}

def make_surface(xname, yname, fixed, expr, ranges, density, render_style):
    ax = np.linspace(ranges[xname][0], ranges[xname][1], samples_from_density(*ranges[xname], density))
    ay = np.linspace(ranges[yname][0], ranges[yname][1], samples_from_density(*ranges[yname], density))
    X, Y = np.meshgrid(ax, ay, indexing="xy")

    # Build variables dict
    vars = {xname: X, yname: Y}
    for k, v in fixed.items():
        vars[k] = np.full_like(X, float(v))

    Z = eval_expr(expr, **vars)
    surface = go.Surface(x=X, y=Y, z=Z, showscale=True, opacity=1.0)
    if render_style == "wire":
        surface.update(contours_z=dict(show=True, usecolormap=True, project_z=True), showscale=False, opacity=0.95)

    fig = go.Figure(data=[surface])
    fig.update_scenes(xaxis_title=xname, yaxis_title=yname, zaxis_title="z")
    return fig

def make_vector_cones(vary, fixed, comps, ranges, density, arrow_pct):
    """
    Vector field cones. Places cones on a grid in the subspace spanned by `vary`.
    If only 2 vary, plot in a slab with the remaining axis fixed (z plane if possible).
    """
    vary = list(vary)
    # Build grid
    arrays = {}
    for vn in vary:
        a, b = ranges[vn]
        arrays[vn] = np.linspace(a, b, samples_from_density(a, b, density))

    # We support 2D grid (with the remaining axis held constant) or full 3D grid.
    if len(vary) == 2:
        # choose an axis for z if not present
        all_vars = ["x1", "x2", "x3"]
        zlike = next((v for v in all_vars if v not in vary), "x3")
        z_fixed = float(fixed.get(zlike, 0.0))
        A, B = np.meshgrid(arrays[vary[0]], arrays[vary[1]], indexing="xy")
        X = A; Y = B
        Z = np.full_like(X, z_fixed)
    elif len(vary) == 3:
        X, Y, Z = np.meshgrid(arrays["x1"], arrays["x2"], arrays["x3"], indexing="xy")
    else:
        raise ValueError("Choose 2 or 3 variables to vary for vector fields.")

    vars = {"x1": X, "x2": Y, "x3": Z}
    # Fill fixed others
    for k, v in fixed.items():
        if k not in vars:
            vars[k] = np.full_like(X, float(v))

    U = eval_expr(comps[0], **vars)
    V = eval_expr(comps[1], **vars)
    W = eval_expr(comps[2] if len(comps) >= 3 else "0", **vars)

    # Cone scaling heuristics
    xspan = max(1e-9, float(ranges["x1"][1] - ranges["x1"][0]))
    yspan = max(1e-9, float(ranges["x2"][1] - ranges["x2"][0]))
    zspan = max(1e-9, float(ranges.get("x3", [0, 1])[1] - ranges.get("x3", [0, 1])[0]))
    mean_span = (xspan + yspan + zspan) / 3.0
    # plotly cones get smaller as sizeref increases; arrow_pct in [1,100]
    s = max(1.0, float(arrow_pct))
    sizeref = max(1e-6, mean_span / (s / 6.0))  # tweak denominator for a good default range

    cones = go.Cone(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        u=U.flatten(), v=V.flatten(), w=W.flatten(),
        sizemode="scaled", sizeref=sizeref, anchor="tail", showscale=True
    )
    fig = go.Figure(data=[cones])
    fig.update_scenes(xaxis_title="x1", yaxis_title="x2", zaxis_title="x3")
    return fig

# ----------------------------
# App
# ----------------------------
app = Dash(__name__)
app.title = "3D Graph Tool (Dash)"

CONTROL_STYLE = {
    "width": "420px",
    "padding": "10px 18px 18px 18px",
    "overflowY": "auto",
    "height": "96vh",
    "boxSizing": "border-box",
    "borderRight": "1px solid #333",
}

GRAPH_STYLE = {"flex": "1", "padding": "10px", "height": "96vh"}

app.layout = html.Div(
    style={"display": "flex", "flexDirection": "row", "gap": "0"},
    children=[
        # Left panel (controls)
        html.Div(
            style=CONTROL_STYLE,
            children=[
                html.H2("3D Graph Tool (Python • Plotly Dash)", style={"marginTop": "6px"}),
                html.Div(id="banner-nature", style={"fontFamily": "monospace", "marginBottom": "10px"}),
                dcc.Dropdown(
                    id="mode",
                    options=[
                        {"label": "General f: ℝⁿ → ℝᵐ (auto)", "value": "general"},
                        {"label": "Scalar surface: z = f(x, y)", "value": "scalar"},
                        {"label": "Parametric height-field: z = z(u, v)", "value": "height"},
                        {"label": "Parametric embedded surface: (x(u, v), y(u, v), z(u, v))", "value": "embedded"},
                        {"label": "Parametric curve: r(t) in ℝ³", "value": "curve"},
                    ],
                    value="general", clearable=False, style={"marginBottom": "10px"},
                ),

                html.Div([
                    html.Label("n (domain ℝⁿ)"),
                    dcc.Input(id="n", type="number", value=3, min=1, max=3, step=1, style={"width": "80px", "marginRight": "12px"}),
                    html.Label("m (codomain ℝᵐ)"),
                    dcc.Input(id="m", type="number", value=1, min=1, max=3, step=1, style={"width": "80px"}),
                ], style={"display": "flex", "gap": "8px", "alignItems": "center", "marginBottom": "8px"}),

                html.Div([
                    html.Label("Vary variables (choose 1–3)"),
                    dcc.Checklist(
                        id="vary-variables",
                        options=[{"label": f"x{i}", "value": f"x{i}"} for i in range(1, 4)],
                        value=["x1", "x2"],  # default surf
                        inline=True,
                    ),
                ], style={"marginBottom": "6px"}),

                html.Label("f(x1, …, xn) as comma-separated expressions (length = m). Example: x1 + x2, x2 + x3, x1",
                           style={"fontSize": "12px"}),
                dcc.Textarea(id="exprs", value="x1 + x2", style={"width": "100%", "height": "58px"}),

                html.Hr(),
                html.Label("Grid density (samples per unit)"),
                dcc.Slider(id="density", min=5, max=60, step=1, value=20, marks=None, tooltip={"placement": "bottom", "always_visible": True}),

                html.Div(id="ranges-container"),  # dynamic ranges + fixed sliders

                html.Label("Arrow scale (%) for vector fields", style={"marginTop": "10px"}),
                dcc.Slider(id="arrow-scale", min=1, max=100, step=1, value=30, marks=None, tooltip={"placement": "bottom", "always_visible": True}),

                html.Div([
                    html.Label("Render style"),
                    dcc.RadioItems(
                        id="render-style",
                        options=[{"label": "Surface", "value": "surf"}, {"label": "Wireframe", "value": "wire"}],
                        value="surf", inline=True
                    ),
                ], style={"marginTop": "8px"}),

                html.Div(id="status", style={"fontSize": "12px", "opacity": 0.7, "marginTop": "6px"}),
            ],
        ),
        # Right panel (graph)
        html.Div(
            style=GRAPH_STYLE,
            children=[
                dcc.Graph(id="graph", style={"height": "100%", "width": "100%"})
            ],
        ),
    ],
)

def build_range_controls(n):
    children = []
    for i in range(1, min(3, n) + 1):
        xi = f"x{i}"
        children.append(html.Div([
            html.Label(f"{xi} range"),
            dcc.RangeSlider(id=f"range-{xi}", min=-6, max=6, step=0.1, value=[-3.0, 3.0],
                            allowCross=False, tooltip={"placement": "bottom", "always_visible": False})
        ], style={"marginBottom": "12px"}))
        children.append(html.Div([
            html.Label(f"{xi} fixed"),
            dcc.Slider(id=f"fixed-{xi}", min=-6, max=6, step=0.1, value=0.0,
                       tooltip={"placement": "bottom", "always_visible": False})
        ], style={"margin": "0 0 12px 0"}))
    return children

@app.callback(Output("ranges-container", "children"), Input("n", "value"))
def update_range_controls(n):
    try:
        n = int(n or 3)
    except:
        n = 3
    return build_range_controls(n)

@app.callback(
    Output("graph", "figure"),
    Output("banner-nature", "children"),
    Output("status", "children"),
    Input("mode", "value"),
    Input("n", "value"),
    Input("m", "value"),
    Input("vary-variables", "value"),
    Input("exprs", "value"),
    Input("density", "value"),
    Input("range-x1", "value"), Input("range-x2", "value"), Input("range-x3", "value"),
    Input("fixed-x1", "value"), Input("fixed-x2", "value"), Input("fixed-x3", "value"),
    Input("arrow-scale", "value"),
    Input("render-style", "value"),
)
def update_fig(mode, n, m, vary_vars, exprs, density,
               rx1, rx2, rx3, fx1, fx2, fx3, arrow_pct, render_style):
    # sanitize
    n = int(n or 3); m = int(m or 1)
    vary_vars = vary_vars or []
    exprs = (exprs or "").strip()
    comps = [c.strip() for c in exprs.split(",") if c.strip()]
    # nature banner
    banner = f"Nature: f : ℝ^{n} → ℝ^{m}"

    # collect ranges
    ranges = {"x1": rx1 or [-3, 3], "x2": rx2 or [-3, 3], "x3": rx3 or [-3, 3]}
    fixed = {"x1": fx1 or 0.0, "x2": fx2 or 0.0, "x3": fx3 or 0.0}

    status = ""
    try:
        # Force general mode semantics even if user picks other modes (we keep the menu for future)
        if mode == "general":
            # Decide visualization
            if m == 1 and len(vary_vars) == 2:
                fig = make_surface(vary_vars[0], vary_vars[1], {k: fixed[k] for k in fixed if k not in vary_vars},
                                   comps[0] if comps else "x1 + x2", ranges, density, render_style)
                status = "Auto: surface plot of scalar f(xi, xj)"
            elif m in (2, 3) and len(vary_vars) in (2, 3):
                # Need at least 2 components; pad if m==2
                while len(comps) < 3:
                    comps.append("0")
                fig = make_vector_cones(vary_vars, {k: fixed[k] for k in fixed if k not in vary_vars},
                                        comps[:3], ranges, density, arrow_pct)
                status = "Auto: 3D vector field via cones (anchor=tail)"
            else:
                # Fallback informative blank
                fig = go.Figure()
                fig.update_layout(
                    template="plotly_dark",
                    title="Pick m=1 with 2 varying variables (surface) or m∈{2,3} with 2–3 varying variables (vector field).",
                )
                status = "Nothing to draw with current n, m, and varying variables."
        else:
            # For now, reuse general logic
            return update_fig("general", n, m, vary_vars, exprs, density, rx1, rx2, rx3, fx1, fx2, fx3, arrow_pct, render_style)

        fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0))
        return fig, banner, status
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title=f"Error: {type(e).__name__}: {e}")
        status = "There was an error evaluating your expressions. Check variable names and expression length."
        return fig, banner, status

if __name__ == "__main__":
    app.run_server(debug=True)
