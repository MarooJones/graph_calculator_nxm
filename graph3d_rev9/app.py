
import re
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, callback_context as ctx, no_update
import plotly.graph_objects as go

# ---------- Math env ----------
SAFE = {
    "np": np,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "ln": np.log,
    "sqrt": np.sqrt, "abs": np.abs,
    "minimum": np.minimum, "maximum": np.maximum, "where": np.where,
    "pow": np.power,
    "pi": np.pi, "e": np.e,
}

def normalize_expr(expr: str) -> str:
    s = expr.strip()
    s = re.sub(r"\^", "**", s)   # allow ^ for exponent
    s = re.sub(r"\bx\b", "x1", s)
    s = re.sub(r"\by\b", "x2", s)
    s = re.sub(r"\bz\b", "x3", s)
    return s

def eval_expr(expr, **vars):
    env = dict(SAFE)
    env.update(vars)
    expr = normalize_expr(expr)
    return eval(expr, {"__builtins__": {}}, env)

# ---------- Themes ----------
THEMES = {
    "dark": {
        "template": "plotly_dark",
        "paper_bg": "#0c0f14",
        "font_color": "#e5e7eb",
        "input_bg": "#1f2937",
        "input_fg": "#e5e7eb",
        "button_bg": "#374151",
        "button_fg": "#e5e7eb",
        "scene": dict(
            bgcolor="#0f1318",
            xaxis=dict(gridcolor="#2a3038", zerolinecolor="#3b4350", showbackground=True, backgroundcolor="#161b21"),
            yaxis=dict(gridcolor="#2a3038", zerolinecolor="#3b4350", showbackground=True, backgroundcolor="#161b21"),
            zaxis=dict(gridcolor="#2a3038", zerolinecolor="#3b4350", showbackground=True, backgroundcolor="#161b21"),
        ),
    },
    "light": {
        "template": "plotly_white",
        "paper_bg": "#ffffff",
        "font_color": "#111827",
        "input_bg": "#ffffff",
        "input_fg": "#111827",
        "button_bg": "#e5e7eb",
        "button_fg": "#111827",
        "scene": dict(
            bgcolor="#f8fafc",
            xaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#94a3b8", showbackground=True, backgroundcolor="#f8fafc"),
            yaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#94a3b8", showbackground=True, backgroundcolor="#f8fafc"),
            zaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#94a3b8", showbackground=True, backgroundcolor="#f8fafc"),
        ),
    },
}

def themed_input(id_, theme_key, value, step=0.1, minv=None, maxv=None):
    t = THEMES[theme_key]
    style={"width":"100px","backgroundColor":t["input_bg"],"color":t["input_fg"],"border":"1px solid #4b5563","borderRadius":"6px","padding":"4px 6px"}
    return dcc.Input(id=id_, type="number", step=step, min=minv, max=maxv, value=value, style=style)

def themed_dd(id_, theme_key):
    t = THEMES[theme_key]
    return dcc.Dropdown(id=id_, placeholder="Quick",
                        options=[{"label":"±1","value":1},{"label":"±3","value":3},{"label":"±5","value":5},{"label":"±10","value":10}],
                        clearable=True, style={"width":"120px","backgroundColor":t["input_bg"],"color":t["input_fg"]})

def themed_btn(id_, label, theme_key, ml="6px"):
    t = THEMES[theme_key]
    return html.Button(label, id=id_, n_clicks=0, style={"marginLeft":ml,"backgroundColor":t["button_bg"],"color":t["button_fg"],"border":"1px solid #4b5563","borderRadius":"6px","padding":"4px 8px"})

# ---------- Sampling ----------
def samples_from_density(a, b, density_per_unit, min_s=10, max_s=240):
    span = max(1e-9, float(b) - float(a))
    s = int(round(span * float(density_per_unit)))
    return max(min_s, min(max_s, s))

# Helpers
def ensure_array(val, like, dtype=float):
    arr = np.asarray(val, dtype=dtype)
    if arr.ndim == 0:
        arr = np.full_like(like, float(arr), dtype=dtype)
    return arr

# ---------- Plotters ----------
def make_surface(xname, yname, fixed, expr, ranges, density, render_style, theme):
    ax = np.linspace(ranges[xname][0], ranges[xname][1], samples_from_density(*ranges[xname], density))
    ay = np.linspace(ranges[yname][0], ranges[yname][1], samples_from_density(*ranges[yname], density))
    X, Y = np.meshgrid(ax, ay, indexing="xy")
    vars = {xname: X, yname: Y}
    for k, v in fixed.items():
        vars[k] = np.full_like(X, float(v))
    Z = ensure_array(eval_expr(expr, **vars), X)
    surface = go.Surface(x=X, y=Y, z=Z, showscale=True, opacity=1.0, colorscale="Turbo")
    if render_style == "wire":
        surface.update(contours_z=dict(show=True, usecolormap=True, project_z=True), showscale=False, opacity=0.95)
    fig = go.Figure(data=[surface])
    fig.update_layout(template=theme["template"])
    fig.update_scenes(**theme["scene"], xaxis_title=xname, yaxis_title=yname, zaxis_title="z")
    return fig

def make_vector_cones(vary, fixed, comps, ranges, density, arrow_pct, theme):
    vary = list(vary)
    arrays = {}
    for vn in vary:
        a, b = ranges[vn]
        arrays[vn] = np.linspace(a, b, samples_from_density(a, b, density))
    if len(vary) == 2:
        all_vars = ["x1", "x2", "x3"]
        zlike = next((v for v in all_vars if v not in vary), "x3")
        z_fixed = float(fixed.get(zlike, 0.0))
        A, B = np.meshgrid(arrays[vary[0]], arrays[vary[1]], indexing="xy")
        X = A; Y = B; Z = np.full_like(X, z_fixed)
    elif len(vary) == 3:
        X, Y, Z = np.meshgrid(arrays["x1"], arrays["x2"], arrays["x3"], indexing="xy")
    else:
        raise ValueError("Choose 2 or 3 variables to vary for vector fields.")
    vars = {"x1": X, "x2": Y, "x3": Z}
    for k, v in fixed.items():
        if k not in vars:
            vars[k] = np.full_like(X, float(v))
    U = ensure_array(eval_expr(comps[0], **vars), X)
    V = ensure_array(eval_expr(comps[1], **vars), X)
    if len(comps) >= 3:
        W = ensure_array(eval_expr(comps[2], **vars), X)
    else:
        W = np.zeros_like(X, dtype=float)
    xspan = max(1e-9, float(ranges["x1"][1] - ranges["x1"][0]))
    yspan = max(1e-9, float(ranges["x2"][1] - ranges["x2"][0]))
    zspan = max(1e-9, float(ranges["x3"][1] - ranges["x3"][0]))
    mean_span = (xspan + yspan + zspan) / 3.0
    s = max(1.0, float(arrow_pct))
    sizeref = max(1e-6, mean_span / (s / 6.0))
    cones = go.Cone(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        u=U.flatten(), v=V.flatten(), w=W.flatten(),
        sizemode="scaled", sizeref=sizeref, anchor="tail", colorscale="Turbo", showscale=True
    )
    fig = go.Figure(data=[cones])
    fig.update_layout(template=theme["template"])
    fig.update_scenes(**theme["scene"], xaxis_title="x1", yaxis_title="x2", zaxis_title="x3")
    return fig

# ---------- App ----------
app = Dash(__name__)
app.title = "3D Graph Tool (Dash)"

CONTROL_STYLE = {"width": "580px","padding": "14px 18px 22px 18px","overflowY": "auto","height": "96vh","boxSizing": "border-box","borderRight": "1px solid #2a3038"}
GRAPH_STYLE = {"flex": "1", "padding": "10px", "height": "96vh"}
LABEL = {"fontSize":"13px", "opacity":0.9}

def range_row(xi, theme_key):
    return html.Div([
        html.Div([themed_input(f"in-{xi}-min", theme_key, -3.0)], style={"width":"110px","flex":"0 0 110px"}),
        dcc.RangeSlider(id=f"range-{xi}", min=-50, max=50, step=0.1, value=[-3.0, 3.0],
                        allowCross=False, updatemode="mouseup", marks=None,
                        tooltip={"always_visible":False}),
        html.Div([themed_input(f"in-{xi}-max", theme_key, 3.0)], style={"width":"110px","flex":"0 0 110px","textAlign":"right"}),
    ], style={"display":"flex","alignItems":"center","gap":"10px","marginTop":"6px","marginBottom":"2px"})

def range_block(xi, theme_key):
    t = THEMES[theme_key]
    return html.Div([
        html.Label(f"{xi} range", style=LABEL),
        range_row(xi, theme_key),
        html.Div([
            dcc.Input(id=f"sym-{xi}-amp", type="number", step=0.1, placeholder="±a",
                      style={"width":"100px","backgroundColor":t["input_bg"],"color":t["input_fg"],"border":"1px solid #4b5563","borderRadius":"6px","padding":"4px 6px"}),
            themed_btn(f"sym-{xi}-apply", "Set ±a", theme_key),
            dcc.Checklist(id=f"sym-{xi}-lock", options=[{"label":" Lock symmetric","value":"lock"}], value=[], inline=True, style={"marginLeft":"10px"}),
            themed_btn(f"zoom-{xi}-in", "Zoom −20%", theme_key, "10px"),
            themed_btn(f"zoom-{xi}-out", "Zoom +20%", theme_key),
            themed_btn(f"reset-{xi}", "Reset", theme_key, "10px"),
            themed_dd(f"quick-{xi}", theme_key),
        ], style={"display":"flex","alignItems":"center","flexWrap":"wrap","gap":"8px","marginTop":"8px","marginBottom":"10px"}),
        html.Div([html.Label(f"{xi} fixed", style=LABEL),
                  dcc.Slider(id=f"fixed-{xi}", min=-50, max=50, step=0.1, value=0.0, updatemode="mouseup", marks=None, tooltip={"always_visible":False})],
                 style={"marginTop":"8px","marginBottom":"10px"}),
    ], style={"marginBottom":"20px"})

def controls_panel(theme_key):
    return html.Div(style=CONTROL_STYLE, children=[
        html.H2("3D Graph Tool (Python • Plotly Dash)", style={"marginTop":"4px"}),
        dcc.Store(id="camera-store"),
        html.Div(id="banner-nature", style={"fontFamily":"monospace","marginBottom":"10px"}),
        dcc.RadioItems(id="theme", options=[{"label":"Dark","value":"dark"},{"label":"Light","value":"light"}], value=theme_key, inline=True, style={"marginBottom":"8px"}),
        dcc.Dropdown(id="mode", options=[
            {"label":"General f: ℝⁿ → ℝᵐ (auto)","value":"general"},
            {"label":"Scalar surface: z = f(x, y)","value":"scalar"},
            {"label":"Parametric height-field: z = z(u, v)","value":"height"},
            {"label":"Parametric embedded surface: (x(u, v), y(u, v), z(u, v))","value":"embedded"},
            {"label":"Parametric curve: r(t) in ℝ³","value":"curve"},
        ], value="general", clearable=False, style={"marginBottom":"10px"}),
        html.Div([html.Label("n (domain ℝⁿ)", style=LABEL),
                  themed_input("n", theme_key, 3, step=1, minv=1, maxv=3),
                  html.Label("m (codomain ℝᵐ)", style={**LABEL,"marginLeft":"10px"}),
                  themed_input("m", theme_key, 1, step=1, minv=1, maxv=3)],
                 style={"display":"flex","gap":"12px","alignItems":"center","marginBottom":"10px"}),
        html.Div([html.Label("Vary variables (choose 1–3)", style=LABEL),
                  dcc.Checklist(id="vary-variables", options=[{"label":f"x{i}","value":f"x{i}"} for i in range(1,4)], value=["x1","x2"], inline=True)], style={"marginBottom":"8px"}),
        html.Label("f(x1, …, xn) as comma-separated expressions (length = m). Example: x1 + x2, x2 + x3, x1", style={"fontSize":"12px"}),
        dcc.Textarea(id="exprs", value="x1 + x2", style={"width":"100%","height":"70px"}),
        html.Hr(),
        html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center"}, children=[
            html.Label("Grid density (samples per unit)", style=LABEL),
            themed_btn("reset-all", "Reset all ranges", theme_key, "10px"),
        ]),
        dcc.Slider(id="density", min=5, max=60, step=1, value=20, updatemode="mouseup", marks=None, tooltip={"always_visible":False}),
        range_block("x1", theme_key),
        range_block("x2", theme_key),
        range_block("x3", theme_key),
        html.Label("Arrow scale (%) for vector fields", style={"marginTop":"12px", **LABEL}),
        dcc.Slider(id="arrow-scale", min=1, max=100, step=1, value=30, updatemode="mouseup", marks=None, tooltip={"always_visible":False}),
        html.Div([html.Label("Render style", style=LABEL),
                  dcc.RadioItems(id="render-style", options=[{"label":"Surface","value":"surf"},{"label":"Wireframe","value":"wire"}], value="surf", inline=True)],
                 style={"marginTop":"8px"}),
        html.Div(id="status", style={"fontSize":"12px","opacity":0.8,"marginTop":"8px"}),
    ])

app.layout = html.Div(id="root", style={"display":"flex","flexDirection":"row","gap":"0"}, children=[
    controls_panel("dark"),
    html.Div(style=GRAPH_STYLE, children=[dcc.Graph(id="graph", style={"height":"100%","width":"100%"})]),
])

# --- Rebuild controls on theme change so inputs adopt theme colors ---
@app.callback(Output("root","children"), Input("theme","value"), State("root","children"))
def rebuild_on_theme(theme_key, current_children):
    return [controls_panel(theme_key or "dark"), current_children[1]]

# ------- Sync: slider -> inputs (values only) -------
def make_slider_to_inputs(xi):
    @app.callback(Output(f"in-{xi}-min","value"), Output(f"in-{xi}-max","value"), Input(f"range-{xi}","value"))
    def slider_to_inputs(r):
        if not r: return -3.0, 3.0
        return float(r[0]), float(r[1])
make_slider_to_inputs("x1"); make_slider_to_inputs("x2"); make_slider_to_inputs("x3")

# ------- Controls -> slider (commit on events; min/max via State) -------
def make_controls_to_slider(xi):
    @app.callback(Output(f"range-{xi}","value"), Output(f"quick-{xi}","value"),
                  Input(f"quick-{xi}","value"),
                  Input(f"sym-{xi}-apply","n_clicks"), State(f"sym-{xi}-amp","value"),
                  Input(f"zoom-{xi}-in","n_clicks"), Input(f"zoom-{xi}-out","n_clicks"),
                  Input(f"reset-{xi}","n_clicks"), Input("reset-all","n_clicks"),
                  Input(f"in-{xi}-min","n_blur"), Input(f"in-{xi}-max","n_blur"),
                  State(f"in-{xi}-min","value"), State(f"in-{xi}-max","value"),
                  State(f"sym-{xi}-lock","value"),
                  State(f"range-{xi}","value"))
    def controls_to_slider(quick, sym_clicks, sym_amp, zin, zout, reset_axis, reset_all, nblur_min, nblur_max, vmin, vmax, lock_vals, current):
        trig = ctx.triggered_id
        current = current or [-3.0, 3.0]
        a, b = float(current[0]), float(current[1])

        if trig == f"reset-{xi}" or trig == "reset-all":
            return [-3.0, 3.0], None
        if trig == f"quick-{xi}" and quick is not None:
            val = float(quick); return [-val, val], None
        if trig == f"sym-{xi}-apply" and (sym_amp is not None):
            val = float(sym_amp); return [-val, val], None
        if trig == f"zoom-{xi}-in":
            c = 0.5*(a+b); half = 0.5*(b-a)*0.8; return [c-half, c+half], no_update
        if trig == f"zoom-{xi}-out":
            c = 0.5*(a+b); half = 0.5*(b-a)/0.8; return [c-half, c+half], no_update
        if trig in (f"in-{xi}-min", f"in-{xi}-max"):
            try:
                aa = float(vmin) if vmin is not None else a
                bb = float(vmax) if vmax is not None else b
                if ("lock" in (lock_vals or [])):
                    m = max(abs(aa), abs(bb)); aa, bb = -m, m
                if aa == bb: bb = aa + 1.0
                if aa > bb: aa, bb = bb, aa
                return [aa, bb], no_update
            except Exception:
                return current, no_update
        return no_update, no_update
make_controls_to_slider("x1"); make_controls_to_slider("x2"); make_controls_to_slider("x3")

# ------- Preserve camera -------
@app.callback(Output("camera-store","data"), Input("graph","relayoutData"), State("camera-store","data"))
def save_camera(relayout, current):
    if not relayout: return no_update
    cam = relayout.get("scene.camera")
    if cam is not None:
        return cam
    return no_update

# ------- Main figure -------
@app.callback(Output("graph","figure"), Output("banner-nature","children"), Output("status","children"),
              Input("theme","value"),
              Input("mode","value"), Input("n","value"), Input("m","value"),
              Input("vary-variables","value"), Input("exprs","value"), Input("density","value"),
              Input("range-x1","value"), Input("range-x2","value"), Input("range-x3","value"),
              Input("fixed-x1","value"), Input("fixed-x2","value"), Input("fixed-x3","value"),
              Input("arrow-scale","value"), Input("render-style","value"),
              State("camera-store","data"))
def update_fig(theme_key, mode, n, m, vary_vars, exprs, density, rx1, rx2, rx3, fx1, fx2, fx3, arrow_pct, render_style, camera):
    theme = THEMES.get(theme_key or "dark")
    # Clamp and coerce n,m to integers within [1,3]
    try:
        n = max(1, min(3, int(float(n or 3))))
        m = max(1, min(3, int(float(m or 1))))
    except Exception:
        n, m = 3, 1
    vary_vars = [v for v in (vary_vars or []) if v in ("x1","x2","x3")]
    exprs = (exprs or "").strip()
    comps = [normalize_expr(c.strip()) for c in exprs.split(",") if c.strip()]
    banner = f"Nature: f : ℝ^{n} → ℝ^{m}"
    ranges = {"x1": rx1 or [-3, 3], "x2": rx2 or [-3, 3], "x3": rx3 or [-3, 3]}
    fixed = {"x1": fx1 or 0.0, "x2": fx2 or 0.0, "x3": fx3 or 0.0}
    status = ""
    try:
        if mode == "general":
            if m == 1 and len(vary_vars) == 2:
                fig = make_surface(vary_vars[0], vary_vars[1], {k: fixed[k] for k in fixed if k not in vary_vars},
                                   comps[0] if comps else "x1 + x2", ranges, density, render_style, theme)
                status = "Auto: surface plot of scalar f(xi, xj)"
            elif m in (2, 3) and len(vary_vars) in (2, 3):
                while len(comps) < 3: comps.append("0")
                fig = make_vector_cones(vary_vars, {k: fixed[k] for k in fixed if k not in vary_vars},
                                        comps[:3], ranges, density, arrow_pct, theme)
                status = "Auto: 3D vector field via cones (anchor=tail)"
            else:
                fig = go.Figure()
                msg = "For m=1, pick exactly two varying variables (surface). For m∈{2,3}, pick 2–3 variables (vector field)."
                fig.update_layout(template=theme["template"], title=msg, paper_bgcolor=theme["paper_bg"], font=dict(color=theme["font_color"]))
                status = "Nothing to draw with current n, m, and varying variables."
        else:
            # For now, route the other modes to 'general' until fully implemented
            return update_fig(theme_key, "general", n, m, vary_vars, exprs, density, rx1, rx2, rx3, fx1, fx2, fx3, arrow_pct, render_style, camera)

        fig.update_layout(template=theme["template"], margin=dict(l=0, r=0, t=24, b=0), paper_bgcolor=theme["paper_bg"], font=dict(color=theme["font_color"]))
        fig.update_scenes(**theme["scene"])
        if camera:
            fig.update_layout(scene=dict(camera=camera))

        return fig, banner, status
    except Exception as e:
        fig = go.Figure()
        hint = "Hint: use ** (or ^) for powers. Variables: x1,x2,x3 (aliases: x,y,z)."
        fig.update_layout(template=theme["template"], title=f"Error: {type(e).__name__}: {e}", paper_bgcolor=theme["paper_bg"], annotations=[dict(text=hint, showarrow=False, font=dict(color=theme["font_color"]))])
        status = "There was an error evaluating your expressions. Check variable names and expression length."
        return fig, banner, status

if __name__ == "__main__":
    app.run_server(debug=True)
