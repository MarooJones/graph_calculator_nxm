from dash import dcc, html

def build_layout():
    return html.Div(
        [
            html.H1("3D Graph Tool (Python • Plotly Dash)"),
            html.Div(
                [
                    html.Label("Mode"),
                    dcc.Dropdown(
                        id="mode",
                        options=[
                            {"label": "Scalar surface: z = f(x, y)", "value": "scalar_xy"},
                            {"label": "Parametric height-field: z = z(u, v)", "value": "param_height"},
                            {"label": "Parametric embedded surface: (x(u, v), y(u, v), z(u, v))", "value": "param_embed"},
                            {"label": "Parametric curve: r(t) in ℝ³", "value": "curve_t"},
                            {"label": "General f: ℝⁿ → ℝᵐ (auto)", "value": "auto_nm"},
                        ],
                        value="scalar_xy",
                        clearable=False,
                    ),
                ],
                style={"maxWidth": 600, "marginBottom": 16},
            ),

            html.Div(
                id="inputs-container",
                children=[
                    # Scalar z = f(x,y)
                    html.Fieldset(
                        id="panel-scalar_xy",
                        children=[
                            html.Legend("Scalar: z = f(x, y) with domain in ℝ² → ℝ"),
                            html.Div([html.Label("f(x, y) ="), dcc.Input(id="scalar-fxy", type="text", value="sin(sqrt(x**2 + y**2))", style={"width": "100%"})]),
                            html.Div([html.Label("x range"), dcc.RangeSlider(id="x-range", min=-10, max=10, step=0.5, value=[-6, 6], tooltip={"placement": "bottom"})], style={"marginTop": 8}),
                            html.Div([html.Label("y range"), dcc.RangeSlider(id="y-range", min=-10, max=10, step=0.5, value=[-6, 6], tooltip={"placement": "bottom"})], style={"marginTop": 8}),
                            html.Div([html.Label("Domain mask (optional, e.g., x**2 + y**2 <= 25)"), dcc.Input(id="mask-xy", type="text", value="", style={"width": "100%"})], style={"marginTop": 8}),
                        ],
                        style={"marginBottom": 16},
                    ),

                    # Param height-field z=z(u,v)
                    html.Fieldset(
                        id="panel-param_height",
                        children=[
                            html.Legend("Parametric height-field: z = z(u, v) with (u, v) ∈ ℝ² → ℝ"),
                            html.Div([html.Label("z(u, v) ="), dcc.Input(id="param-z-uv", type="text", value="cos(u) * sin(v)", style={"width": "100%"})]),
                            html.Div([html.Label("u range"), dcc.RangeSlider(id="u-range", min=-6.28, max=6.28, step=0.1, value=[-3.14, 3.14], tooltip={"placement": "bottom"})], style={"marginTop": 8}),
                            html.Div([html.Label("v range"), dcc.RangeSlider(id="v-range", min=-6.28, max=6.28, step=0.1, value=[-3.14, 3.14], tooltip={"placement": "bottom"})], style={"marginTop": 8}),
                            html.Div([html.Label("Domain mask (optional, in u, v)"), dcc.Input(id="mask-uv", type="text", value="", style={"width": "100%"})], style={"marginTop": 8}),
                        ],
                        style={"marginBottom": 16, "display": "none"},
                    ),

                    # Param embedded (x(u,v),y(u,v),z(u,v))
                    html.Fieldset(
                        id="panel-param_embed",
                        children=[
                            html.Legend("Parametric embedded surface: (x(u, v), y(u, v), z(u, v)) with (u, v) ∈ ℝ² → ℝ³"),
                            html.Div([html.Label("x(u, v) ="), dcc.Input(id="embed-x-uv", type="text", value="(2 + cos(v)) * cos(u)", style={"width": "100%"})],),
                            html.Div([html.Label("y(u, v) ="), dcc.Input(id="embed-y-uv", type="text", value="(2 + cos(v)) * sin(u)", style={"width": "100%"})], style={"marginTop": 6}),
                            html.Div([html.Label("z(u, v) ="), dcc.Input(id="embed-z-uv", type="text", value="sin(v)", style={"width": "100%"})], style={"marginTop": 6}),
                            html.Div([html.Label("u range"), dcc.RangeSlider(id="embed-u-range", min=-6.28, max=6.28, step=0.1, value=[-3.14, 3.14], tooltip={"placement": "bottom"})], style={"marginTop": 8}),
                            html.Div([html.Label("v range"), dcc.RangeSlider(id="embed-v-range", min=-6.28, max=6.28, step=0.1, value=[-3.14, 3.14], tooltip={"placement": "bottom"})], style={"marginTop": 8}),
                            html.Div([html.Label("Domain mask (optional, in u, v)"), dcc.Input(id="mask-embed-uv", type="text", value="", style={"width": "100%"})], style={"marginTop": 8}),
                        ],
                        style={"marginBottom": 16, "display": "none"},
                    ),

                    # Curve r(t)
                    html.Fieldset(
                        id="panel-curve_t",
                        children=[
                            html.Legend("Parametric curve: r(t) = (x(t), y(t), z(t)) with t ∈ ℝ → ℝ³"),
                            html.Div([html.Label("x(t) ="), dcc.Input(id="curve-x-t", type="text", value="cos(5*t) * (2 + cos(t))", style={"width": "100%"})],),
                            html.Div([html.Label("y(t) ="), dcc.Input(id="curve-y-t", type="text", value="sin(5*t) * (2 + cos(t))", style={"width": "100%"})], style={"marginTop": 6}),
                            html.Div([html.Label("z(t) ="), dcc.Input(id="curve-z-t", type="text", value="sin(t)", style={"width": "100%"})], style={"marginTop": 6}),
                            html.Div([html.Label("t range"), dcc.RangeSlider(id="t-range", min=-12.56, max=12.56, step=0.05, value=[-6.28, 6.28], tooltip={"placement": "bottom"})], style={"marginTop": 8}),
                        ],
                        style={"marginBottom": 16, "display": "none"},
                    ),

                    # General f: R^n -> R^m (auto)
                    html.Fieldset(
                        id="panel-auto_nm",
                        children=[
                            html.Legend("General mapping: f: ℝⁿ → ℝᵐ (auto visualization)"),
                            html.Div([
                                html.Div([html.Label("n (domain ℝⁿ)"), dcc.Input(id="auto-n", type="number", min=1, max=6, step=1, value=3)], style={"marginRight": 12}),
                                html.Div([html.Label("m (codomain ℝᵐ)"), dcc.Input(id="auto-m", type="number", min=1, max=3, step=1, value=2)], style={"marginRight": 12}),
                                html.Div([html.Label("Vary variables (choose 1–3)"), dcc.Dropdown(id="auto-vary-vars", multi=True, options=[{"label": f"x{i}", "value": i} for i in range(1,7)], value=[1,2])], style={"minWidth": 280, "flex": 1}),
                            ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px"}),
                            html.Div([html.Label("f(x1, ..., xn) as comma-separated expressions (length = m). Example: x1 - x3, x2 + x3"), dcc.Textarea(id="auto-exprs", value="x1 - x3, x2 + x3", style={"width": "100%", "height": 70})], style={"marginTop": 8}),
                            html.Div([html.Label("Domain mask (optional, over x1..xn; only applied when 2 variables vary)"), dcc.Input(id="auto-mask", type="text", value="", style={"width": "100%"})], style={"marginTop": 8}),

                            html.Div([
                                html.Div([html.Label("Grid / sample count"), dcc.Slider(id="auto-resolution", min=10, max=200, step=5, value=100, tooltip={"placement": "bottom"})], style={"minWidth": 260, "flex": 1}),
                                html.Div([html.Label("Arrow scale (for vector fields)"), dcc.Slider(id="auto-arrow-scale", min=0.05, max=1.0, step=0.05, value=0.2, tooltip={"placement": "bottom"})], style={"minWidth": 260, "flex": 1}),
                            ], style={"display": "flex", "gap": "12px", "marginTop": 8, "flexWrap": "wrap"}),

                            html.Div(id="auto-var-controls", children=[
                                html.Div([html.Label("x1 range"), dcc.RangeSlider(id="auto-range-1", min=-6.28, max=6.28, step=0.1, value=[-3.14, 3.14])], id="auto-range-wrap-1"),
                                html.Div([html.Label("x2 range"), dcc.RangeSlider(id="auto-range-2", min=-6.28, max=6.28, step=0.1, value=[-3.14, 3.14])], id="auto-range-wrap-2"),
                                html.Div([html.Label("x3 range"), dcc.RangeSlider(id="auto-range-3", min=-6.28, max=6.28, step=0.1, value=[-3.14, 3.14])], id="auto-range-wrap-3"),
                                html.Div([html.Label("x4 range"), dcc.RangeSlider(id="auto-range-4", min=-6.28, max=6.28, step=0.1, value=[-3.14, 3.14])], id="auto-range-wrap-4"),
                                html.Div([html.Label("x5 range"), dcc.RangeSlider(id="auto-range-5", min=-6.28, max=6.28, step=0.1, value=[-3.14, 3.14])], id="auto-range-wrap-5"),
                                html.Div([html.Label("x6 range"), dcc.RangeSlider(id="auto-range-6", min=-6.28, max=6.28, step=0.1, value=[-3.14, 3.14])], id="auto-range-wrap-6"),

                                html.Div([html.Label("x1 fixed"), dcc.Input(id="auto-fixed-1", type="number", value=0.0)], id="auto-fixed-wrap-1"),
                                html.Div([html.Label("x2 fixed"), dcc.Input(id="auto-fixed-2", type="number", value=0.0)], id="auto-fixed-wrap-2"),
                                html.Div([html.Label("x3 fixed"), dcc.Input(id="auto-fixed-3", type="number", value=0.0)], id="auto-fixed-wrap-3"),
                                html.Div([html.Label("x4 fixed"), dcc.Input(id="auto-fixed-4", type="number", value=0.0)], id="auto-fixed-wrap-4"),
                                html.Div([html.Label("x5 fixed"), dcc.Input(id="auto-fixed-5", type="number", value=0.0)], id="auto-fixed-wrap-5"),
                                html.Div([html.Label("x6 fixed"), dcc.Input(id="auto-fixed-6", type="number", value=0.0)], id="auto-fixed-wrap-6"),
                            ], style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(260px, 1fr))", "gap": "8px", "marginTop": 8}),
                        ],
                        style={"marginBottom": 16, "display": "none"},
                    ),

                    html.Div(
                        [
                            html.Label("Grid resolution"),
                            dcc.Slider(id="resolution", min=10, max=300, step=5, value=120, tooltip={"placement": "bottom"}),
                            html.Label("Render style"),
                            dcc.RadioItems(
                                id="render-style",
                                options=[
                                    {"label": "Surface", "value": "surface"},
                                    {"label": "Wireframe", "value": "wireframe"},
                                ],
                                value="surface",
                                inline=True,
                            ),
                        ],
                        style={"marginTop": 8},
                    ),

                    html.Div(id="error", style={"color": "crimson", "whiteSpace": "pre-wrap", "marginTop": 8}),
            ]),

            dcc.Graph(id="graph-3d", style={"height": "80vh"}),

            html.Div([
                html.H4("Help: syntax & tips"),
                html.Ul([
                    html.Li("Functions use Python/SymPy syntax: sin, cos, tan, exp, log, sqrt, abs, asin, acos, atan, sinh, cosh, tanh, floor, ceil."),
                    html.Li("Constants: pi, E."),
                    html.Li("Use ** for exponentiation (e.g., x**2)."),
                    html.Li("Domain masks can restrict the surface: e.g., x**2 + y**2 <= 25 or (u**2 + v**2 <= 9)."),
                    html.Li("Modes correspond to function types: ℝ²→ℝ (scalar, height-field) vs ℝ²→ℝ³ (embedded) vs ℝ→ℝ³ (curve)."),
                ]),
            ], style={"marginTop": 12})
        ],
        style={"padding": 16, "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial"}
    )
