from dash import Input, Output, State, no_update
import numpy as np
import plotly.graph_objects as go

from graph3d.core.parser import make_func_xy, make_mask_xy, make_func_uv, make_mask_uv, make_funcs_embed_uv, make_funcs_curve_t, make_funcs_general, make_mask_general
from graph3d.core.samplers import grid_xy, grid_uv, lin_t
from graph3d.core.plotters import surface_from_xyz, wireframe_from_xyz, line_from_xyz, quiver2d, quiver3d

def _hide_all():
    return [{"display": "none"}] * 5

def _show(panel_id):
    # order: scalar_xy, param_height, param_embed, curve_t, auto_nm
    style = [{"display": "none"}] * 5
    idx = {"scalar_xy": 0, "param_height": 1, "param_embed": 2, "curve_t": 3, "auto_nm": 4}[panel_id]
    style[idx] = {}
    return style

def register_callbacks(app):
    @app.callback(
        Output("panel-scalar_xy", "style"),
        Output("panel-param_height", "style"),
        Output("panel-param_embed", "style"),
        Output("panel-curve_t", "style"),
        Output("panel-auto_nm", "style"),
        Input("mode", "value"),
    )
    def toggle_panels(mode):
        return _show(mode)

    @app.callback(
        Output("graph-3d", "figure"),
        Output("error", "children"),
        Input("mode", "value"),
        Input("scalar-fxy", "value"),
        Input("x-range", "value"),
        Input("y-range", "value"),
        Input("mask-xy", "value"),
        Input("param-z-uv", "value"),
        Input("u-range", "value"),
        Input("v-range", "value"),
        Input("mask-uv", "value"),
        Input("embed-x-uv", "value"),
        Input("embed-y-uv", "value"),
        Input("embed-z-uv", "value"),
        Input("embed-u-range", "value"),
        Input("embed-v-range", "value"),
        Input("mask-embed-uv", "value"),
        Input("curve-x-t", "value"),
        Input("curve-y-t", "value"),
        Input("curve-z-t", "value"),
        Input("t-range", "value"),
        Input("resolution", "value"),
        Input("render-style", "value"),
        # Auto n→m
        Input("auto-n", "value"),
        Input("auto-m", "value"),
        Input("auto-vary-vars", "value"),
        Input("auto-exprs", "value"),
        Input("auto-mask", "value"),
        Input("auto-resolution", "value"),
        Input("auto-arrow-scale", "value"),
        Input("auto-range-1", "value"), Input("auto-range-2", "value"), Input("auto-range-3", "value"),
        Input("auto-range-4", "value"), Input("auto-range-5", "value"), Input("auto-range-6", "value"),
        Input("auto-fixed-1", "value"), Input("auto-fixed-2", "value"), Input("auto-fixed-3", "value"),
        Input("auto-fixed-4", "value"), Input("auto-fixed-5", "value"), Input("auto-fixed-6", "value"),
        prevent_initial_call=False
    )
    def update_figure(
        mode,
        fxy, x_range, y_range, mask_xy_expr,
        z_uv, u_range, v_range, mask_uv_expr,
        x_uv, y_uv, z_uv_embed, u_range_e, v_range_e, mask_e_expr,
        xt, yt, zt, t_range, resolution, render_style,
        # auto
        an, am, vary_vars, exprs, auto_mask, auto_res, arrow_scale,
        r1, r2, r3, r4, r5, r6,
        fx1, fx2, fx3, fx4, fx5, fx6
    ):
        try:
            if mode == "scalar_xy":
                xs = np.linspace(x_range[0], x_range[1], int(resolution))
                ys = np.linspace(y_range[0], y_range[1], int(resolution))
                X, Y = grid_xy(xs, ys)
                f = make_func_xy(fxy)
                Z = f(X, Y)

                if (mask_xy_expr or "").strip():
                    m = make_mask_xy(mask_xy_expr)(X, Y)
                    X, Y, Z = X[m], Y[m], Z[m]
                    if render_style == "surface":
                        fig = go.Figure(go.Mesh3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), opacity=0.95, intensity=Z.flatten(), colorscale="Viridis", showscale=True, alphahull=0))
                    else:
                        fig = wireframe_from_xyz(X, Y, Z, as_points=True)
                else:
                    if render_style == "surface":
                        fig = surface_from_xyz(X, Y, Z)
                    else:
                        fig = wireframe_from_xyz(X, Y, Z)

            elif mode == "param_height":
                us = np.linspace(u_range[0], u_range[1], int(resolution))
                vs = np.linspace(v_range[0], v_range[1], int(resolution))
                U, V = grid_uv(us, vs)
                fz = make_func_uv(z_uv)
                Z = fz(U, V)
                X, Y = U, V

                if (mask_uv_expr or "").strip():
                    m = make_mask_uv(mask_uv_expr)(U, V)
                    X, Y, Z = X[m], Y[m], Z[m]
                    if render_style == "surface":
                        fig = go.Figure(go.Mesh3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), opacity=0.95, intensity=Z.flatten(), colorscale="Viridis", showscale=True, alphahull=0))
                    else:
                        fig = wireframe_from_xyz(X, Y, Z, as_points=True)
                else:
                    if render_style == "surface":
                        fig = surface_from_xyz(X, Y, Z, x=X, y=Y)
                    else:
                        fig = wireframe_from_xyz(X, Y, Z)

            elif mode == "param_embed":
                us = np.linspace(u_range_e[0], u_range_e[1], int(resolution))
                vs = np.linspace(v_range_e[0], v_range_e[1], int(resolution))
                U, V = grid_uv(us, vs)
                fx, fy, fz = make_funcs_embed_uv(x_uv, y_uv, z_uv_embed)
                X = fx(U, V); Y = fy(U, V); Z = fz(U, V)

                if (mask_e_expr or "").strip():
                    m = make_mask_uv(mask_e_expr)(U, V)
                    X, Y, Z = X[m], Y[m], Z[m]
                    if render_style == "surface":
                        fig = go.Figure(go.Mesh3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), opacity=0.95, intensity=Z.flatten(), colorscale="Viridis", showscale=True, alphahull=0))
                    else:
                        fig = wireframe_from_xyz(X, Y, Z, as_points=True)
                else:
                    if render_style == "surface":
                        fig = surface_from_xyz(X, Y, Z, parametric=True)
                    else:
                        fig = wireframe_from_xyz(X, Y, Z)

            elif mode == "curve_t":
                ts = lin_t(t_range[0], t_range[1], int(resolution))
                fx, fy, fz = make_funcs_curve_t(xt, yt, zt)
                X = fx(ts); Y = fy(ts); Z = fz(ts)
                fig = line_from_xyz(X, Y, Z)

            else:  # auto_nm
                n = int(max(1, min(6, an or 1)))
                m = int(max(1, min(3, am or 1)))
                vv = sorted(set([v for v in (vary_vars or []) if isinstance(v, int) and 1 <= v <= n]))
                if len(vv) == 0: vv = [1]
                if len(vv) > 3: vv = vv[:3]
                k = len(vv)

                funcs = make_funcs_general(exprs or "", n, m)

                ranges = {1: r1, 2: r2, 3: r3, 4: r4, 5: r5, 6: r6}
                fixeds = {1: fx1, 2: fx2, 3: fx3, 4: fx4, 5: fx5, 6: fx6}

                if k == 1:
                    i = vv[0]
                    lo, hi = ranges[i] if ranges[i] else (-3.14, 3.14)
                    t = lin_t(lo, hi, int(auto_res))
                    args = [ (t if (j==i) else np.full_like(t, float(fixeds[j] or 0.0))) for j in range(1, n+1) ]
                    outs = [f(*args) for f in funcs]
                    if m == 1:
                        X = t; Y = outs[0]; Z = np.zeros_like(X)
                        fig = line_from_xyz(X, Y, Z)
                    elif m == 2:
                        X = outs[0]; Y = outs[1]; Z = np.zeros_like(X)
                        fig = line_from_xyz(X, Y, Z)
                    elif m == 3:
                        X = outs[0]; Y = outs[1]; Z = outs[2]
                        fig = line_from_xyz(X, Y, Z)
                    else:
                        raise ValueError("m>3 not directly visualizable.")
                elif k == 2:
                    i, j = vv[0], vv[1]
                    lox, hix = ranges[i] if ranges[i] else (-3.14, 3.14)
                    loy, hiy = ranges[j] if ranges[j] else (-3.14, 3.14)
                    xs = np.linspace(lox, hix, int(auto_res))
                    ys = np.linspace(loy, hiy, int(auto_res))
                    A, B = grid_xy(xs, ys)
                    def arg_for(idx):
                        if idx == i: return A
                        if idx == j: return B
                        return float(fixeds[idx] or 0.0)
                    args = [arg_for(ii+1) for ii in range(n)]
                    outs = [f(*args) for f in funcs]

                    if (auto_mask or "").strip():
                        msk = make_mask_general(auto_mask, n)(*args)
                    else:
                        msk = None

                    if m == 1:
                        Z = outs[0]
                        if msk is not None:
                            X, Y, Z = A[msk], B[msk], Z[msk]
                            if render_style == "surface":
                                fig = go.Figure(go.Mesh3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), opacity=0.95, intensity=Z.flatten(), colorscale="Viridis", showscale=True, alphahull=0))
                            else:
                                fig = wireframe_from_xyz(X, Y, Z, as_points=True)
                        else:
                            if render_style == "surface":
                                fig = surface_from_xyz(A, B, Z)
                            else:
                                fig = wireframe_from_xyz(A, B, Z)
                    elif m == 2:
                        U, V = outs[0], outs[1]
                        if msk is not None:
                            Xp, Yp = A[msk], B[msk]
                            Up, Vp = U[msk], V[msk]
                        else:
                            Xp, Yp, Up, Vp = A, B, U, V
                        fig = quiver2d(Xp, Yp, Up, Vp, scale=float(arrow_scale or 0.2))
                    elif m == 3:
                        X, Y, Z = outs[0], outs[1], outs[2]
                        if msk is not None:
                            X, Y, Z = X[msk], Y[msk], Z[msk]
                        fig = surface_from_xyz(X, Y, Z) if (X.ndim==2 and Y.ndim==2 and Z.ndim==2) else wireframe_from_xyz(X, Y, Z, as_points=True)
                    else:
                        raise ValueError("m>3 not directly visualizable.")
                else:
                    if m == 3 and k == 3:
                        i, j, k3 = vv[0], vv[1], vv[2]
                        lo1, hi1 = ranges[i] if ranges[i] else (-3.14, 3.14)
                        lo2, hi2 = ranges[j] if ranges[j] else (-3.14, 3.14)
                        lo3, hi3 = ranges[k3] if ranges[k3] else (-3.14, 3.14)
                        xs = np.linspace(lo1, hi1, max(8, int(auto_res/4)))
                        ys = np.linspace(lo2, hi2, max(8, int(auto_res/4)))
                        zs = np.linspace(lo3, hi3, max(8, int(auto_res/4)))
                        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
                        def arg_for(idx):
                            if idx == i: return X
                            if idx == j: return Y
                            if idx == k3: return Z
                            return float(fixeds[idx] or 0.0)
                        args = [arg_for(ii+1) for ii in range(n)]
                        outs = [f(*args) for f in funcs]
                        U, V, W = outs[0], outs[1], outs[2]
                        fig = quiver3d(X, Y, Z, U, V, W, scale=float(arrow_scale or 0.2))
                    else:
                        raise ValueError("Choose 1–3 varying variables. For vector fields in ℝ³→ℝ³, select 3.")
            fig.update_layout(
                scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            return fig, ""
        except Exception as e:
            return no_update, f"Error: {e}"

    @app.callback(
        Output("auto-range-wrap-1", "style"), Output("auto-fixed-wrap-1", "style"),
        Output("auto-range-wrap-2", "style"), Output("auto-fixed-wrap-2", "style"),
        Output("auto-range-wrap-3", "style"), Output("auto-fixed-wrap-3", "style"),
        Output("auto-range-wrap-4", "style"), Output("auto-fixed-wrap-4", "style"),
        Output("auto-range-wrap-5", "style"), Output("auto-fixed-wrap-5", "style"),
        Output("auto-range-wrap-6", "style"), Output("auto-fixed-wrap-6", "style"),
        Input("auto-n", "value"), Input("auto-vary-vars", "value"),
    )
    def toggle_auto_controls(n, vary_vars):
        n = int(max(1, min(6, n or 1)))
        vv = set(vary_vars or [])
        def style_for(i):
            if i > n:
                return {"display": "none"}, {"display": "none"}
            if i in vv:
                return {}, {"display": "none"}
            return {"display": "none"}, {}
        s = []
        for i in range(1,7):
            s.extend(style_for(i))
        return tuple(s)
