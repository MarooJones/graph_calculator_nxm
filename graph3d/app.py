#!/usr/bin/env python3
import dash
from dash import html
from graph3d.components.layout import build_layout
from graph3d.components.callbacks import register_callbacks

external_scripts = []
external_stylesheets = []

app = dash.Dash(__name__, external_scripts=external_scripts, external_stylesheets=external_stylesheets)
server = app.server  # for deployment

app.layout = build_layout()

register_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True)
