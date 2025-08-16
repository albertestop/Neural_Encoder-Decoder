import numpy as np
import plotly.graph_objects as go
import PIL.Image as Image
import base64
import io
from typing import List
import dash_bootstrap_components as dbc
from dash import dcc, html


def bar_layout(n_frames):
    _pct = [0, 0.25, 0.5, 0.75, 1]
    MARKS = {int(p * n_frames): f"{int(p * n_frames)}s" for p in _pct}
    bar = dbc.Row([
        dbc.Col(dcc.Slider(
            id="time-slider",
            min=0,
            max=n_frames - 1,
            step=1,
            value=0,
            marks=MARKS,  # sparse numeric labels
            tooltip={"placement": "bottom", "always_visible": True},
        ), width=10),
        dbc.Col(html.Button("▶", id="play-btn", n_clicks=0,
                            className="btn btn-secondary",
                            **{"aria-label": "Play/Pause"}), width=1),
        dbc.Col(html.Div(id="t-label"), width=1),
    ], align="center", class_name="mt-2")
    return bar

class Video():
    def __init__(self, data, N_frames):
        self.data = data

    def frame_to_b64(self, frame: np.ndarray) -> str:
        """Convert a single RGB uint8 frame → base64‑encoded PNG suitable for <img>."""
        buff = io.BytesIO()
        Image.fromarray(frame).save(buff, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode()

    def __call__(self, i):
        return self.frame_to_b64(self.data[i])


def rp_plot(data, N_FRAMES):
    figure = go.Figure(
        data=[go.Image(z=data)],
        layout=go.Layout(
            template="plotly_white",
            xaxis=dict(showticklabels=False, range=[0, N_FRAMES]),
            yaxis=dict(showticklabels=False),
            margin=dict(l=30, r=10, t=10, b=30),
            height=150,
        ),
    )
    return figure


def line_plot(data, N_FRAMES):
    figure = go.Figure(
        data=[go.Scatter(y=data, mode="lines", line=dict(color="gold"))],
        layout=go.Layout(
            template="plotly_white",
            xaxis=dict(range=[0, N_FRAMES]),
            margin=dict(l=30, r=10, t=10, b=30),
            height=200,
            showlegend=False,
        ),
    )
    return figure


def gen_layout(plot, data_par):
    N_frames = data_par.shape[-1]
    N_frames = 100
    if plot[1] == 'line':
        return line_plot(data_par, N_frames)
    elif plot[1] == 'rp':
        return rp_plot(data_par, N_frames)
    elif plot[1] == 'video':
        return Video(data_par, N_frames)


def gen_figures(plots, data):
    figs = []
    for plot, data_par in zip(plots, data):
        figs.append(gen_layout(plot, data_par))

    return figs