"""
Dash GUI for synchronized video frame + wheel‑speed playback + filmstrip
────────────────────────────────────────────────────────────────────────
Change log
──────────
• Initial optimisations and two‑plot layout (20 Jun 2025)
• Sparse slider marks (20 Jun 2025)
• Added horizontal filmstrip of frames with synced red cursor (option C) (20 Jun 2025)
"""

from __future__ import annotations

import base64
import io
from typing import List

import numpy as np
from PIL import Image
import dash
from dash import dcc, html, Output, Input, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# ────────────────────────────────────────────────────────────────────────
# Demo data – swap with your real stream
# ────────────────────────────────────────────────────────────────────────
FPS: int = 10
DURATION_SEC: int = 10
N_FRAMES: int = FPS * DURATION_SEC

HEIGHT, WIDTH = 180, 320  # original frame size

# Fake RGB video
np.random.seed(0)
video = (255 * np.random.rand(N_FRAMES, HEIGHT, WIDTH, 3)).astype(np.uint8)

t_video = np.linspace(0, DURATION_SEC, N_FRAMES, endpoint=False)
speed = 50 + 10 * np.sin(2 * np.pi * 0.5 * t_video)


# ────────────────────────────────────────────────────────────────────────
# Helpers – encode individual frames + build filmstrip image
# ────────────────────────────────────────────────────────────────────────

def frame_to_b64(frame: np.ndarray) -> str:
    """RGB uint8 frame → base64‑encoded PNG suitable for <img>."""
    buff = io.BytesIO()
    Image.fromarray(frame).save(buff, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode()


# Pre‑encode video frames for quick access in the callback
frames_b64: List[str] = [frame_to_b64(f) for f in video]

# Build a compact “filmstrip” image: each column = one frame (down‑sampled)
THUMB_H = 60  # pixels tall in the strip
cols = []
for frame in video:
    # Down‑sample the frame to (THUMB_H × 1) then keep as column
    col = Image.fromarray(frame).resize((1, THUMB_H), Image.Resampling.LANCZOS)
    cols.append(np.asarray(col))
filmstrip = np.concatenate(cols, axis=1)  # shape (THUMB_H, N_FRAMES, 3)

# ────────────────────────────────────────────────────────────────────────
# Static Plotly figures (will patch red cursor each tick)
# ────────────────────────────────────────────────────────────────────────
# Speed plot
BASE_SPEED = go.Figure(
    data=[go.Scatter(y=speed, mode="lines", line=dict(color="gold"))],
    layout=go.Layout(
        template="plotly_white",
        xaxis_title="Frame #",
        yaxis_title="Wheel speed (Hz)",
        xaxis=dict(range=[0, N_FRAMES]),
        margin=dict(l=30, r=10, t=10, b=30),
        height=300,
    ),
)

# Filmstrip image plot
BASE_STRIP = go.Figure(
    data=[go.Image(z=filmstrip)],
    layout=go.Layout(
        template="plotly_white",
        xaxis=dict(showticklabels=False, range=[0, N_FRAMES]),
        yaxis=dict(showticklabels=False),
        margin=dict(l=30, r=10, t=10, b=30),
        height=150,
    ),
)

STORE_ID = "playing-flag"

# Sparse marks at 0 ‑ ¼ ‑ ½ ‑ ¾ ‑ 1 of duration
_pct = [0, 0.25, 0.5, 0.75, 1]
MARKS = {int(p * (N_FRAMES - 1)): f"{p * DURATION_SEC:.1f}s" for p in _pct}

# ────────────────────────────────────────────────────────────────────────
# Layout
# ────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(fluid=True, children=[
    # Slider + play button row
    dbc.Row([
        dbc.Col(dcc.Slider(
            id="time-slider",
            min=0,
            max=N_FRAMES - 1,
            step=1,
            value=0,
            marks=MARKS,
            tooltip={"placement": "bottom", "always_visible": True},
        ), width=10),
        dbc.Col(html.Button("▶", id="play-btn", n_clicks=0,
                            className="btn btn-secondary",
                            **{"aria-label": "Play/Pause"}), width=1),
        dbc.Col(html.Div(id="t-label"), width=1),
    ], align="center", class_name="mt-2"),

    # Centre video
    dbc.Row([
        dbc.Col(html.Img(id="frame", style={
            "max-width": "100%",
            "display": "block",
            "margin": "0 auto",
        }), width=6),
    ], justify="center", class_name="mt-3"),

    # Speed plot full‑width
    dbc.Row([
        dbc.Col(dcc.Graph(id="speed-plot", figure=BASE_SPEED,
                          config={"displayModeBar": False}), width=12),
    ], class_name="mt-3"),

    # Filmstrip plot full‑width (height 150)
    dbc.Row([
        dbc.Col(dcc.Graph(id="strip-plot", figure=BASE_STRIP,
                          config={"displayModeBar": False}), width=12),
    ], class_name="mt-3"),

    dcc.Interval(id="interval-js", interval=int(1000 / FPS), n_intervals=0),
    dcc.Store(id=STORE_ID, data=False),
])

# ────────────────────────────────────────────────────────────────────────
# Callbacks
# ────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("frame", "src"),
    Output("speed-plot", "figure"),
    Output("strip-plot", "figure"),
    Output("t-label", "children"),
    Input("time-slider", "value"),
)
def update_views(i: int):
    """Update current PNG frame and repaint red cursor on both plots."""
    img_src = frame_to_b64(video[i])  # or frames_b64[i]

    # Helper to copy fig + patch red cursor
    def with_cursor(fig_in):
        fig = fig_in.to_dict()
        fig["layout"]["shapes"] = [dict(
            type="line", x0=i, x1=i, y0=0, y1=1,
            xref="x", yref="paper", line=dict(color="red"),
        )]
        return fig

    return img_src, with_cursor(BASE_SPEED), with_cursor(BASE_STRIP), f"{i / FPS:.2f} s"


@app.callback(
    Output(STORE_ID, "data"),
    Output("play-btn", "children"),
    Input("play-btn", "n_clicks"),
    State(STORE_ID, "data"),
    prevent_initial_call=True,
)
def toggle_play(_clicks: int, playing: bool):
    playing = not playing
    return playing, ("❚❚" if playing else "▶")


app.clientside_callback(
    """
    function(n_intervals, playing, current, max_frame){
        if(!playing){ return window.dash_clientside.no_update; }
        return (current + 1) % (max_frame + 1);
    }
    """,
    Output("time-slider", "value"),
    Input("interval-js", "n_intervals"),
    State(STORE_ID, "data"),
    State("time-slider", "value"),
    State("time-slider", "max"),
)

# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)
