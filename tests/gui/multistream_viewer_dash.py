"""
Dash GUI for synchronized video frame + wheel‑speed playback.
Implements:
  • Single BASE_FIG reused every frame (lighter traffic)
  • Clientside interval + dcc.Store for ultra‑smooth play/pause
  • ctx.triggered_id (Dash ≥2.12)
  • Minor UX polish (ARIA label, Bootstrap button)

For long recordings replace the in‑memory `frames_b64` list with a
/frames/<i>.jpg route or <video> component – see README.
"""

import dash
from dash import dcc, html, Output, Input, State, ctx
import dash_bootstrap_components as dbc

import utils_gui as utils
from load_data import load_data
import config

FPS: int = 10
N_FRAMES: int = 100


# LOAD DATA

data = load_data(config.plots)


# FIGURE GENERATION

BASE_FIG = []

BASE_FIG = utils.gen_figures(config.plots, data)

STORE_ID = "playing-flag"  # shared between callbacks


# LAYOUT

children = [utils.bar_layout(N_FRAMES)]

for i, plot in enumerate(config.plots):
    if plot[1] == 'line':
        children.append(dbc.Row([dbc.Col(dcc.Graph(id=str(i), figure=BASE_FIG[i], config={"displayModeBar": False}), width=12),], class_name="mt-3"))

    elif plot[1] == 'rp':
        children.append(dbc.Row([dbc.Col(dcc.Graph(id=str(i), figure=BASE_FIG[i], config={"displayModeBar": False}), width=12),], class_name="mt-3"))

    elif plot[1] == 'video':
        children.append(dbc.Row([dbc.Col(html.Img(id=str(i), style={"width": "100%"}), width=10),], justify="center", class_name="mt-3"))

children.append(dcc.Interval(id="interval-js", interval=int(1000 / FPS), n_intervals=0))
children.append(dcc.Store(id=STORE_ID, data=False))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(fluid=True, children=children)


# CALLBACKS

callbacks = []

for i, plot in enumerate(config.plots):
    if plot[1] == 'line':
        callbacks.append(Output(str(i), "figure"))
    elif plot[1] == 'rp':
        callbacks.append(Output(str(i), "figure"))
    elif plot[1] == 'video':
        callbacks.append(Output(str(i), "src"))

callbacks.append(Output("t-label", "children"))
callbacks.append(Input("time-slider", "value"))

@app.callback(*callbacks)

def update_views(i: int):

    # Helper to clone base fig and patch the red cursor
    def with_cursor(base_fig):
        fig_dict = base_fig.to_dict()
        fig_dict["layout"]["shapes"] = [dict(
            type="line", x0=i, x1=i, y0=0, y1=1,
            xref="x", yref="paper", line=dict(color="red"),
        )]
        return fig_dict
    
    returns = []
    for k, plot in enumerate(config.plots):
        if plot[1] == 'line':
            returns.append(with_cursor(BASE_FIG[k]))
        elif plot[1] == 'filmstrip':
            returns.append(with_cursor(BASE_FIG[k]))
        elif plot[1] == 'video':
            returns.append(BASE_FIG[k](i))
    returns.append(f"{i / FPS:.2f} s")

    return tuple(returns)


@app.callback(
    Output(STORE_ID, "data"),
    Output("play-btn", "children"),
    Input("play-btn", "n_clicks"),
    State(STORE_ID, "data"),
    prevent_initial_call=True,
)
def toggle_play(_clicks: int, playing: bool):
    """Flip play/pause flag and change the button glyph."""
    playing = not playing
    return playing, ("❚❚" if playing else "▶")


# Clientside callback: runs in the browser every Interval tick
app.clientside_callback(
    """
    function(n_intervals, playing, current, max_frame){
        if(!playing){
            return window.dash_clientside.no_update;
        }
        return (current + 1) % (max_frame + 1);
    }
    """,
    Output("time-slider", "value"),
    Input("interval-js", "n_intervals"),
    State(STORE_ID, "data"),
    State("time-slider", "value"),
    State("time-slider", "max"),
)



if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)
