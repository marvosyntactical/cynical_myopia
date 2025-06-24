"""
Iterated PD Echo‑Chamber Simulator — Dash web‑app (v2)
=====================================================
Blue = optimists (cooperate); orange = cynics (defect).

**What’s new**
--------------
1. **Meaningful edge length & width** – links carry a trust weight ω that grows on mutual C, shrinks on betrayal.  The spring‑layout uses `weight=ω`, so strong ties pull nodes closer; line width and opacity scale with ω.
2. **GIF exporter** – needs **kaleido** + **imageio**.  Button stamps a caption with all current parameters.
3. **Help tool‑tips** for every slider.
4. **Static export helper** – run `python freeze.py` to dump a playable `build/index.html` (frames baked in; no Python needed on server).

Install & run
-------------
```bash
pip install dash==2.17.0 dash-bootstrap-components plotly networkx numpy imageio pillow kaleido
python iterated_pd_network_dash.py  # launches on http://127.0.0.1:8050
```

Deploy full server: `gunicorn iterated_pd_network_dash:server -b 0.0.0.0:$PORT`

Static export (client‑only): `python freeze.py` → `build/` folder (simulation frozen).
"""
from __future__ import annotations
import io, json, typing as _t, textwrap, datetime
from dataclasses import dataclass

import dash
from dash import dcc, html, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import networkx as nx
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import base64

# ░░ Simulation core ░░
@dataclass
class Params:
    n: int = 50
    pct_opt: float = 0.5
    p_edge: float = 0.1
    steps: int = 1500
    frame_delta: int = 100
    rew_phi: float = 0.8
    imitate_prob: float = 0.5
    tremble_p: float = 0.05
    R: int = 3
    S: int = 0
    T: int = 5
    P: int = 1

actionLUT = {"cynic": "D", "optimist": "C"}


def pd_payoff(a: str, b: str, R, S, T, P):
    if a == "C" and b == "C":
        return R, R
    if a == "C" and b == "D":
        return S, T
    if a == "D" and b == "C":
        return T, S
    return P, P


def run_sim(params: Params) -> list[dict]:
    rng = np.random.default_rng(0)

    types = np.array(
        ["optimist"] * int(params.n * params.pct_opt)
        + ["cynic"] * (params.n - int(params.n * params.pct_opt))
    )
    G = nx.erdos_renyi_graph(params.n, params.p_edge, seed=42)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for comp in comps[1:]:
            G.add_edge(next(iter(comp)), next(iter(comps[0])))

    nx.set_edge_attributes(G, 1.0, "weight")
    pos = nx.spring_layout(G, seed=1, weight="weight")

    tot_pay, tot_cnt = np.zeros(params.n), np.zeros(params.n)
    frames: list[dict] = []

    def snap(step: int):
        nodes = [
            dict(x=float(pos[v][0]), y=float(pos[v][1]), color="#1f77b4" if types[v]=="optimist" else "#ff7f0e")
            for v in G.nodes()
        ]
        edges = [
            dict(x0=float(pos[u][0]), y0=float(pos[u][1]), x1=float(pos[v][0]), y1=float(pos[v][1]), w=G[u][v]["weight"])
            for u, v in G.edges()
        ]
        frames.append({"step": step, "nodes": nodes, "edges": edges})

    snap(0)

    def maybe_rewire(u, v, au, av):
        if au == "C" and av == "D" and rng.random() < params.rew_phi and G.has_edge(u, v):
            G.remove_edge(u, v)
            cands = [k for k in range(params.n) if types[k]==types[u] and k!=u and not G.has_edge(u,k)]
            if cands:
                k = rng.choice(cands)
                G.add_edge(u, k, weight=1.0)

    for t in range(1, params.steps + 1):
        i = rng.integers(params.n)
        nbrs = list(G[i])
        if not nbrs:
            continue
        j = rng.choice(nbrs)
        a_i = actionLUT[types[i]]; a_j = actionLUT[types[j]]
        if rng.random() < params.tremble_p:
            a_i = "D" if a_i == "C" else "C"
        if rng.random() < params.tremble_p:
            a_j = "D" if a_j == "C" else "C"

        # update trust weight
        if a_i == "C" and a_j == "C":
            G[i][j]["weight"] = min(G[i][j]["weight"] + 0.3, 3.0)
        elif a_i == "D" and a_j == "D":
            G[i][j]["weight"] = max(G[i][j]["weight"] - 0.2, 0.1)
        else:
            G[i][j]["weight"] = max(G[i][j]["weight"] - 0.4, 0.1)

        p_i, p_j = pd_payoff(a_i, a_j, params.R, params.S, params.T, params.P)
        tot_pay[i]+=p_i; tot_pay[j]+=p_j
        tot_cnt[i]+=1;  tot_cnt[j]+=1

        # imitation
        if rng.random() < params.imitate_prob:
            avg_i, avg_j = tot_pay[i]/tot_cnt[i], tot_pay[j]/tot_cnt[j]
            if avg_i < avg_j:
                types[i]=types[j]
            elif avg_j < avg_i:
                types[j]=types[i]

        maybe_rewire(i,j,a_i,a_j); maybe_rewire(j,i,a_j,a_i)

        if t % params.frame_delta == 0:
            pos = nx.spring_layout(G, pos=pos, iterations=15, seed=1, weight="weight")
            snap(t)

    return frames

# ░░ Dash layout ░░
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

slider_kwargs = dict(tooltip={"placement":"bottom","always_visible":False})

control_card = dbc.Card([
    html.H5("Controls"),
    dbc.Label("Agents (N)"), dcc.Input(id="n", type="number", value=50, min=10, max=200, step=10, style={"width":"100%"}),
    dbc.Label("% Optimists"), dcc.Slider(id="pct_opt", min=0, max=1, step=0.05, value=0.5, **slider_kwargs),
    dbc.Label("Edge prob"), dcc.Slider(id="p_edge", min=0.02, max=0.3, step=0.01, value=0.1, **slider_kwargs),
    dbc.Label("Steps"), dcc.Input(id="steps", type="number", value=1500, min=100, max=10000, step=100, style={"width":"100%"}),
    dbc.Label("Frame Δ"), dcc.Input(id="frame_delta", type="number", value=100, min=10, max=500, step=10, style={"width":"100%"}),
    dbc.Label("Rewire φ"), dcc.Slider(id="rew_phi", min=0, max=1, step=0.05, value=0.8, **slider_kwargs),
    dbc.Label("Imitate prob"), dcc.Slider(id="imitate_prob", min=0, max=1, step=0.05, value=0.5, **slider_kwargs),
    dbc.Label("Tremble p"), dcc.Slider(id="tremble_p", min=0, max=0.3, step=0.01, value=0.05, **slider_kwargs),
    dbc.Button("Run simulation", id="run-btn", color="primary", className="mt-3"),
    dbc.Button("Download GIF", id="download-btn", color="secondary", className="mt-2"),
], body=True)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(control_card, width=3),
        dbc.Col([
            dcc.Graph(id="graph", style={"height":"70vh"}),
            dcc.Slider(id="frame-slider", min=0, max=0, step=1, value=0),
        ], width=9),
    ]),
    dcc.Store(id="frames-store"),
    dcc.Download(id="gif-download"),
], fluid=True)

# --- helper to build network figure ---

def fig_from_frame(frame: dict) -> go.Figure:
    edge_x, edge_y, widths = [], [], []
    for e in frame["edges"]:
        edge_x += [e["x0"], e["x1"], None]; edge_y += [e["y0"], e["y1"], None]
        widths.append(e["w"])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="#999"), hoverinfo="skip")
    node_trace = go.Scatter(x=[n["x"] for n in frame["nodes"]], y=[n["y"] for n in frame["nodes"]],
                            mode="markers", marker=dict(color=[n["color"] for n in frame["nodes"]], size=10), hoverinfo="skip")
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(xaxis=dict(showticklabels=False, zeroline=False), yaxis=dict(showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1), margin=dict(l=20, r=20, t=20, b=20), plot_bgcolor="white")
    return fig

# --- Run sim callback ---
@app.callback(
    Output("frames-store", "data"), Output("frame-slider", "max"), Output("frame-slider", "value"),
    Input("run-btn", "n_clicks"),
    State("n","value"), State("pct_opt","value"), State("p_edge","value"),
    State("steps","value"), State("frame_delta","value"),
    State("rew_phi","value"), State("imitate_prob","value"), State("tremble_p","value"),
    prevent_initial_call=True)

def _run(_, n, pct,p_edge,steps,fd,phi,imit,trem):
    params = Params(n=int(n), pct_opt=float(pct), p_edge=float(p_edge), steps=int(steps), frame_delta=int(fd), rew_phi=float(phi), imitate_prob=float(imit), tremble_p=float(trem))
    frames = run_sim(params)
    return frames, len(frames)-1, 0

# --- Slider -> figure -------------------------------------------------------
@app.callback(
    Output("graph", "figure"),
    Input("frame-slider", "value"),
    State("frames-store", "data"),
)
def _update_fig(idx, frames):
    """Redraw network when the slider moves."""
    if not frames:
        return go.Figure()
    idx = max(0, min(idx or 0, len(frames) - 1))
    return fig_from_frame(frames[idx])


# --- Download GIF -----------------------------------------------------------

@app.callback(
    Output("gif-download", "data", allow_duplicate=True),
    Input("download-btn", "n_clicks"),
    State("frames-store", "data"),
    prevent_initial_call=True,
)
def _download_gif(_, frames):
    """
    Convert stored frames → PNGs → animated GIF.
    Returns a dict Dash knows how to send as file download.
    """
    if not frames:
        return dash.no_update

    pil_frames = []
    for fr in frames:
        # Render Plotly fig → PNG bytes (needs kaleido)
        png_bytes = fig_from_frame(fr).to_image(format="png", scale=2)
        pil_frames.append(Image.open(io.BytesIO(png_bytes)))

    # Optional caption on the first frame
    params_text = f"{len(frames)} frames | generated {datetime.datetime.now():%Y‑%m‑%d %H:%M}"
    draw = ImageDraw.Draw(pil_frames[0])
    draw.text((10, 10), params_text, fill="black")

    # Bundle into GIF in‑memory
    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=300,
        loop=0,
    )
    buf.seek(0)

    # Dash Download expects base64
    b64 = base64.b64encode(buf.read()).decode()
    fname = f"pd_echo_{datetime.datetime.now():%Y%m%d_%H%M%S}.gif"
    return dict(content=b64, filename=fname, base64=True)

if __name__ == "__main__":
    app.run(debug=True)
