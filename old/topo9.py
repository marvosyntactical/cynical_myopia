"""
Iterated PD Echo‑Chamber Simulator — Dash web‑app (clean, no‑GIF build)
====================================================================
Blue = optimists (cooperate); orange = cynics (defect).
Edge length & stroke ≈ trust weight ω (strong co‑op tie ⇒ short & thick).

Install & run
-------------
```bash
pip install dash==2.17.0 dash-bootstrap-components plotly networkx numpy
python iterated_pd_network_dash.py   # open http://127.0.0.1:8050
```
"""
from __future__ import annotations
import base64, io, datetime, textwrap
from dataclasses import dataclass
from typing import List, Dict

import dash, dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import numpy as np

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

_action = {"cynic": "D", "optimist": "C"}


def _pd(a: str, b: str, p: Params):
    if a == "C" and b == "C":
        return p.R, p.R
    if a == "C" and b == "D":
        return p.S, p.T
    if a == "D" and b == "C":
        return p.T, p.S
    return p.P, p.P


def run_sim(prm: Params) -> List[Dict]:
    rng = np.random.default_rng(0)
    types = np.array(["optimist"] * int(prm.n * prm.pct_opt) + ["cynic"] * (prm.n - int(prm.n * prm.pct_opt)))

    G = nx.erdos_renyi_graph(prm.n, prm.p_edge, seed=42)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for comp in comps[1:]:
            G.add_edge(next(iter(comp)), next(iter(comps[0])))

    nx.set_edge_attributes(G, 1.0, "weight")
    pos = nx.spring_layout(G, seed=2, weight="weight")

    tot_pay = np.zeros(prm.n)
    tot_cnt = np.zeros(prm.n)
    frames: List[Dict] = []

    def snapshot(step: int):
        nodes = [{"x": float(pos[v][0]), "y": float(pos[v][1]), "c": "#1f77b4" if types[v] == "optimist" else "#ff7f0e"} for v in G.nodes()]
        edges = [{"x0": float(pos[u][0]), "y0": float(pos[u][1]), "x1": float(pos[v][0]), "y1": float(pos[v][1]), "w": G[u][v]["weight"]} for u, v in G.edges()]
        frames.append({"step": step, "nodes": nodes, "edges": edges})

    snapshot(0)

    def maybe_rewire(u, v, au, av):
        if au == "C" and av == "D" and rng.random() < prm.rew_phi and G.has_edge(u, v):
            G.remove_edge(u, v)
            cands = [k for k in range(prm.n) if types[k] == types[u] and k != u and not G.has_edge(u, k)]
            if cands:
                G.add_edge(u, rng.choice(cands), weight=1.0)

    for t in range(1, prm.steps + 1):
        i = rng.integers(prm.n)
        nbrs = list(G[i])
        if not nbrs:
            continue
        j = rng.choice(nbrs)
        a_i, a_j = _action[types[i]], _action[types[j]]
        if rng.random() < prm.tremble_p:
            a_i = "D" if a_i == "C" else "C"
        if rng.random() < prm.tremble_p:
            a_j = "D" if a_j == "C" else "C"

        # trust weight update
        if a_i == a_j == "C":
            G[i][j]["weight"] = min(G[i][j]["weight"] + 0.3, 3.0)
        elif a_i == a_j == "D":
            G[i][j]["weight"] = max(G[i][j]["weight"] - 0.2, 0.1)
        else:
            G[i][j]["weight"] = max(G[i][j]["weight"] - 0.4, 0.1)

        p_i, p_j = _pd(a_i, a_j, prm)
        tot_pay[i] += p_i
        tot_pay[j] += p_j
        tot_cnt[i] += 1
        tot_cnt[j] += 1

        # imitation
        if rng.random() < prm.imitate_prob and tot_cnt[i] and tot_cnt[j]:
            avg_i, avg_j = tot_pay[i] / tot_cnt[i], tot_pay[j] / tot_cnt[j]
            if avg_i < avg_j:
                types[i] = types[j]
            elif avg_j < avg_i:
                types[j] = types[i]

        maybe_rewire(i, j, a_i, a_j)
        maybe_rewire(j, i, a_j, a_i)

        if t % prm.frame_delta == 0:
            pos = nx.spring_layout(G, pos=pos, iterations=15, seed=2, weight="weight")
            snapshot(t)

    return frames

# ░░ Dash app ░░
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

_slider_kw = dict(tooltip={"placement": "bottom", "always_visible": False})
controls = dbc.Card([
    html.H5("Controls"),
    dbc.Label("Agents (N)"),
    dcc.Input(id="n", type="number", value=50, min=10, max=200, step=10, style={"width": "100%"}),
    dbc.Label("% Optimists"), dcc.Slider(id="pct", min=0, max=1, step=0.05, value=0.5, **_slider_kw),
    dbc.Label("Edge prob"), dcc.Slider(id="p_edge", min=0.02, max=0.3, step=0.01, value=0.1, **_slider_kw),
    dbc.Label("Steps"), dcc.Input(id="steps", type="number", value=1500, min=100, max=10000, step=100, style={"width": "100%"}),
    dbc.Label("Frame Δ"), dcc.Input(id="frame_delta", type="number", value=100, min=10, max=500, step=10, style={"width": "100%"}),
    dbc.Label("Rewire φ"), dcc.Slider(id="phi", min=0, max=1, step=0.05, value=0.8, **_slider_kw),
    dbc.Label("Imitate prob"), dcc.Slider(id="imit", min=0, max=1, step=0.05, value=0.5, **_slider_kw),
    dbc.Label("Tremble p"), dcc.Slider(id="trem", min=0, max=0.3, step=0.01, value=0.05, **_slider_kw),
    dbc.Button("Run simulation", id="run", color="primary", className="mt-2"),
], body=True)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(controls, width=3),
        dbc.Col([
            dcc.Graph(id="graph", style={"height": "70vh"}),
            dcc.Slider(id="slider", min=0, max=0, step=1, value=0),
        ], width=9),
    ]),
    dcc.Store(id="frames"),
], fluid=True)

# --- helper to build figure

def frame_to_figure(frame: Dict) -> go.Figure:
    if not frame:
        return go.Figure()

    edge_x, edge_y = [], []
    for e in frame["edges"]:
        edge_x.extend([e["x0"], e["x1"], None])
        edge_y.extend([e["y0"], e["y1"], None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        # Plotly Scatter can't vary line.width per‑segment, so use a fixed thin stroke
        line=dict(width=1.2, color="#9e9e9e"),
        hoverinfo="skip",
    )
    node_trace = go.Scatter(
        x=[n["x"] for n in frame["nodes"]],
        y=[n["y"] for n in frame["nodes"]],
        mode="markers",
        marker=dict(color=[n["c"] for n in frame["nodes"]], size=11, line=dict(width=0)),
        hoverinfo="skip",
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
    )
    return fig

# --- run sim callback
@app.callback(
    Output("frames", "data"), Output("slider", "max"), Output("slider", "value"),
    Input("run", "n_clicks"),
    State("n", "value"), State("pct", "value"), State("p_edge", "value"),
    State("steps", "value"), State("frame_delta", "value"),
    State("phi", "value"), State("imit", "value"), State("trem", "value"),
    prevent_initial_call=True,
)
def run_simulation(n_clicks, n, pct, p_edge, steps, frame_delta, phi, imit, trem):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    prm = Params(
        n=int(n),
        pct_opt=float(pct),
        p_edge=float(p_edge),
        steps=int(steps),
        frame_delta=int(frame_delta),
        rew_phi=float(phi),
        imitate_prob=float(imit),
        tremble_p=float(trem),
    )

    frames = run_sim(prm)
    # store frames, set slider range to number of frames − 1, reset to 0
    return frames, len(frames) - 1, 0


# --- slider -> figure -------------------------------------------------------
@app.callback(
    Output("graph", "figure"),
    Input("slider", "value"),
    State("frames", "data"),
)
def update_graph(idx, frames):
    if not frames:
        raise dash.exceptions.PreventUpdate
    idx = int(idx or 0)
    idx = max(0, min(idx, len(frames) - 1))
    return frame_to_figure(frames[idx])


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
