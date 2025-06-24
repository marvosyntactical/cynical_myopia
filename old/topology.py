"""
Iterated PD Echo‑Chamber Simulator — Dash webapp
================================================
Fire up with
    $ pip install dash==2.17.0 dash-bootstrap-components plotly networkx numpy
    $ python iterated_pd_network_dash.py

Blue = optimists (cooperate), orange = cynics (defect).  Use the control panel to set
population size, edge density, noise, etc., then hit **Run simulation**.  A time‑slider
lets you scrub through the generated frames.
"""
from __future__ import annotations
import json, time, typing as _t, itertools as it
from dataclasses import dataclass

import dash
from dash import dcc, html, Output, Input, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import networkx as nx
import numpy as np

# ░░░░░░░░░░░░░░░░░░  Simulation core  ░░░░░░░░░░░░░░░░░░#
@dataclass
class Params:
    n: int = 50
    pct_opt: float = 0.4
    p_edge: float = 0.05
    steps: int = 1500
    frame_delta: int = 100
    rew_phi: float = 0.9
    imitate_prob: float = 0.3
    tremble_p: float = 0.05
    R: int = 3
    S: int = 0
    T: int = 5
    P: int = 1

Action = str  # "C" or "D"


def pd_payoff(a: Action, b: Action, R, S, T, P):
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
        ["optimist"] * int(params.n * params.pct_opt) + ["cynic"] * (params.n - int(params.n * params.pct_opt))
    )
    actions_lookup = {"cynic": "D", "optimist": "C"}

    G = nx.erdos_renyi_graph(params.n, params.p_edge, seed=42)
    # ensure connectivity
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for comp in comps[1:]:
            G.add_edge(next(iter(comp)), next(iter(comps[0])))

    pos = nx.spring_layout(G, seed=1)  # initial layout; reused each frame for smoother viz
    total_payoff = np.zeros(params.n)
    total_rounds = np.zeros(params.n)

    frames: list[dict] = []

    def snapshot(step: int):
        nodes = [
            {
                "x": float(pos[v][0]),
                "y": float(pos[v][1]),
                "color": "#1f77b4" if types[v] == "optimist" else "#ff7f0e",
            }
            for v in G.nodes()
        ]
        edges = [
            {
                "x0": float(pos[u][0]),
                "y0": float(pos[u][1]),
                "x1": float(pos[v][0]),
                "y1": float(pos[v][1]),
            }
            for u, v in G.edges()
        ]
        frames.append({"step": step, "nodes": nodes, "edges": edges})

    snapshot(0)

    def try_rewire(u, v, au, av):
        if au == "C" and av == "D" and rng.random() < params.rew_phi and G.has_edge(u, v):
            G.remove_edge(u, v)
            candidates = [k for k in range(params.n) if types[k] == types[u] and k != u and not G.has_edge(u, k)]
            if candidates:
                G.add_edge(u, rng.choice(candidates))

    for t in range(1, params.steps + 1):
        i = rng.integers(params.n)
        nbrs = list(G[i])
        if not nbrs:
            continue
        j = rng.choice(nbrs)

        # actions with tremble
        a_i = actions_lookup[types[i]]
        a_j = actions_lookup[types[j]]
        if rng.random() < params.tremble_p:
            a_i = "D" if a_i == "C" else "C"
        if rng.random() < params.tremble_p:
            a_j = "D" if a_j == "C" else "C"

        p_i, p_j = pd_payoff(a_i, a_j, params.R, params.S, params.T, params.P)
        total_payoff[i] += p_i
        total_payoff[j] += p_j
        total_rounds[i] += 1
        total_rounds[j] += 1

        # imitation
        if rng.random() < params.imitate_prob:
            avg_i = total_payoff[i] / total_rounds[i]
            avg_j = total_payoff[j] / total_rounds[j]
            if avg_i < avg_j:
                types[i] = types[j]
            elif avg_j < avg_i:
                types[j] = types[i]

        try_rewire(i, j, a_i, a_j)
        try_rewire(j, i, a_j, a_i)

        if t % params.frame_delta == 0:
            # recompute layout occasionally so graph doesn’t drift off‑screen
            pos = nx.spring_layout(G, pos=pos, iterations=10, seed=1)
            snapshot(t)

    return frames

# ░░░░░░░░░░░░░░░░░░  Dash app  ░░░░░░░░░░░░░░░░░░#
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

control_card = dbc.Card(
    [
        html.H5("Controls", className="card-title"),
        dbc.Label("Agents (N)"),
        dcc.Input(id="n", type="number", value=50, min=10, max=200, step=10, className="mb-2", style={"width": "100%"}),
        dbc.Label("% Optimists"),
        dcc.Slider(id="pct_opt", min=0, max=1, step=0.05, value=0.5, tooltip={"placement": "bottom"}),
        dbc.Label("Edge prob"),
        dcc.Slider(id="p_edge", min=0.02, max=0.3, step=0.01, value=0.1, tooltip={"placement": "bottom"}),
        dbc.Label("Steps"),
        dcc.Input(id="steps", type="number", value=1500, min=100, max=10000, step=100, className="mb-2", style={"width": "100%"}),
        dbc.Label("Frame Δ"),
        dcc.Input(id="frame_delta", type="number", value=20, min=5, max=500, step=10, className="mb-2", style={"width": "100%"}),
        dbc.Label("Rewire φ"),
        dcc.Slider(id="rew_phi", min=0, max=1, step=0.05, value=0.8, tooltip={"placement": "bottom"}),
        dbc.Label("Imitate prob"),
        dcc.Slider(id="imitate_prob", min=0, max=1, step=0.05, value=0.5, tooltip={"placement": "bottom"}),
        dbc.Label("Tremble p"),
        dcc.Slider(id="tremble_p", min=0, max=0.3, step=0.01, value=0.05, tooltip={"placement": "bottom"}),
        dbc.Button("Run simulation", id="run-btn", color="primary", className="mt-3"),
    ],
    body=True,
    style={"height": "100%"},
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(control_card, width=3),
                dbc.Col(
                    [
                        dcc.Graph(id="graph", style={"height": "70vh"}),
                        dcc.Slider(id="frame-slider", min=0, max=0, step=1, value=0),
                    ],
                    width=9,
                ),
            ],
            align="start",
        ),
        # hidden storage for frames
        dcc.Store(id="frames-store"),
    ],
    fluid=True,
)

# -- Callback to run simulation and populate frames --
@app.callback(
    Output("frames-store", "data"),
    Output("frame-slider", "max"),
    Output("frame-slider", "value"),
    Input("run-btn", "n_clicks"),
    State("n", "value"),
    State("pct_opt", "value"),
    State("p_edge", "value"),
    State("steps", "value"),
    State("frame_delta", "value"),
    State("rew_phi", "value"),
    State("imitate_prob", "value"),
    State("tremble_p", "value"),
    prevent_initial_call=True,
)

def run_and_store(_, n, pct_opt, p_edge, steps, frame_delta, rew_phi, imitate_prob, tremble_p):
    params = Params(
        n=int(n),
        pct_opt=float(pct_opt),
        p_edge=float(p_edge),
        steps=int(steps),
        frame_delta=int(frame_delta),
        rew_phi=float(rew_phi),
        imitate_prob=float(imitate_prob),
        tremble_p=float(tremble_p),
    )
    frames = run_sim(params)
    # JSON‑serialisable
    json_frames = json.loads(json.dumps(frames))
    return json_frames, len(frames) - 1, 0

# -- Callback to draw selected frame --
@app.callback(
    Output("graph", "figure"),
    Input("frame-slider", "value"),
    State("frames-store", "data"),
)

def update_figure(frame_idx, data):
    if not data:
        # empty placeholder
        fig = go.Figure()
        fig.update_layout( xaxis_visible=False, yaxis_visible=False, plot_bgcolor="white")
        return fig

    frame = data[frame_idx]
    nodes = frame["nodes"]
    edges = frame["edges"]

    # build edge scatter
    edge_x, edge_y = [], []
    for e in edges:
        edge_x += [e["x0"], e["x1"], None]
        edge_y += [e["y0"], e["y1"], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#888"), hoverinfo="skip")

    # node scatter
    node_x = [n["x"] for n in nodes]
    node_y = [n["y"] for n in nodes]
    colors = [n["color"] for n in nodes]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(color=colors, size=10, line=dict(width=0)),
        hoverinfo="skip",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        xaxis=dict(showticklabels=False, zeroline=False),
        yaxis=dict(showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
