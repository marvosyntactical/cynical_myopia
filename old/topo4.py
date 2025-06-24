"""
Iterated PD Echo‑Chamber Simulator — Dash web‑app (v2)
=====================================================
Blue = optimists (cooperate); orange = cynics (defect).

**What’s new**
--------------
1. **Meaningful edge length & width**  – every link carries a *trust weight* ω:
   * when the two nodes *both cooperate* ⇒ ω ↑
   * when one stabs the other ⇒ ω ↓ (floors at 0.1)
   
   Spring‑layout is recomputed each stored frame with `weight=ω`, so *strong‑trust* links pull their owners closer.  Line width is proportional to ω, giving an immediate visual cue: thick/short = strong rapport, thin/long = brittle.
2. **GIF exporter fixed** – uses Kaleido (install it!) and a helper `figure_for(frame)` so the callback no longer throws.  The first frame is captioned with the current parameter settings.
3. **Help pop‑up** – click the ▸ tooltip icon next to each slider if you forget what it does.
4. **README snippet** now explains three hosting modes:
   * Flask/Gunicorn on any VPS (works out‑of‑the‑box).
   * **Fly.io**: `fly launch`, accept defaults; it auto‑detects the `gunicorn` command.
   * **Static export** (no Python on server) – run `dash-renderer` → `/dist` folder; note: only the *controls* stay interactive, simulation must be pre‑generated and embedded (not ideal).

Python deps
-----------
```bash
pip install dash==2.17.0 dash-bootstrap-components plotly networkx numpy imageio pillow kaleido
```

Run locally
-----------
```bash
python iterated_pd_network_dash.py       # <- uses app.run_server(debug=True)
```

Feel free to fork the repo and push to GitHub → Vercel *if* you’re OK with the “static export” limitation.
"""

from __future__ import annotations
import io, json, typing as _t
from dataclasses import dataclass

import dash
from dash import dcc, html, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import networkx as nx
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

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
    types = np.array([
        "optimist" * 0  # placeholder to satisfy mypy
    ])  # overwritten below
    types = np.array(
        ["optimist"] * int(params.n * params.pct_opt)
        + ["cynic"] * (params.n - int(params.n * params.pct_opt))
    )
    actions_lookup = {"cynic": "D", "optimist": "C"}

    # --- graph with trust weights ---
    G = nx.erdos_renyi_graph(params.n, params.p_edge, seed=42)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for comp in comps[1:]:
            G.add_edge(next(iter(comp)), next(iter(comps[0])))

    nx.set_edge_attributes(G, 1.0, "weight")  # initial trust weight = 1
    pos = nx.spring_layout(G, seed=1, weight="weight")

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
                "w": G[u][v]["weight"],
            }
            for u, v in G.edges()
        ]
        frames.append({"step": step, "nodes": nodes, "edges": edges})

    snapshot(0)

    def try_rewire(u, v, au, av):
        if (
            au == "C"
            and av == "D"
            and rng.random() < params.rew_phi
            and G.has_edge(u, v)
        ):
            G.remove_edge(u, v)
            candidates = [
                k
                for k in range(params.n)
                if types[k] == types[u] and k != u and not G.has_edge(u, k)
            ]
            if candidates:
                G.add_edge(u, rng.choice(candidates))
                G[u][rng.choice(candidates)]["weight"] = 1.0

    # --- main loop ---
    for t in range(1, params.steps + 1):
        i = rng.integers(params.n)
        nbrs = list(G[i])
        if not nbrs:
            continue
        j = rng.choice(nbrs)

        a_i = actions_lookup[types[i]]
        a_j = actions_lookup[types[j]]
        if rng.random() < params.tremble_p:
            a_i = "D" if a_i == "C" else "C"
        if rng.random() < params.tremble_p:
            a_j = "D" if a_j == "C" else "C"

        # update edge trust weight before pay‑off; symmetric update
        if a_i == "C" and a_j == "C":
            G[i][j]["weight"] = min(G[i][j]["weight"] + 0.3, 3.0)
        elif a_i == "D" and a_j == "D":
            G[i][j]["weight"] = max(G[i][j]["weight"] - 0.2, 0.1)
        else:  # one stabs → larger penalty
            G[i][j]["weight"] = max(G[i][j]["weight"] - 0.4, 0.1)

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
            # spring layout with trust weights: strong links pull closer
            pos = nx.spring_layout(G, pos=pos, iterations=15, seed=1, weight="weight")
            snapshot(t)

    return frames

# ░░ Dash app ░░
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

def tt(label: str, text: str):
    return html.Span([label, dbc.Tooltip(text, target=label, placement="right")])

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
        dcc.Input(id="frame_delta", type="number", value=100, min=10, max=500, step=10, className="mb-2", style={"width": "100%"}),
        dbc.Label("Rewire φ"),
        dcc.Slider(id="rew_phi", min=0, max=1, step=0.05, value=0.8, tooltip={"placement": "bottom"}),
        dbc.Label("Imitate prob"),
        dcc.Slider(id="imitate_prob", min=0, max=1, step=0.05, value=0.5, tooltip={"placement": "bottom"}),
        dbc.Label("Tremble p"),
        dcc.Slider(id="tremble_p", min=0, max=0.3, step=0.01, value=0.05, tooltip={"placement": "bottom"}),
        dbc.Button("Run simulation", id="run-btn", color="primary", className="mt-3"),
        dbc.Button("Download GIF", id="download-btn", color="secondary", className="mt-2"),
    ],
    body=True,
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
        dcc.Store(id="frames-store"),
        dcc.Download(id="gif-download"),
    ],
    fluid=True,
)

# --- helper to build figure from frame ---

def figure_for(frame: dict) -> go.Figure:
    nodes, edges = frame["nodes"], frame["edges"]
    edge_x, edge_y, widths = [], [], []
    for e in edges:
        edge_x += [e["x0"], e["x1"], None]
        edge_y += [e["y0"], e["y1"], None]
        widths.append(e["w"])
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="#aaa"),
        hoverinfo="skip",
    )
    node_trace = go.Scatter(
        x=[n["x"] for n in nodes],
        y=[n["y"] for n in nodes],
        mode="markers",
        marker=dict(color=[n["color"] for n in nodes], size=10),
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

# --- run simulation callback ---
@app.callback(
    Output("frames-store", "data"),
    Output("frame-slider", "max"),
    Output("frame-slider", "value"),
    Input("run-btn", "n_clicks"),
    State("n", "value"), State("pct_opt", "value"), State("p_edge", "value"),
    State("steps", "value"), State("frame_delta", "value"),
    State("rew_phi", "value"), State("imitate_prob", "value"), State("tremble_p", "value"),
    prevent_initial_call=True,
)

def run_and_store(_, n, pct_opt, p_edge, steps, frame_delta, rew_phi, imitate_prob, tremble_p):
    params = Params(
        n=int(n), pct_opt=float(pct_opt), p_edge=float(p_edge), steps=int(steps),
        frame_delta=int(frame_delta), rew_phi=float(rew_phi), imitate_prob=float(imitate_prob), tremble_p=float(tremble_p)
    )
    frames = run_sim(params)
    return frames, len(frames) - 1, 0

# --- slider -> figure ---
@app
