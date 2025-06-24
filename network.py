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
from PIL import Image
import imageio.v2 as imageio   # pillow backend
import kaleido                 # make sure kaleido is installed
import base64, io

import base64, io, datetime, textwrap
from dataclasses import dataclass
from typing import List, Dict
import os

import dash, dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import numpy as np


DEBUG = 0 # 0 for fly, 1 for local

# ░░ Simulation core ░░
@dataclass
class Params:
    """
    Parameter bundle for the iterated‑PD network.

    n            – number of agents (nodes).
    pct_opt      – share of agents initialised as ‘optimist’ (cooperate).
    p_edge       – Erdős–Rényi edge probability at t=0.
    steps        – total pair‑encounters to simulate.
    frame_delta  – store one animation frame every Δ encounters.
    rew_phi      – rewiring chance when C meets D (0 = never, 1 = always).
    imitate_prob – probability lower‑earner copies strategy of richer partner.
    tremble_p    – probability an intended move flips (execution noise).
    R,S,T,P      – classical Prisoner’s‑Dilemma pay‑offs.
    """
    n: int = 100
    pct_opt: float = 0.5
    p_edge: float = 0.1
    steps: int = 1500
    frame_delta: int = 100
    rew_phi: float = 0.8
    imitate_prob: float = 0.05
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
        nodes = [
            {
                "x": float(pos[v][0]),
                "y": float(pos[v][1]),
                "v": 1.0 if types[v] == "optimist" else 0.0,   # numeric flag
            }
            for v in G
        ]
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
    dcc.Input(id="n", type="number", value=100, min=10, max=200, step=10, style={"width": "100%"}),
    dbc.Label("% Init Optimists"), dcc.Slider(id="pct", min=0, max=1, step=0.05, value=0.5, **_slider_kw),
    dbc.Label("Edge prob"), dcc.Slider(id="p_edge", min=0.05, max=0.3, step=0.05, value=0.1, **_slider_kw),
    dbc.Label("Steps"), dcc.Input(id="steps", type="number", value=1500, min=100, max=10000, step=100, style={"width": "100%"}),
    dbc.Label("Frame Delta"), dcc.Input(id="frame_delta", type="number", value=10, min=10, max=500, step=10, style={"width": "100%"}),
    dbc.Label("Rewire φ"), dcc.Slider(id="phi", min=0, max=1, step=0.1, value=0.8, **_slider_kw),
    dbc.Label("Imitate prob"), dcc.Slider(id="imit", min=0, max=1, step=0.05, value=0.05, **_slider_kw),
    dbc.Label("Tremble p"), dcc.Slider(id="trem", min=0, max=0.3, step=0.05, value=0.05, **_slider_kw),
    dbc.Button("Run simulation", id="run", color="primary", className="mt-2"),
    dbc.Button("Save GIF", id="btn-gif", color="secondary", className="mt-2"),  # ← add me
  

], body=True)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(controls, width=3),
                
                dbc.Col(
    [
        # ── upper half: network + slider ──────────────────────────
        html.Div(
            [
                dcc.Graph(id="graph", style={"height": "40vh"}),
                dcc.Slider(id="slider", min=0, max=0, step=1, value=0),
            ]
        ),

        # ── lower half: markdown doc pane ─────────────────────────
        html.Div(
            id="doc-pane",
            style={"height": "45vh", "overflowY": "auto", "padding": "0.5rem"},
            children=dcc.Markdown(
                r"""
**Click 'Run Simulation' and give it a couple o' seconds :)**
### What you’re seeing

**Nodes** represent agents playing an iterated, noisy Prisoner’s Dilemma.  
Node colour is a gradient; red = pure cynic (always *D*), green = pure optimist
(always *C*).

**Edges** have a *trust weight* ω (thicker = stronger).  
ω grows when both endpoints cooperate, shrinks on betrayal, and drives the
spring layout: strong ties pull nodes closer, brittle ones stretch out.

### Parameter cheat‑sheet
| Slider | Meaning | Typical effect |
|--------|---------|----------------|
| Agents (N) | population size | bigger slows dynamics |
| % Optimists | initial green share | raises co‑op seed |
| Edge prob | initial density | dense ⇢ early exploitation |
| Steps | run length | just more frames |
| Frame Δ | store every Δ steps | x-axis granularity |
| Rewire φ | chance C cuts link to D | high φ lets greens self‑segregate |
| Imitate prob | social learning rate | high spreads whichever wins |
| Tremble p | move‑flip noise | noise hurts defectors |

### Reading the plot
* A tight green blob = self‑reinforcing trust island.
* Sparse red web = cynics unable to milk anyone.
* If the graph freezes orange, the temptation T still beats the long‑term reward R.

### Model primer

#### 1. Players  
* \(N\) agents on a graph \(G_t=(V,E_t)\)  
* Binary strategy label  
  \[
    s_i(t)\in\{\text{optimist }(C),\;\text{cynic }(D)\}
  \]

#### 2. Pay‑off matrix  

\[
\begin{array}{c|cc}
 & C & D\\ \hline
C & (R,R) & (S,T) \\\\
D & (T,S) & (P,P)
\end{array}
\qquad
T>R>P>S
\]

#### 3. Single interaction step  

1. Pick random edge \((i,j)\).  
2. Each plays \(a_i=s_i\) with tremble error \(p_\textrm{flip}\).  
3. Receive pay‑offs \(\pi_i,\pi_j\) from the matrix.  
4. **Social learning** with prob \(\lambda\) the lower earner copies the higher earner’s strategy.  
5. **Re‑wiring** if \(i\!:\!C,\;j\!:\!D\) then  
   \[
     \Pr\bigl[(i,j)\text{ cut}\bigr] = \phi
   \]
   and \(i\) reconnects to a like‑minded node.  
6. **Trust update** on edge weight \(\omega_{ij}\):

\[
\omega_{ij}(t{+}1)=
\begin{cases}
\min(1,\;\omega_{ij}+\eta) & a_i=a_j=C\\\\
\max(\omega_{\min},\;\omega_{ij}-\eta) & a_i\neq a_j
\end{cases}
\]

The spring layout uses \(\omega_{ij}\) as its spring constant  
⇒ thick green ties pull nodes together; thin red ties stretch.

#### 4. Key parameters  
| Symbol/slider | Meaning |
|---------------|---------|
| \(N\) | population size |
| pct\_opt | initial share of optimists |
| \(p_\text{edge}\) | initial ER density |
| \(\phi\) | rewiring probability |
| \(\lambda\) | imitate probability |
| \(p_\text{flip}\) | tremble noise |
| \(R,S,T,P\) | PD pay‑offs |

#### 5. Dynamics in one line  

\[
\dot s_i = \lambda\;\bigl[\pi_j - \pi_i\bigr]_+\;
      \;\bigl(s_j - s_i\bigr)
\]

i.e. a discrete replicator update modulated by network rewiring.


### Exporting as GIF
Doesn't work yet. Check back soon!
"""
            ),
        ),
    ],
    width=9,
),

            ]
        ),
        dcc.Store(id="frames"),
        dcc.Download(id="dl-gif"),          # ← must match callback Output
    ],
    fluid=True,
)


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
    vals = [n["v"] for n in frame["nodes"]]       # 0 → red, 1 → green

    node_trace = go.Scatter(
        x=[n["x"] for n in frame["nodes"]],
        y=[n["y"] for n in frame["nodes"]],
        mode="markers",
        marker=dict(
            color=vals,
            colorscale=[[0.0, "red"], [1.0, "green"]],
            cmin=0, cmax=1,
            size=11,
            colorbar=dict(title="optimism"),
        ),
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

# I dont GIF a frick

@app.callback(
    Output("dl-gif", "data", allow_duplicate=True),   # <─ ID matches the downloader
    Input("btn-gif", "n_clicks"),                     # <─ ID matches the button
    State("frames", "data"),
    prevent_initial_call=True,
)
def export_gif(n_clicks, frames):
    if not n_clicks or not frames:
        raise dash.exceptions.PreventUpdate

    # Render each stored frame → PNG bytes via kaleido
    pngs = [frame_to_figure(f).to_image(format="png", scale=2) for f in frames]

    # Assemble GIF with Pillow / imageio
    pil_frames = [Image.open(io.BytesIO(b)) for b in pngs]
    buf = io.BytesIO()
    imageio.mimsave(buf, pil_frames, format="GIF", duration=350)  # 350 ms between frames
    buf.seek(0)

    # Hand it to Dash Download
    b64 = base64.b64encode(buf.read()).decode()
    fname = f"pd_echo_{datetime.datetime.now():%Y%m%d_%H%M%S}.gif"
    return {"content": b64, "filename": fname, "base64": True}

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if DEBUG:
        # local
        app.run(debug=True)
    else:
        # fly
        port = int(os.environ.get("PORT", 8050))
        app.run(host="0.0.0.0", port=port, debug=False)
