# Cynicism Game Theory (Iterated Prisoner's Dilemma)

Modelling the dynamics of cynicism on a co-evolving social graph using a
trembling-hand iterated Prisoner's Dilemma with Bayesian beliefs, weighted
trust ties, and within-group rewiring.

Live demo: [https://pd-echo.fly.dev/](https://pd-echo.fly.dev/)

![dash](img/dash_example.png)

## The simulation in [network.py](network.py)

`N` agents on an Erdős–Rényi graph (edge prob `p_edge`, weights `ω = 1`).
Each agent carries a Beta prior over partner defection: belief
`p_i = β_i / (α_i + β_i)`. Optimists start `(α, β) = (5, 1)`; the rest
start `(2, 2)`.

Each step:

1. Pick a random agent `i`, random neighbour `j`.
2. Draw `a_i ~ Bernoulli(p_i)` mapped `1→D, 0→C`; flip with prob `p_flip`.
3. Pay PD payoffs (`T > R > P > S`, defaults `R=3, S=0, T=5, P=1`).
4. **Bayesian update:** observed `C` increments `α`, observed `D` increments `β`.
5. **Trust update on `ω_ij`:** `+0.3` if both `C` (cap 3.0); `−0.2` if both `D`,
   `−0.4` otherwise (floor 0.1). Spring layout uses `ω` as stiffness.
6. **Reputation EMA:** `rep_i ← (1 − α_rep) rep_i + α_rep · π_i`.
7. **Popularity-weighted imitation** (prob `λ`): copy partner's belief with
   probability `∝ deg · avg_payoff`.
8. **Within-group rewiring:** if `i` cooperated and `j` defected, with prob
   `φ` cut `(i, j)` and reconnect `i` to a random same-belief-side node.

## Parameters

| Symbol / slider          | Meaning                                          |
|--------------------------|--------------------------------------------------|
| `N`                      | number of agents                                 |
| `pct_opt`                | initial share with optimist prior                |
| `p_edge`                 | initial ER density                               |
| `steps`, `frame_delta`   | total encounters; snapshot interval              |
| `φ` (`rew_phi`)          | rewiring probability                             |
| `λ` (`imitate_prob`)     | social-learning rate                             |
| `p_flip` (`tremble_p`)   | execution noise                                  |
| `α_rep`                  | reputation EMA coefficient                       |
| `R, S, T, P`             | PD payoffs                                       |

## Plots

### [img/simple.png](img/simple.png) — self-reinforcing beliefs vs Tit-for-Tat
![Simple](img/simple.png)

One agent vs Tit-for-Tat with `p_flip = 0.05`. Cynic's defections beget
retaliation and lock its posterior `P(coop)` near 0; optimist's cooperation
locks it near 1. Same mechanics, opposite stable beliefs.

### [img/assort.png](img/assort.png) — within-group sweep
![Assort](img/assort.png)

50 always-`D` vs 50 always-`C`, with within-group preference `r ∈ [0, 1]`.
At `r = 0` cynics exploit and dominate; as `r → 1` cynics get stuck at `P`
and cooperators converge to `R`. The curves cross — perfect echo chambers
favour cooperators.

### [img/evo_strategies.png](img/evo_strategies.png) — strategy share
![evo1](img/evo_strategies.png)

Co-evolving run: fraction of optimists over 20 000 steps. Outcome depends
on whether rewiring segregates cooperators before imitation copies them
into defection.

### [img/evo_payoffs.png](img/evo_payoffs.png) — payoff trajectories
![evo3](img/evo_payoffs.png)

Same run, running mean payoff per type. Cynics start near `T` (exploiting),
end near `P` (isolated); cooperators climb to `R` once their clique forms.

### [img/evo_assortativity.png](img/evo_assortativity.png) — echo-chamber formation
![evo2](img/evo_assortativity.png)

Same-type edge fraction `r(t)`. Climb rate is how fast `φ`-rewiring
segregates the network.

### [img/optimal_action.png](img/optimal_action.png) — HJB best response
![optimal](img/optimal_action.png)

Value iteration over `(belief p, discount ρ)` with belief drift
`p ← p + dt · k · (a − p)`. Heatmap of optimal action: red = cooperate,
blue = defect. The boundary slopes up — patient agents cooperate at lower
beliefs. Cynicism is myopia.

### [img/coop_benefit.png](img/coop_benefit.png) — value gap and drift
![coop](img/coop_benefit.png)

Same HJB, but plotting continuous `V_C − V_D` (seismic) with the drift
field `dp/dt = k(a* − p)` overlaid. Arrows are bistable around the
boundary — small belief noise decides which basin the agent falls into.
