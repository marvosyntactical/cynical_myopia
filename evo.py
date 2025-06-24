import numpy as np
import matplotlib.pyplot as plt

# === Co-evolving echo‐chamber simulation === #

# PD pay-offs
R, S, T, P = 3, 0, 5, 1

def pd_payoff(a, b):
    if a == 'C' and b == 'C':
        return R, R
    if a == 'C' and b == 'D':
        return S, T
    if a == 'D' and b == 'C':
        return T, S
    return P, P  # D,D

# ----- parameters -----
N            = 120           # agents
init_opt     = 0.5           # initial share of optimists
p_edge       = 0.06          # initial ER connection prob
steps        = 20000
sample_every = 200
rewire_phi   = 0.8           # prob of rewiring a "bad" tie
imitate_prob = 0.5           # chance low-earner copies high-earner
tremble_p    = 0.05          # flips intended action

rng = np.random.default_rng(2)

# ----- initialise population -----
types = np.array(['optimist'] * int(N*init_opt) + ['cynic'] * (N - int(N*init_opt)))

# Random ER graph as adjacency sets
adj = [set() for _ in range(N)]
for i in range(N):
    for j in range(i+1, N):
        if rng.random() < p_edge:
            adj[i].add(j)
            adj[j].add(i)

# payoff trackers
total_payoff = np.zeros(N)
total_rounds = np.zeros(N)

# data record
fractions_opt = []
avg_payoff_opt = []
avg_payoff_cyn = []
assort_vals    = []
timeline       = []

def assortativity():
    same = 0
    total = 0
    for i in range(N):
        for j in adj[i]:
            if j > i:  # count each undirected edge once
                total += 1
                if types[i] == types[j]:
                    same += 1
    return same / total if total else 0

def record(t):
    opt_mask = (types == 'optimist')
    cyn_mask = ~opt_mask
    fractions_opt.append(opt_mask.mean())
    avg_payoff_opt.append(total_payoff[opt_mask].sum() / max(total_rounds[opt_mask].sum(),1))
    avg_payoff_cyn.append(total_payoff[cyn_mask].sum() / max(total_rounds[cyn_mask].sum(),1))
    assort_vals.append(assortativity())
    timeline.append(t)

# ---------- simulation loop ----------
record(0)
actions_lookup = {'cynic':'D', 'optimist':'C'}

for t in range(1, steps+1):
    i = rng.integers(N)
    if not adj[i]:
        continue
    j = rng.choice(list(adj[i]))
    
    # intended actions
    a_i_intended = actions_lookup[types[i]]
    a_j_intended = actions_lookup[types[j]]
    # tremble errors
    a_i = a_i_intended if rng.random() > tremble_p else ('C' if a_i_intended=='D' else 'D')
    a_j = a_j_intended if rng.random() > tremble_p else ('C' if a_j_intended=='D' else 'D')
    
    pay_i, pay_j = pd_payoff(a_i, a_j)
    total_payoff[i] += pay_i
    total_payoff[j] += pay_j
    total_rounds[i] += 1
    total_rounds[j] += 1
    
    # --- strategy imitation ---
    if rng.random() < imitate_prob:
        avg_i = total_payoff[i]/total_rounds[i]
        avg_j = total_payoff[j]/total_rounds[j]
        if avg_i < avg_j:
            types[i] = types[j]
        elif avg_j < avg_i:
            types[j] = types[i]
    
    # --- rewiring (homophily) ---
    def try_rewire(u, v, action_u, action_v):
        """u decides whether to cut tie to v (who defected) and connect to like-minded."""
        if action_u == 'C' and action_v == 'D' and rng.random() < rewire_phi:
            # cut tie
            adj[u].discard(v)
            adj[v].discard(u)
            # pick new partner of same type not already connected
            candidates = np.where((types == types[u]) & (np.arange(N) != u))[0]
            rng.shuffle(candidates)
            for k in candidates:
                if k not in adj[u]:
                    adj[u].add(k)
                    adj[k].add(u)
                    break
    
    try_rewire(i, j, a_i, a_j)
    try_rewire(j, i, a_j, a_i)
    
    if t % sample_every == 0:
        record(t)

# --------------- plots --------------- #

plt.figure(figsize=(8,5))
plt.plot(timeline, fractions_opt, label="Share optimists")
plt.xlabel("Interaction step")
plt.ylabel("Fraction of population")
plt.title("Strategy share over time with co-evolving homophily")
plt.legend()
plt.tight_layout()
plt.savefig("img/evo_strategies.png")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(timeline, avg_payoff_opt, label="Optimist avg payoff")
plt.plot(timeline, avg_payoff_cyn, label="Cynic avg payoff")
plt.xlabel("Interaction step")
plt.ylabel("Running mean payoff")
plt.title("Pay-off trajectories during network-strategy co-evolution")
plt.legend()
plt.tight_layout()
plt.savefig("img/evo_payoffs.png")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(timeline, assort_vals, label="Network assortativity (same-type edge share)")
plt.xlabel("Interaction step")
plt.ylabel("Assortativity r(t)")
plt.title("Echo-chamber formation through rewiring")
plt.savefig("img/evo_assortativity.png")
plt.legend()
plt.tight_layout()
plt.show()
