import numpy as np
import matplotlib.pyplot as plt

# Prisoner's Dilemma pay-offs (row player's view)
R, S, T, P = 3, 0, 5, 1

def play_pd(a, b):
    """Return payoffs (to a, to b) given actions 'C' or 'D'."""
    if a == 'C' and b == 'C':
        return R, R
    if a == 'C' and b == 'D':
        return S, T
    if a == 'D' and b == 'C':
        return T, S
    return P, P  # D,D

def simulate_homophily(r, n_agents=100, steps=5000, seed=None):
    """
    Mix of 50 cynics (always D) & 50 optimists (always C).
    Parameter r ∈ [0,1] = homophily strength.
    """
    rng = np.random.default_rng(seed)
    types = np.array(['cynic'] * (n_agents//2) + ['optimist'] * (n_agents//2))
    actions_lookup = {'cynic': 'D', 'optimist': 'C'}
    
    payoff_sum = {'cynic': 0.0, 'optimist': 0.0}
    counts = {'cynic': 0, 'optimist': 0}
    
    for _ in range(steps):
        i = rng.integers(n_agents)
        i_type = types[i]
        
        # choose partner with homophily r
        if rng.random() < r:
            candidates = np.where(types == i_type)[0]
        else:
            candidates = np.arange(n_agents)
        j = rng.choice(candidates)
        while j == i:  # avoid self‐match
            j = rng.choice(candidates)
        
        j_type = types[j]
        
        a_i = actions_lookup[i_type]
        a_j = actions_lookup[j_type]
        p_i, p_j = play_pd(a_i, a_j)
        
        payoff_sum[i_type] += p_i
        payoff_sum[j_type] += p_j
        counts[i_type] += 1
        counts[j_type] += 1
    
    avg_payoff = {t: payoff_sum[t] / counts[t] for t in payoff_sum}
    return avg_payoff['cynic'], avg_payoff['optimist']

# sweep r
rs = np.linspace(0, 1, 21)
avg_cynic = []
avg_optim = []
for r in rs:
    c, o = simulate_homophily(r, seed=42)
    avg_cynic.append(c)
    avg_optim.append(o)

# Plot
plt.figure(figsize=(8,5))
plt.plot(rs, avg_cynic, label="Cynic avg payoff")
plt.plot(rs, avg_optim, label="Optimist avg payoff")
plt.xlabel("Homophily r (0=random mix, 1=perfect echo chamber)")
plt.ylabel("Average PD payoff per encounter")
plt.title("Echo-chamber homophily and strategic efficacy")
plt.legend()
plt.tight_layout()
plt.savefig("img/assort.png")
plt.show()
