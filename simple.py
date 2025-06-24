import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
rounds = 200
tremble_p = 0.05  # probability an intended action flips

# Helper: simulate an agent interacting with Tit-for-Tat partner
def simulate(agent_type):
    # Beta prior on P(cooperate)
    if agent_type == "cynic":
        a, b = 1, 5   # expects low cooperation
        intended_action = "D"
    elif agent_type == "optimist":
        a, b = 5, 1   # expects high cooperation
        intended_action = "C"
    else:
        raise ValueError
    
    partner_last = "C"  # Tit-for-Tat starts nice
    posterior_means = []
    partner_actions = []
    
    for _ in range(rounds):
        # Agent action (with tremble)
        if np.random.rand() < tremble_p:
            agent_action = "C" if intended_action == "D" else "D"
        else:
            agent_action = intended_action
        
        # Partner Tit-for-Tat response (with tremble)
        partner_intended = "C" if partner_last == "C" else "D"
        if np.random.rand() < tremble_p:
            partner_action = "C" if partner_intended == "D" else "D"
        else:
            partner_action = partner_intended
        
        # Observe partner action and update Beta(a,b) where a = coop count, b = defect count
        if partner_action == "C":
            a += 1
        else:
            b += 1
        
        posterior_means.append(a / (a + b))
        partner_actions.append(partner_action)
        
        # For next round, partner remembers our actual action
        partner_last = agent_action
    
    return np.array(posterior_means), np.array(partner_actions) == "C"

# Run simulations
np.random.seed(42)
post_cynic, coop_cynic = simulate("cynic")
post_optim, coop_optim = simulate("optimist")

# Plot posterior expected cooperation
plt.figure(figsize=(9,5))
plt.plot(post_cynic, label="Cynic's Posterior P(cooperate)")
plt.plot(post_optim, label="Optimist's Posterior P(cooperate)")
plt.xlabel("Round")
plt.ylabel("Estimated partner cooperation")
plt.title("Self-reinforcing beliefs in trembling-hand Iterated PD")
plt.legend()
plt.tight_layout()
plt.savefig("img/simple.png")
plt.show()
