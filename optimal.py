import numpy as np
import matplotlib.pyplot as plt

# Parameters
k = 2.0            # speed of belief drift
R, S, T, P = 3, 0, 5, 1
dt = 0.03
p_grid = np.linspace(0, 1, 101)
rho_vals = np.linspace(0.05, 1.0, 40)
N = len(p_grid)

# Pre-allocate policy grid: 1=Cooperate, 0=Defect
policy = np.zeros((len(rho_vals), N))

# Value iteration settings
max_iter = 400
tol = 1e-6

for ridx, rho in enumerate(rho_vals):
    gamma = np.exp(-rho * dt)
    
    V = np.zeros(N)  # start all zeros
    for _ in range(max_iter):
        V_old = V.copy()
        # compute V for each state under both controls
        # Transition indices
        next_C_idx = np.clip(((p_grid + dt*k*(1 - p_grid)) * 100).round().astype(int), 0, 100)
        next_D_idx = np.clip(((p_grid + dt*k*(0 - p_grid)) * 100).round().astype(int), 0, 100)
        
        imm_C = R * p_grid + S * (1 - p_grid)
        imm_D = T * p_grid + P * (1 - p_grid)
        
        V_C = imm_C + gamma * V_old[next_C_idx]
        V_D = imm_D + gamma * V_old[next_D_idx]
        
        V = np.maximum(V_C, V_D)
        if np.max(np.abs(V - V_old)) < tol:
            break
    
    # store optimal policy: 1 if cooperate else 0
    policy[ridx] = (V_C >= V_D).astype(float)

# Plot heatmap of optimal action regions
plt.figure(figsize=(8,6))
plt.imshow(policy, origin='lower', aspect='auto',
           extent=[0,1,rho_vals[0], rho_vals[-1]], cmap='coolwarm')
plt.colorbar(label='Optimal action (1 = Cooperate, 0 = Defect)')
plt.xlabel("Belief p that partner cooperates")
plt.ylabel("Discount rate ρ (higher = myopic)")
plt.title("HJB-derived optimal action surface")
plt.tight_layout()
plt.show()
