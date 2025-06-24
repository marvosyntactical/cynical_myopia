import numpy as np
import matplotlib.pyplot as plt

# --- model parameters ---
k = 2.0
R, S, T, P = 3, 0, 5, 1
dt = 0.02
p_grid = np.linspace(0, 1, 201)          # finer grid for smooth colour
rho_grid = np.linspace(0.05, 1.0, 80)    # patience spectrum
N = len(p_grid)

# pre-compute index look-ups for dynamics under C and D
next_C_idx = np.clip(np.rint((p_grid + dt * k * (1 - p_grid)) * 200).astype(int), 0, 200)
next_D_idx = np.clip(np.rint((p_grid + dt * k * (0 - p_grid)) * 200).astype(int), 0, 200)

# immediate rewards
imm_C = R * p_grid + S * (1 - p_grid)
imm_D = T * p_grid + P * (1 - p_grid)

policy = np.zeros((len(rho_grid), N))
value_diff = np.zeros_like(policy)  # V_C − V_D for colour

max_iter = 600
tol = 1e-6

for ridx, rho in enumerate(rho_grid):
    gamma = np.exp(-rho * dt)
    V = np.zeros(N)
    for _ in range(max_iter):
        V_old = V.copy()
        V_C = imm_C + gamma * V_old[next_C_idx]
        V_D = imm_D + gamma * V_old[next_D_idx]
        V = np.maximum(V_C, V_D)
        if np.max(np.abs(V - V_old)) < tol:
            break
    best_C = V_C >= V_D
    policy[ridx] = best_C.astype(float)
    value_diff[ridx] = V_C - V_D  # positive => C better

# --- vector field: horizontal velocity dp/dt ---
dpdt = k * (policy - p_grid)  # same shape as policy
# we'll down-sample for quiver clarity
skip_r = slice(None, None, 5)
skip_p = slice(None, None, 8)

P, RHO = np.meshgrid(p_grid, rho_grid)

plt.figure(figsize=(9,6))
# colour map: value_diff (seismic centred at 0)
plt.imshow(value_diff, origin='lower', aspect='auto',
           extent=[0,1,rho_grid[0], rho_grid[-1]],
           cmap='seismic')
plt.colorbar(label='Value difference V_C − V_D')

# overlay vector field arrows
plt.quiver(P[skip_r, skip_p], RHO[skip_r, skip_p],
           dpdt[skip_r, skip_p], np.zeros_like(dpdt[skip_r, skip_p]),
           pivot='mid', scale=30, color='k', alpha=0.7)

plt.xlabel("Belief p that partner cooperates")
plt.ylabel("Discount rate ρ  (higher = more myopic)")
plt.title("Smooth HJB landscape with optimal-drift vector field")
plt.tight_layout()
plt.show()
