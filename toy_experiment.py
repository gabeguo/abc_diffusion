import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

output_folder = 'toy_experiment_outputs'
import os
os.makedirs(output_folder, exist_ok=True)

x0 = 1.0
N = 4000          # discretization steps
M = 1000          # number of trajectories
eps = 1e-6
dt = 1.0 / N
t = np.linspace(0.0, 1.0, N + 1)
idx_45 = int(round(0.8 * N))   # index for t = 4/5

# ---------- Simulate X(t) ----------
X = np.zeros((M, N + 1))
X[:, 0] = x0
sqrt_dt = np.sqrt(dt)
for k in range(N):
    tk = t[k]
    if k < idx_45:
        drift = (-x0 - X[:, k]) / (0.8 - tk + eps)
    else:
        drift = ( x0 - X[:, k]) / (1.0 - tk + eps)
    X[:, k + 1] = X[:, k] + drift * dt + sqrt_dt * np.random.randn(M)

# ---------- Simulate Y(t) ----------
# Segment 1: drift (-x0 - Y)/(4/5 - t), vol sqrt(5/4)
# Segment 2: drift ( x0 - Y)/(1   - t), vol sqrt(5)
Y = np.zeros((M, N + 1))
Y[:, 0] = x0
vol1 = np.sqrt(5.0 / 4.0)
vol2 = np.sqrt(5.0)
for k in range(N):
    tk = t[k]
    if k < idx_45:
        drift = (-x0 - Y[:, k]) / (0.8 - tk + eps)
        vol = vol1
    else:
        drift = ( x0 - Y[:, k]) / (1.0 - tk + eps)
        vol = vol2
    Y[:, k + 1] = Y[:, k] + drift * dt + vol * sqrt_dt * np.random.randn(M)

plt.rcParams.update({'font.size': 15})

# ---------- Plot 1: trajectories ----------
fig1, ax = plt.subplots(figsize=(10, 6))
for i in range(M):
    ax.plot(t, X[i], color='tab:blue',   alpha=0.03, lw=0.6, linestyle='-')
    ax.plot(t, Y[i], color='tab:red',    alpha=0.03, lw=0.6, linestyle='--')
# Legend proxies
ax.plot([], [], color='tab:blue', linestyle='-',  label='X(t)')
ax.plot([], [], color='tab:red',  linestyle='--', label='Y(t)')
ax.set_xticks(np.arange(0, 1.0001, 1/5))
ax.set_yticks(np.arange(-2, 2.0001, 1))
ax.set_xlim(0, 1)
ax.set_ylim(-2, 2)
ax.set_xlabel('t')
ax.set_ylabel('process value')
ax.set_title('Trajectories of X(t) and Y(t), 1000 samples')
ax.grid(True, which='both', alpha=0.5)
ax.legend(loc='upper left')
fig1.tight_layout()
fig1.savefig(os.path.join(output_folder, 'trajectories.png'), dpi=130)

# ---------- Plot 2: 2x2 marginal histograms ----------
times = [0.5, 0.8, 0.9, 1.0]
labels = ['t = 1/2', 't = 4/5', 't = 9/10', 't = 1']
fig2, axes = plt.subplots(2, 2, figsize=(10, 6))
bins = np.linspace(-1.5, 1.5, 61)
for ax2, tt, lab in zip(axes.ravel(), times, labels):
    k = int(round(tt * N))
    ax2.hist(X[:, k], bins=bins, alpha=0.5, color='tab:blue',
             label='X', density=True)
    ax2.hist(Y[:, k], bins=bins, alpha=0.5, color='tab:red',
             label='Y', density=True)
    ax2.set_xlim(-1.5, 1.5)   # width = 3
    ax2.set_title(lab)
    ax2.legend()
    ax2.grid(True, alpha=0.4)
fig2.suptitle('Empirical marginals of X(t) vs Y(t)')
fig2.tight_layout()
fig2.savefig(os.path.join(output_folder, 'marginals.png'), dpi=130)

print('done')
print('X endpoints:', X[:, -1].mean(), '±', X[:, -1].std())
print('Y endpoints:', Y[:, -1].mean(), '±', Y[:, -1].std())
print('X at 4/5:', X[:, idx_45].mean(), '±', X[:, idx_45].std())
print('Y at 4/5:', Y[:, idx_45].mean(), '±', Y[:, idx_45].std())