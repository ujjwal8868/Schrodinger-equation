# Schrodinger-equation(Quantum mechanics)
The Schrodinger equation is the fundamental equation of quantum mechanics that describes how the quantum state(wave function) of a physical system evolves with time.
Two types of Schrodinger equation:
(i) Time-dependent Schrödinger Equation(TDSE)
(ii) Time-independent Schrödinger Equation(TISE)

# Code

# Time-dependent 1D Schrödinger Equation (Crank–Nicolson) — Python simulation and animation
# This code numerically solves the 1D Time-Dependent Schrödinger Equation using the
# Crank–Nicolson scheme for a Gaussian wave packet in a potential V(x).

# Outputs:
#  - A short animation saved as /mnt/data/tdse_wavepacket.gif showing |ψ(x,t)|^2 over time
#  - A final static plot comparing initial and final probability densities

# Notes:
#  - Uses atomic units: ħ = 1, m = 1 for simplicity (you can change constants below)
#  - The code intentionally avoids seaborn and specific colors per tool rules.
#  - If the animation file is created it will be available for download after execution.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from math import pi
import time

# --- Physical & numerical parameters ---
hbar = 1.0      # Planck's reduced constant (set to 1 for convenience)
m = 1.0         # particle mass (set to 1)
L = 200.0       # spatial domain length
N = 800         # number of spatial grid points (adjust for speed/accuracy)
dx = L / N
x = np.linspace(-L/2, L/2, N)

# Time parameters
dt = 0.02       # time step
n_steps = 600   # number of time steps to evolve (adjust for longer/shorter sim)
save_every = 6  # frames saved for the animation (to reduce gif size)

# --- Potential: example - free particle, barrier, or harmonic oscillator ---
# Choose scenario by uncommenting the desired potential

# 1) Free particle (V = 0)
V = np.zeros_like(x)

# 2) Single Gaussian barrier - uncomment to use
#V0 = 1.0
#V = V0 * np.exp(- (x/2.0)**2)

# 3) Harmonic oscillator - uncomment to use
#k = 0.002
#V = 0.5 * k * x**2

# --- Initial wavefunction: Gaussian wavepacket ---
x0 = -40.0      # initial center
k0 = 2.0        # average momentum (wave number)
sigma = 5.0     # width of the Gaussian

# Gaussian wavepacket (normalized)
psi0 = (1/(sigma * np.sqrt(pi)))**0.5 * np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j * k0 * x)
# Normalize explicitly (numerical integration)
norm0 = np.sqrt(np.trapz(np.abs(psi0)**2, x))
psi0 /= norm0

# --- Construct Crank-Nicolson matrices ---
# Hamiltonian kinetic term discretized with second-order central differences
coef = -hbar**2 / (2*m * dx**2)

# Tridiagonal Hamiltonian H = coef * tridiag(1, -2, 1) + diag(V)
main_diag = np.full(N, -2.0) * coef + V
off_diag = np.full(N-1, 1.0) * coef

# Build CN matrices A and B where:
# A = I - i*dt/(2*hbar) * H
# B = I + i*dt/(2*hbar) * H
# We will construct A as a dense banded matrix for simplicity (N~800 is reasonable).
I = np.eye(N, dtype=complex)
H = np.zeros((N, N), dtype=complex)
H[np.arange(N), np.arange(N)] = main_diag
H[np.arange(N-1), np.arange(1, N)] = off_diag
H[np.arange(1, N), np.arange(N-1)] = off_diag

A = I - 1j * dt / (2*hbar) * H
B = I + 1j * dt / (2*hbar) * H

# Pre-factorize A using numpy.linalg for repeated solves (LU would be better for large N)
# Using numpy.linalg.solve repeatedly is fine for moderate N here.
# --- Time evolution ---
psi = psi0.copy()
frames = []  # store frames for animation

start_time = time.time()
for n in range(n_steps):
    # Right-hand side
    rhs = B.dot(psi)
    # Solve A psi_{n+1} = rhs
    psi = np.linalg.solve(A, rhs)
    # Renormalize occasionally to reduce accumulated numerical drift
    if n % 50 == 0:
        norm = np.sqrt(np.trapz(np.abs(psi)**2, x))
        psi /= norm
    # Save frames for animation
    if n % save_every == 0:
        frames.append(np.abs(psi)**2)

end_time = time.time()
runtime = end_time - start_time

# --- Create animation ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(x[0], x[-1])
ax.set_ylim(0, np.max(np.abs(psi0)**2)*1.8)
ax.set_xlabel('x')
ax.set_ylabel(r'$|\psi(x,t)|^2$')
ax.set_title('Time evolution of probability density |ψ|^2 (Crank–Nicolson)')

line, = ax.plot(x, frames[0])

def animate(i):
    line.set_ydata(frames[i])
    ax.set_title(f'Time evolution of |ψ|^2 — frame {i+1}/{len(frames)}')
    return (line,)

anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50, blit=True)

# Save animation as GIF
out_path = '/mnt/data/tdse_wavepacket.gif'
try:
    writer = animation.PillowWriter(fps=20)
    anim.save(out_path, writer=writer)
    saved = True
except Exception as e:
    saved = False
    print("Could not save animation as GIF:", e)

plt.close(fig)  # don't display the large animation twice in the notebook output

# --- Final static plot: initial vs final probability density ---
plt.figure(figsize=(8,4))
plt.plot(x, np.abs(psi0)**2, label='Initial |ψ|^2')
plt.plot(x, np.abs(psi)**2, label='Final |ψ|^2 (after evolution)')
plt.xlabel('x')
plt.ylabel(r'$|\psi(x)|^2$')
plt.title('Initial vs Final probability densities')
plt.legend()
plt.tight_layout()
plt.show()
