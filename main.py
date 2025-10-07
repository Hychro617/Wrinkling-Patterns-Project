import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class Conditions:
    def __init__(self):
        pass

    def parameters(self, val1, val2):
        self.epsilon = 1.3  # small random growth
        self.delta = val1
        self.g = val2
        return self.epsilon, self.delta, self.g

    def domain(self, Lx, Ly, resolution):
        self.Lx, self.Ly = Lx, Ly
        self.N = resolution

        x = np.linspace(0, Lx, self.N)
        y = np.linspace(0, Ly, self.N)
        self.dx = Lx / (self.N - 1)
        self.dy = Ly / (self.N - 1)
        self.mesh = np.meshgrid(x, y, indexing='ij') 

        # Wavenumbers
        kx = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dy)
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        self.k2 = kx**2 + ky**2
        return self.mesh

    def initial_condition(self):
        u = np.random.uniform(-np.sqrt(self.epsilon), np.sqrt(self.epsilon), (self.N, self.N))
        return u

    def linear(self):
        L = self.epsilon - (1 - self.k2)**2
        return L

    def nonlinear(self, u):
        N_hat = np.fft.fft2(-self.g * u**3 + self.delta * u**2)
        return N_hat

# Initialize
cond = Conditions()
epsilon, delta, g = cond.parameters(0.7, 1)
cond.domain(Lx=150, Ly=150, resolution=1024)

u = cond.initial_condition()
u_hat = np.fft.fft2(u)
linear = cond.linear()

# Sliding window for last few steps
queue_size = 2
queue = deque(maxlen=queue_size)

# Simulation parameters
dt = 0.1
T = 100.0
steps = int(T / dt)

# Setup interactive plot
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(u, cmap="RdBu", origin="lower", interpolation="bilinear",
               extent=[0, cond.Lx, 0, cond.Ly])
fig.colorbar(im, ax=ax)
ax.set_xlabel('x')
ax.set_ylabel('y')

# Simulation loop
for i in range(steps):
    u = np.fft.ifft2(u_hat).real
    u_hat_new = (u_hat + dt * cond.nonlinear(u)) / (1 - dt * linear)

    # Check for blow-up
    if not np.isfinite(u).all() or not np.isfinite(u_hat_new).all() or np.max(np.abs(u_hat_new)) > 1e8:
        print(f"Instability detected at step {i}, stopping simulation.")
        break

    # Add the last stable one to the queue
    queue.append({'u': np.copy(u), 'u_hat': np.copy(u_hat_new)})

    # Update for next iteration
    u_hat = u_hat_new

    # Update live plot every 50 steps
    if i % 50 == 0:
        im.set_data(u)
        im.set_clim(u.min(), u.max())
        ax.set_title(f"t = {i*dt:.2f}, ε = {epsilon:.3f}")
        plt.pause(0.01)

plt.ioff()
plt.show()

# Plot the last stable time step
last_good = queue[-1]  # last stable step
final_u = np.fft.ifft2(last_good['u_hat']).real
final_u_hat = last_good['u_hat']

plt.figure(figsize=(12, 5))

# Final real-space pattern
plt.subplot(1, 2, 1)
plt.imshow(final_u, cmap="RdBu", origin="lower", interpolation="bilinear",
           extent=[0, cond.Lx, 0, cond.Ly])
plt.colorbar()
plt.title("Final Pattern")

# Fourier spectrum
plt.subplot(1, 2, 2)
plt.imshow(np.log1p(np.abs(np.fft.fftshift(final_u_hat))), cmap="viridis",
           origin="lower", extent=[-cond.Lx/2, cond.Lx/2, -cond.Ly/2, cond.Ly/2])
plt.colorbar()
plt.title("Fourier Spectrum")

plt.tight_layout()
plt.show()
