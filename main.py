import numpy as np
import matplotlib.pyplot as plt

class Conditions:

    def __init__(self):
        pass

    def parameters(self, val1, val2):
        self.epsilon = np.random.uniform(0, 0.3)  # small random growth
        self.delta = val1
        self.g = val2
        return self.epsilon, self.delta, self.g

    def domain(self, Lx, Ly, resolution):
        self.Lx, self.Ly = Lx, Ly
        self.N = resolution

        # Spatial grid
        x = np.linspace(0, Lx, self.N)
        y = np.linspace(0, Ly, self.N)
        self.dx = Lx / (self.N - 1)
        self.dy = Ly / (self.N - 1)
        self.mesh = np.meshgrid(x, y, indexing='ij')  # Important for FFT

        # Wavenumbers
        kx = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dy)
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        self.k2 = kx**2 + ky**2
        return self.mesh

    def initial_condition(self):
        # Small random noise
        u = np.random.uniform(-0.1, 0.1, (self.N, self.N))
        return u

    def linear(self):
        # Linear operator in Fourier space
        L = self.epsilon - (1 - self.k2)**2
        return L

    def nonlinear(self, u):
        # Nonlinear term in Fourier space (cubic only)
        N_hat = np.fft.fft2(-self.g * u**3)
        return N_hat


# ----------------------
# Initialize simulation
# ----------------------
cond = Conditions()
epsilon, delta, g = cond.parameters(0, 1)
cond.domain(Lx=100, Ly=100, resolution=1024)

u = cond.initial_condition()
u_hat = np.fft.fft2(u)
linear = cond.linear()

# Simulation parameters
dt = 0.001   # slightly larger for efficiency
T = 1000.0    # shorter for testing
steps = int(T / dt)

# ----------------------
# Visualization setup
# ----------------------
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(u, cmap="RdBu", origin="lower", interpolation="bilinear",
               extent=[0, cond.Lx, 0, cond.Ly])
fig.colorbar(im, ax=ax)
ax.set_xlabel('x')
ax.set_ylabel('y')

# ----------------------
# Simulation loop
# ----------------------
for i in range(steps):
    u = np.fft.ifft2(u_hat).real
    N_hat = cond.nonlinear(u)
    u_hat = (u_hat + dt * N_hat) / (1 - dt * linear)

    if i % 50 == 0:
        im.set_data(u)
        im.set_clim(u.min(), u.max())
        ax.set_title(f"t = {i*dt:.2f}, ε = {epsilon:.3f}")
        plt.pause(0.01)

plt.ioff()
plt.show()

# ----------------------
# Final pattern analysis
# ----------------------
final_u = np.fft.ifft2(u_hat).real

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(final_u, cmap="RdBu", origin="lower", interpolation="bilinear",
           extent=[0, cond.Lx, 0, cond.Ly])
plt.colorbar()
plt.title("Final Pattern")

plt.subplot(1, 2, 2)
