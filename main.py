import numpy as np
import matplotlib.pyplot as plt

# Defining variables - ADJUSTED PARAMETERS
epsilon = 0.2
delta = 0.5
g = 1.0

# Defining the Domain
Lx = 100  # Larger domain for better pattern formation
Ly = 100
N = 256  # Higher resolution
dx = Lx / (N )
x = np.linspace(0, Lx, N) 
dy = Ly / (N)
y = np.linspace(0, Ly, N)

kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)  
ky = 2 * np.pi * np.fft.fftfreq(N, d=dy)
kx, ky = np.meshgrid(kx, ky)
k2 = kx**2 + ky**2

linear = epsilon - (1 + k2)**2  

# Initial conditions
u = np.random.uniform(-0.1, 0.1, (N, N))
u_hat = np.fft.fft2(u)

# Nonlinear part 
def nonlinear(u, delta=0.5, g=1.0):
    u2 = u**2
    u2_hat = np.fft.fft2(delta * u2)
    u3 = u**3
    u3_hat = np.fft.fft2(-g * u3)
    N_hat = u2_hat + u3_hat
    return N_hat

# Simulation parameters
dt = 0.01  # timestep
T = 100.0  
steps = int(T / dt)

# Visualization
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(u, cmap="RdBu", origin="lower", interpolation="bilinear", 
               extent=[0, Lx, 0, Ly])
fig.colorbar(im, ax=ax)
ax.set_xlabel('x')
ax.set_ylabel('y')

# IMPROVED integration scheme
for i in range(steps):
    u = np.fft.ifft2(u_hat).real
    
    # Semi-implicit Euler with exponential time differencing
    N_hat = nonlinear(u, delta=delta, g=g)
    
    # More stable integration
    u_hat = (u_hat + dt * N_hat) / (1 - dt * linear)
        #u_hat = u_hat + dt * (linear * u_hat + N_hat)  # Original explicit Euler

    if i % 100 == 0:
        # Remove high-frequency noise
        u_hat[np.abs(k2) > np.max(k2)/4] *= 0.9
    
    if i % 500 == 0:
        im.set_data(u)
        im.set_clim(u.min(), u.max())
        ax.set_title(f"t = {i*dt:.2f}, ε = {epsilon}")
        plt.pause(0.01)
        

plt.ioff()
plt.show()

# Final pattern analysis
final_u = np.fft.ifft2(u_hat).real
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(final_u, cmap="RdBu", origin="lower", interpolation="bilinear")
plt.colorbar()
plt.title("Final Pattern")
