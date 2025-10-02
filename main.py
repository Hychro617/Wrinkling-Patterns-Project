import numpy as np
import matplotlib.pyplot as plt
import random

<<<<<<< HEAD
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
=======
class Conditions:

    def __init__(self):
        pass

    def parameters(self, val1, val2): #Making the parameters
        self.epsilon = random.uniform(0, 0.3) 
        self.delta = val1
        self.g = val2
        return self.epsilon, self.delta, self.g
    
    def domain(self, Lx, Ly, resolution): # Initialising the domain
        self.Lx, self.Ly = Lx, Ly
        self.N = resolution
        x = np.linspace(0, Lx, self.N) 
        self.dy = Ly / (self.N - 1)
        y = np.linspace(0, Ly, self.N)
        self.dx = Lx / (self.N - 1)
        self.mesh = np.meshgrid(x, y)
        return self.mesh

    def kvalues(self):
        # Defining variables the variables
        kx = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)  
        ky = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dy)
        kx, ky = np.meshgrid(kx, ky)
        self.k2 = kx**2 + ky**2
        u = 0.01 * (2 * np.random.rand(self.N, self.N) - 1)
        return u

    def linear(self): #Defining linear part of equation
        linear = self.epsilon - (1 + self.k2)**2
        return linear
    
    def nonlinear(self, u): #Defining non-linear part of equation
        u2 = u**2
        u2_hat = np.fft.fft2(self.delta * u2)
        u3 = u**3
        u3_hat = np.fft.fft2(-self.g * u3)
        N_hat = u2_hat + u3_hat
        return N_hat

#Initialise all conditions
cond = Conditions()
epsilon, delta, g = cond.parameters(0.5, 1.0)
cond.domain(Lx=20, Ly=20, resolution=256)
u = cond.kvalues()
u_hat = np.fft.fft2(u)
linear = cond.linear()

#Simulation parameters
dt = 0.001
T = 100.0
steps = int(T / dt)

#Visualization setup
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(u, cmap="RdBu", origin="lower", interpolation="bilinear", 
               extent=[0, cond.Lx, 0, cond.Ly])
>>>>>>> ab24c25c74c0323394fbe4609d013307a583b48a
fig.colorbar(im, ax=ax)
ax.set_xlabel('x')
ax.set_ylabel('y')

<<<<<<< HEAD
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
        
=======
# Simulation loop
for i in range(steps):
    u = np.fft.ifft2(u_hat).real
    N_hat = cond.nonlinear(u)
    u_hat = (u_hat + dt * N_hat) / (1 - dt * linear)

    if i % 500 == 0:
        im.set_data(u)
        im.set_clim(u.min(), u.max())
        ax.set_title(f"t = {i*dt:.2f}, ε = {epsilon:.3f}")
        plt.pause(0.01)
>>>>>>> ab24c25c74c0323394fbe4609d013307a583b48a

plt.ioff()
plt.show()

# Final pattern analysis
final_u = np.fft.ifft2(u_hat).real
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(final_u, cmap="RdBu", origin="lower", interpolation="bilinear")
plt.colorbar()
plt.title("Final Pattern")
<<<<<<< HEAD
=======

plt.subplot(1, 2, 2)
power_spectrum = np.abs(np.fft.fftshift(u_hat))**2
plt.imshow(np.log1p(power_spectrum), cmap="hot", origin="lower")
plt.colorbar()
plt.title("Power Spectrum")
plt.tight_layout()
plt.show()
>>>>>>> ab24c25c74c0323394fbe4609d013307a583b48a
