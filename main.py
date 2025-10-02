import numpy as np
import matplotlib.pyplot as plt

#defining variables
epsilon = 0.5 # bifurcation parameter
delta = 0.5

#Defining the Domain
Lx = 1 
Ly = 1
N = 100 
dx = Lx / (N - 1)
x = np.linspace(0, Lx, N) 
dy = Ly / (N - 1)
y = np.linspace(0, Ly, N)


#Defining discrete wavenumbers
kx = 2 * np.pi * np.fft.fftfreq(N, Lx/N) #2*np.pi Converts to angular wavevectors)
ky = 2 * np.pi * np.fft.fftfreq(N, Ly/N)
kx, ky = np.meshgrid(kx, ky)        #2D grid of kx and ky values
k2 = kx**2 + ky**2                  #k2 is the laplace in the equation

#initial conditions
u0 = np.random.rand(N,N)    #physical space
u0_hat = np.fft.fft2(u0)   #fourier space

#fourier transform of the linear part
linear = epsilon - (1 - k2)**2  

#nonlinear part (physical to fourier)
def nonlinear(u0, delta = 0.5):
    u2 =  u0**2
    u2_hat = np.fft.fft2(delta*u2)
    u3 = u0**3
    u3_hat = np.fft.fft2(-u3)
    N = u2_hat + u3_hat
    return N

#!!periodic boundary conditions not needed!! -> fourier basis inherently periodic

#simulate fourier frequency domain


#plotting
