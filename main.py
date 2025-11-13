import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path 
import os
import random
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class Conditions:
    def __init__(self, epsilon, delta, g, Lx, Ly, N):
        self.epsilon = epsilon
        self.delta = delta
        self.g = g
        
        self.Lx, self.Ly = Lx, Ly
        self.N = N
        
        x = np.linspace(0, Lx, self.N)
        y = np.linspace(0, Ly, self.N)
        self.dx = Lx / (self.N - 1)
        self.dy = Ly / (self.N - 1)
        self.mesh = np.meshgrid(x, y, indexing='ij')

        self.kx = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dy)
        self.kx, self.ky = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2
        

        self.linear = self.epsilon - (1 - self.k2)**2
        
        print(f"  ε={self.epsilon:.3f}, δ={self.delta:.3f}, g(γ)={self.g:.3f}")
        print(f"  Domain: {self.Lx}x{self.Ly}, Resolution: {self.N}x{self.N}")

    def initial_condition(self):
        u = np.random.uniform(-np.sqrt(self.epsilon), np.sqrt(self.epsilon), (self.N, self.N))
        return u

    def nonlinear(self, u):
        N_real = -self.delta * u**2 - self.g * u**3
        return np.fft.fft2(N_real)

def generate_steady_state_data():
    #manually adjust to generate image with particular parameters -> can randomise
    Epsilon = 0.6
    Delta = 0.406
    Gamma = 0.196
    Lx, Ly = 50.0, 50.0
    Resolution = 100
    dt = 0.1
    total_time = 2000.0 # Run for longer to ensure steady state
    
    # detecting steady state to stop simulation
    Steadystate_tol = 1e-5  # Tolerance
    History_len = 100       # Steps to average for |du/dt|
    Plot_freq = 100         # Update plot less often to speed up
    
    # Non-Hard Coded Data folder relative to the main script
    ss_data_dir = Path(__file__).parent / "data"
    # File paths
    ss_data_filename = ss_data_dir / "pattern_eps0.600_delta0.406_gamma0.196_PINN.npy"
    ss_plot_filename = ss_data_dir / "pattern_eps0.600_delta0.406_gamma0.196_PINN.png"
    # Ensure the data directory exists
    ss_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured directory exists:\n - {ss_data_dir}")

    cond = Conditions(
        epsilon=Epsilon, delta=Delta, g=Gamma,
        Lx=Lx, Ly=Ly, N=Resolution
    )
    
    u = cond.initial_condition()
    u_hat = np.fft.fft2(u)

    steps_total = int(total_time / dt)
    change_history = deque(maxlen=History_len)
    u_prev = u.copy()

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: Real space pattern
    im1 = ax1.imshow(u, cmap="RdBu", origin="lower", interpolation="bilinear",
                     extent=[0, cond.Lx, 0, cond.Ly])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.set_title(f't = {0.0:.1f}, ε = {cond.epsilon:.3f}')
    
    # Plot 2: Fourier space (FFT)
    # Calculate initial FFT data for plot
    fft_plot_data = np.log10(1 + np.abs(np.fft.fftshift(u_hat)))
    im2 = ax2.imshow(fft_plot_data, cmap="inferno", origin="lower", 
                     extent=[cond.kx.min(), cond.kx.max(), cond.ky.min(), cond.ky.max()])
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_xlabel('kx (shifted)'); ax2.set_ylabel('ky (shifted)')
    ax2.set_title('log10 | 1 + FFT(u)|')
    
    plt.tight_layout()

    print("--- Starting Simulation ---")
    print(f"Running until t={total_time} or steady state (avg |du/dt| < {Steadystate_tol:.1e})")
    
    i = 0
    current_time = 0.0
    avg_change = 0.0
    
    while True:
        current_time = i * dt
        
        # Get real-space 'u' for this timestep
        u = np.fft.ifft2(u_hat).real
        
        # --- Steady state check ---
        if i > 0:
            change = np.mean(np.abs(u - u_prev)) / dt
            change_history.append(change)
            if len(change_history) == change_history.maxlen:
                avg_change = np.mean(change_history)
        
        u_prev = u.copy() # Store current u for next step's comparison
        
        # Calculate nonlinear term in Fourier space
        N_hat = cond.nonlinear(u) 
        # Evolve u_hat
        u_hat_new = (u_hat + dt * N_hat) / (1 - dt * cond.linear)

        # --- Update for next iteration ---
        u_hat = u_hat_new

        # --- Update live plot ---
        if i % Plot_freq == 0:
            im1.set_data(u)
            im1.set_clim(u.min(), u.max())
            if i > change_history.maxlen:
                title_str = f"t = {current_time:.1f}, |du/dt|~{avg_change:.2e}"
            else:
                title_str = f"t = {current_time:.1f}, (collecting stats)"
            ax1.set_title(title_str)
            
            fft_plot_data = np.log10(1 + np.abs(np.fft.fftshift(u_hat)))
            im2.set_data(fft_plot_data)
            im2.set_clim(fft_plot_data.min(), fft_plot_data.max())

            plt.pause(0.01)
    
        if (not np.isfinite(u).all() or
            not np.isfinite(u_hat).all() or
            np.max(np.abs(u_hat)) > 1e8):
            print(f"Instability detected at step {i}, stopping simulation.")
            break 

        if i > change_history.maxlen and avg_change < Steadystate_tol:
            print(f"\n--- Steady state reached at t = {current_time:.1f} ---")
            print(f"Average change |du/dt| = {avg_change:.2e} < {Steadystate_tol:.2e}")
            break 

        if i >= steps_total:
            print(f"\n--- Max time reached at t = {current_time:.1f} ---")
            print("Simulation finished without reaching steady state tolerance.")
            break 


        i += 1
    
    plt.ioff()

    # Save the final steady state images
    u_ss = np.fft.ifft2(u_hat).real
    #This does the numpy array
    try:
        ss_data_path = ss_data_dir / ss_data_filename 
        
        np.save(ss_data_path, u_ss)
        print(f"\n Saved SS data to:\n{ss_data_path}")
    except Exception as e:
        print(f"\nERROR saving SS data: {e}")


    try:
        # Update plot one last time to show final state
        im1.set_data(u_ss)
        im1.set_clim(u_ss.min(), u_ss.max())
        ax1.set_title(f"Steady State at t = {current_time:.1f}, |du/dt|~{avg_change:.2e}")
        fft_plot_data = np.log10(1 + np.abs(np.fft.fftshift(u_hat)))
        im2.set_data(fft_plot_data)
        im2.set_clim(fft_plot_data.min(), fft_plot_data.max())
        ss_plot_path = ss_data_dir / ss_plot_filename
        fig.savefig(ss_plot_path, dpi=150, bbox_inches='tight')
        print(f"Successfully saved FINAL SS plot to:\n{ss_plot_path}")
        
    except Exception as e:
        print(f"\nERROR saving SS plot: {e}")

    # Keep the plot window open
    plt.show()
    
    return

# Main execution
if __name__ == "__main__":
    generate_steady_state_data()
