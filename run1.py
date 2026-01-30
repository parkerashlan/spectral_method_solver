import numpy as np
import matplotlib.pyplot as plt
from utils.params import Params
from solver.fft_solver import FFT2DSolver

from utils.animate import show_animation


# Initialize parameters
# f = lambda x, y: 4.0*np.sin(4*np.pi*x)*np.cos(4*np.pi*y/6)
f = lambda x, y: 4.0*(+np.exp(-64*(x-0.25)**2-64*(y+0.25)**2)+np.exp(-64*(x+0.25)**2-64*(y-0.25)**2)-np.exp(-64*(x+0.25)**2-64*(y+0.25)**2)-np.exp(-64*(x-0.25)**2-64*(y-0.25)**2))

params = Params(Nx=64, 
                Ny=64, 
                Lx=4.0, 
                Ly=4.0, 
                dt=1.0/16.0, 
                nt=512, 
                nt_save=10, 
                f=f)

# Initialize solver
solver = FFT2DSolver(params)

# Solve
solver_results = solver.solve()

wzq = solver_results['vorticity']
cfl = solver_results['cfl']
vx = solver_results['vx']
vy = solver_results['vy']
psi = solver_results['psi']

print(wzq.shape)

# Plot
max_init = np.max(wzq[:,:,0])
min_init = np.min(wzq[:,:,0])

show_animation(wzq, cmap='jet', vmin=min_init, vmax=max_init, title='Vorticity', colorbar=True, figsize=(8, 6), save_path='wzq.mp4')