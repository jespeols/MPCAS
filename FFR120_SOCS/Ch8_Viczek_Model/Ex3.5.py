# %% Run simulation
import os
import numpy as np 
import matplotlib.pyplot as plt
import time
from scipy.spatial import cKDTree
from copy import deepcopy
from Viczek_model_functions import *

w_dir = os.getcwd()

start_time = time.time() 

# Define parameters
L = 100     # Box is of size (L, L), with coordinates ([0, L], [0, L]) for cKDTree
N = 100     # Number of particles
V = 1
time_step = 1
noise = 0.1
R_f_list = [1, 2, 5, 10]
folder = 'Plots_Ex3.5'

S = int(1e4)    # number of time steps

# Generate initial particle configuration
r_initial = np.random.uniform(0, L, (N, 2))
thetas_initial = np.random.uniform(0, 2*np.pi, (N,))
v_initial = np.array([np.zeros((N, 2))+V])

# plot initial configuration
plot_particle_config(0, r_initial, 0, S, noise, folder)

for R_f in R_f_list:
    print('R_f =', R_f)

    r = deepcopy(r_initial)
    thetas = deepcopy(thetas_initial)
    v = deepcopy(v_initial)

    psi_list = []
    c_list = []
    for t in range(S):
        # print('time step: ' + str(t))

        tree = cKDTree(r, boxsize=L)
        particles_in_range = tree.query_ball_point(r, R_f)

        if t > 0:   # Update particles
            # update orientations
            thetas = update_orientations(thetas, particles_in_range, noise, time_step)

            # update velocities
            v = update_velocities(v, thetas, V)

            # update positions
            r = update_positions(r, v, L, time_step)

        # calculate coefficients
        psi = calc_alignment_coeff(v, V)
        psi_list.append(psi)

        c = calc_clustering_coeff(r, R_f)
        c_list.append(c)
        
        if t in (10, 100, 500, 1000):
            plot_particle_config(t, r, R_f, S, noise, folder)

    # plot final configuration
    plot_particle_config(S, r, R_f, S, noise, folder)

    plt.figure()
    plt.plot(range(S), c_list, label=r'$c$')
    plt.plot(range(S)[1:-1], psi_list[1:-1], label=r'$\psi$')
    plt.title(r'Alignment and clustering coefficients,  $R_f = $' + str(R_f) + ', $\eta$ = ' + str(noise))
    plt.xlabel('t')
    plt.ylim(0, 1.02)
    plt.legend()
    savename = 'coefficients_Rf' + str(R_f) + '_eta' + str(noise)
    plt.savefig(w_dir+'/'+folder+'/Rf'+str(R_f)+'/'+savename+'.png', dpi=300)
    plt.show()

stop_time = time.time()
runtime = (stop_time-start_time)/60
print('Runtime: ', round(runtime, 1), 'minutes')