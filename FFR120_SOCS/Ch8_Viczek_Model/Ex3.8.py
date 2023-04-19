# %% a), b)
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
L = 1000     # Box is of size (L, L), with coordinates ([0, L], [0, L]) for cKDTree
N = 100     # Number of particles
V = 3
h_list = [0, 1, 2, 5, 10, 25]
time_step = 1
noise = 0.4
R_f = 20
folder = 'Plots_Ex3.8'

S = int(1e4)    # number of time steps

# Generate initial particle configuration
r_initial = np.random.uniform(0, L, (N, 2))
thetas_initial = np.random.uniform(0, 2*np.pi, (N,))
v_initial = np.array([np.zeros((N, 2))+V])

# plot initial configuration
plot_particle_config_2(0, r_initial, 0, S, folder)

psi_h_list = []
c_h_list = []

for h in h_list:
    print('h =', h)

    r = deepcopy(r_initial)
    thetas = deepcopy(thetas_initial)
    v = deepcopy(v_initial)

    psi_list = []
    c_list = []

    r_record = np.zeros((N, 2, S))
    thetas_record = np.zeros((N, S))

    for t in range(S):
        # print('time step: ' + str(t))

        tree = cKDTree(r, boxsize=L)
        particles_in_range = tree.query_ball_point(r, R_f)

        if t > h:   # Update particles
            # update orientations
            thetas = update_orientations(thetas_record[:,t-h-1], particles_in_range, noise, time_step)
            thetas_record[:,t] = thetas

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
            plot_particle_config_2(t, r, h, S, folder)

    # plot final configuration
    plot_particle_config_2(S, r, h, S, folder)

    # calculate average coefficients
    psi_h = np.average(psi_list[5000:-1])
    psi_h_list.append(psi_h)
    c_h = np.average(c_list[5000:-1])
    c_h_list.append(c_h)

    # plot coefficients
    plt.figure()
    plt.plot(range(S), c_list, label=r'$c$')
    plt.plot(range(S)[1:-1], psi_list[1:-1], label=r'$\psi$')
    plt.title(r'Alignment and clustering coefficients,  h = ' + str(h))
    plt.xlabel('t')
    plt.ylim(0, 1.02)
    plt.legend()
    savename = 'coefficients_h' + str(h)
    plt.savefig(w_dir+'/'+folder+'/h'+str(h)+'/'+savename+'.png', dpi=300)
    plt.show()

stop_time = time.time()
runtime = (stop_time-start_time)/60
print('Runtime: ', round(runtime, 1), 'minutes')

# %% c), d)

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
L = 1000     # Box is of size (L, L), with coordinates ([0, L], [0, L]) for cKDTree
N = 100     # Number of particles
V = 3
h_list = [0, 1, 2, 5, 10, 25]
time_step = 1
noise = 0.4
R_f = 20
folder = 'Plots_Ex3.8'

S = int(1e4)    # number of time steps

psi_h_avg = [] 
c_h_avg = []

h_list = range(26)

for h in h_list:
    print('h =', h)

    psi_h_list = []
    c_h_list = []
    for _ in range(5):

        # Generate initial particle configuration
        r = np.random.uniform(0, L, (N, 2))
        thetas = np.random.uniform(0, 2*np.pi, (N,))
        v = np.array([np.zeros((N, 2))+V])

        psi_list = []
        c_list = []

        r_record = np.zeros((N, 2, S))
        thetas_record = np.zeros((N, S))

        for t in range(S):
            # print('time step: ' + str(t))

            tree = cKDTree(r, boxsize=L)
            particles_in_range = tree.query_ball_point(r, R_f)

            if t > h:   # Update particles
                # update orientations
                thetas = update_orientations(thetas_record[:,t-h-1], particles_in_range, noise, time_step)
                thetas_record[:,t] = thetas

                # update velocities
                v = update_velocities(v, thetas, V)

                # update positions
                r = update_positions(r, v, L, time_step)

            # calculate coefficients
            psi = calc_alignment_coeff(v, V)
            psi_list.append(psi)

            c = calc_clustering_coeff(r, R_f)
            c_list.append(c)

        # calculate average coefficients
        psi_h = np.average(psi_list[5000:-1])
        psi_h_list.append(psi_h)
        c_h = np.average(c_list[5000:-1])
        c_h_list.append(c_h)
    
    # For every h, calculate the average of the 5 runs
    psi_h_avg.append(np.average(psi_h_list))
    c_h_avg.append(np.average(c_h_list))

stop_time = time.time()
runtime = (stop_time-start_time)/60
print('Runtime: ', round(runtime, 1), 'minutes')

# %% plot coefficients as function of h
fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
ax1.scatter(h_list, psi_h_avg, label=r'$\psi$')
ax1.set_xlabel('h')
ax1.set_ylabel(r'$\psi$')
ax1.set_title(r'$\psi$ as function of h')
ax1.set_xlim(0, 25)
ax1.set_ylim(0, 1.02)

ax2.scatter(h_list, c_h_avg, label=r'$c$')
ax2.set_xlabel('h')
ax2.set_ylabel(r'$c$')
ax2.set_title(r'$c$ as function of h')
ax2.set_xlim(0, 25)
ax2.set_ylim(0, 1.02)

savename = 'coefficients_vs_h'
plt.savefig(w_dir+'/'+folder+'/'+savename+'.png', dpi=300)

plt.show()