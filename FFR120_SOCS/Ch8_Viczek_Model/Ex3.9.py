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
L = 1000     # Box is of size (L, L), with coordinates ([0, L], [0, L]) for cKDTree
N = 100     # Number of particles
V = 3
h_list = [0, -1, -2, -5, -10, -15]
s = 5   # number of time steps to interpolate from
time_step = 1
noise = 0.4
R_f = 20
folder = 'Plots_Ex3.9'

S = int(1e4)    # number of time steps

# Generate initial particle configuration
r_initial = np.random.uniform(0, L, (N, 2))
thetas_initial = np.random.uniform(0, 2*np.pi, (N,))
v_initial = np.array([np.zeros((N, 2))+V])

c_array = np.zeros((len(h_list), S))
psi_array = np.zeros((len(h_list), S))

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

        if t > s:   # Update particles
            # update orientations
            thetas_extrap = get_extrapolated_thetas(t, time_step, s, thetas_record, h)
            thetas = update_orientations(thetas_extrap, particles_in_range, noise, time_step)
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
    
    c_array[h_list.index(h), :] = c_list
    psi_array[h_list.index(h), :] = psi_list

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

plt.figure()
# for i in range(len(h_list)):
for i in (0, 2, -2, -1):
    plt.plot(range(S), c_array[i,:], label='h = ' + str(h_list[i]))

plt.title('clustering for different h')
plt.legend()
plt.xlabel('t')
plt.ylabel('c')
plt.ylim(0, 1.02)
plt.savefig(w_dir+'/'+folder+'/clustering_h.png', dpi=300)
plt.show()

plt.figure()
for i in (0, 2, -2, -1):
    plt.plot(range(S), psi_array[i,:], label='h = ' + str(h_list[i]))

plt.title('alignment for different h')
plt.legend()
plt.xlabel('t')
plt.ylabel(r'$\psi$')
plt.ylim(0, 1.02)
plt.savefig(w_dir+'/'+folder+'/alignment_h.png', dpi=300)

plt.show()
stop_time = time.time()
runtime = (stop_time-start_time)/60
print('Runtime: ', round(runtime, 1), 'minutes')