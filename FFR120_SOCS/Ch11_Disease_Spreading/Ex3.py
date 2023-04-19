# %% a)
import os
import numpy as np
import matplotlib.pyplot as plt
from agents import Agent
import time as time
from SIR_model import SIRD_simulation

start_time = time.time()

# Define parameters
gamma = 0.005  # recovery probability
beta = 0.6    # infection probability
mu_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1]  # death probability
number_of_iterations = 3
D_inf_list = []
for mu in mu_list:
    print('mu =', mu)
    D_inf = SIRD_simulation(gamma, beta, mu, number_of_iterations)
    D_inf_list.append(D_inf)

print('D_inf_list =', D_inf_list)
runtime = (time.time() - start_time)/60
print('Runtime:', runtime, 'minutes')

# %% Plot the results
w_dir = os.getcwd()
folder = 'Plots_Ex3'
if not os.path.exists(folder):
    os.makedirs(folder)

plt.figure()
plt.plot(mu_list, D_inf_list, 'o', markersize=5, color='black')
plt.xscale('log')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$D_\infty$')
plt.xticks([1e-5, 1e-4, 1e-4, 1e-3, 1e-2, 1e-1])
plt.title(label=r'number of dead for different $\mu$, $\beta$ = ' + str(beta) + ', $\gamma$ = ' + str(gamma))

savename = 'mortality_beta' + str(beta) + '_gamma' + str(gamma)
plt.savefig(w_dir+'/'+folder+'/'+savename+'.png', dpi=300)
plt.show()