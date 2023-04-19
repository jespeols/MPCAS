# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from agents import Agent
import time as time
from SIR_model import SIRI_simulation

start_time = time.time()

# Define parameters
gamma = 0.01  # recovery probability
beta = 0.6    # infection probability
alpha = 1e-3  # immunity loss probability
maxIter = 10000

infected_hist, healthy_hist = SIRI_simulation(gamma, beta, alpha, maxIter)

runtime = (time.time() - start_time)/60
print('Runtime:', runtime, 'minutes')

# %% Plot the results
w_dir = os.getcwd()
folder = 'Plots_Ex4'
if not os.path.exists(folder):
    os.makedirs(folder)

plt.figure()
plt.plot(healthy_hist, label='Healthy')
plt.plot(infected_hist, label='Infected')
plt.title('Disease dynamics' + r', $\beta$ = ' + str(beta) + r', $\gamma$ = ' + str(gamma) + r', $\alpha$ = ' + str(alpha))
plt.xlabel('time steps')
plt.ylabel('number of agents')
plt.ylim(0, 1000)
plt.legend(frameon=False)
savename = 'gamma_'+str(gamma)+'_beta_'+str(beta)+'_alpha_'+str(alpha)+'.png'
plt.savefig(os.path.join(w_dir, folder, savename), dpi=300)

# %% b) dependence on fraction initially infected
import os
import numpy as np
import matplotlib.pyplot as plt
from agents import Agent
import time as time
from SIR_model import SIRI_simulation_2

start_time = time.time()

# Define parameters
gamma = 0.01  # recovery probability
beta = 0.6    # infection probability
alpha = 5e-3  # immunity loss probability
fraction_infected = 0.2
maxIter = 10000

infected_hist, healthy_hist = SIRI_simulation_2(gamma, beta, alpha, fraction_infected, maxIter)

runtime = (time.time() - start_time)/60
print('Runtime:', runtime, 'minutes')

# %% Plot the results
w_dir = os.getcwd()
folder = 'Plots_Ex4'
if not os.path.exists(folder):
    os.makedirs(folder)

plt.figure()
plt.plot(healthy_hist, label='Healthy')
plt.plot(infected_hist, label='Infected')
plt.title('Disease dynamics' + r', $\beta$ = ' + str(beta) + r', $\gamma$ = ' + str(gamma) + r', $\alpha$ = ' + str(alpha) + ', ' + str(int(fraction_infected*100)) + '% infected')
plt.xlabel('time steps')
plt.ylabel('number of agents')
plt.ylim(0, 1000)
plt.legend(frameon=False)
savename = 'gamma_'+str(gamma)+'_beta_'+str(beta)+'_alpha_'+str(alpha)+ '_' + str(int(fraction_infected*100)) + '%_infected'+'.png'
plt.savefig(os.path.join(w_dir, folder, savename), dpi=300)
