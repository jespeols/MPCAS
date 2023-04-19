# %% a)
import os
import numpy as np
import matplotlib.pyplot as plt
from agents import Agent
import time as time
from SIR_model import SIR_simulation

start_time = time.time()

# Define parameters
gamma = 0.01  # recovery probability
beta_list = np.arange(0, 1+0.05, 0.05)  # infection probability

number_of_iterations = 10
R_inf_list = []
for beta in beta_list:
    print('beta =', beta)
    R_inf = SIR_simulation(gamma, beta, number_of_iterations)
    R_inf_list.append(R_inf)

print('R_inf_list =', R_inf_list)
runtime = (time.time() - start_time)/60
print('Runtime:', runtime, 'minutes')

# %% Plot the results
w_dir = os.getcwd()
folder = 'Plots_Ex2'
if not os.path.exists(folder):
    os.makedirs(folder)

plt.figure()
plt.plot(beta_list, R_inf_list, 'o', label=r'$\gamma$ ='+str(gamma), color='blue')
plt.title('Recovered agents vs infection probability')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$R_\infty$')
plt.legend(frameon=False)
savename = 'gamma_'+str(gamma)+'.png'
plt.savefig(os.path.join(w_dir, folder, savename), dpi=300)

plt.show()

# %% b)
import os
import numpy as np
import matplotlib.pyplot as plt
from agents import Agent
import time as time
from SIR_model import SIR_simulation

start_time = time.time()

# Define parameters
gamma_list = [0.01, 0.02]  # recovery probability
beta_list = np.arange(0, 1+0.05, 0.05)  # infection probability
number_of_iterations = 10

R_inf_array = np.zeros((len(gamma_list), len(beta_list)))
for gamma in gamma_list:
    print('gamma =', gamma)
    R_inf_list = []
    for beta in beta_list:
        print('beta =', beta)
        R_inf = SIR_simulation(gamma, beta, number_of_iterations)
        R_inf_list.append(R_inf)
    R_inf_array[gamma_list.index(gamma), :] = R_inf_list

print('R_inf_array =', R_inf_array)
runtime = (time.time() - start_time)/60
print('Runtime:', runtime, 'minutes')

# %% Plot the results
w_dir = os.getcwd()
folder = 'Plots_Ex2'
if not os.path.exists(folder):
    os.makedirs(folder)

plt.figure()
plt.plot(np.divide(beta_list, gamma_list[0]), R_inf_array[0, :], 'o', label=r'$\gamma$ ='+str(gamma_list[0]), color='blue')
plt.plot(np.divide(beta_list, gamma_list[1]), R_inf_array[1, :], 'o', label=r'$\gamma$ ='+str(gamma_list[1]), color='green')
plt.xlabel(r'$\beta/\gamma$')
plt.ylabel(r'$R_\infty$')
plt.title(r'Recovered agents vs $\beta/\gamma$')
plt.legend(frameon=False)

savename = 'beta_over_gamma.png'
plt.savefig(os.path.join(w_dir, folder, savename), dpi=300)
plt.show()

plt.figure()
plt.plot(beta_list, R_inf_array[0,:], 'o', label=r'$\gamma$ ='+str(gamma_list[0]), color='blue')
plt.plot(beta_list, R_inf_array[1,:], 'o', label=r'$\gamma$ ='+str(gamma_list[1]), color='green')
plt.title('Recovered agents vs infection probability')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$R_\infty$')
plt.legend(frameon=False)
savename = 'recovered_agents_diff_gamma.png'
plt.savefig(os.path.join(w_dir, folder, savename), dpi=300)

plt.show()

# %% c) Plot heatmap
import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
from agents import Agent
import time as time
from SIR_model import SIR_simulation

start_time = time.time()

# Define parameters
number_of_iterations = 1
map_width = 10
map_height = 40
# generate lists of x and y coordinates 
beta_list_temp = list(range(map_width))
beta_list = [(x+1)/map_width for x in beta_list_temp]
gamma_list_temp = list(range(map_height))
beta_over_gamma_list = [(x+1)*2 for x in gamma_list_temp]

# generate the gamma/beta grid
gamma_matrix = np.zeros((map_height, map_width))
for i in range(map_height):
    for j in range(map_width):
        # gamma = beta/beta_over_gamma
        gamma_matrix[i, j] = beta_list[j]/beta_over_gamma_list[i]

# generate the R_inf grid
R_inf_matrix = np.zeros((map_height, map_width))
for i in range(map_height):
    for j in range(map_width):
        gamma = gamma_matrix[i][j]
        beta = beta_list[j]
        R_inf = SIR_simulation(gamma, beta, number_of_iterations)
        R_inf_matrix[i, j] = R_inf

print('gamma_over_beta_matrix =', gamma_over_beta_matrix)
print('R_inf_matrix =', R_inf_matrix)

runtime = (time.time() - start_time)/60
print('Runtime:', runtime, 'minutes')

# %% Plot the results
from copy import deepcopy
import os

w_dir = os.getcwd()
folder = 'Plots_Ex2'
if not os.path.exists(folder):
    os.makedirs(folder)

beta_over_gamma_list_copy = deepcopy(gamma_over_beta_list)
R_inf_matrix_copy = deepcopy(R_inf_matrix)

# create a dataframe from the matrix
df = pd.DataFrame(np.flipud(R_inf_matrix_copy), index=reversed(beta_over_gamma_list_copy), columns=beta_list)
plt.figure()
sns.heatmap(df, cmap='autumn_r', linewidths=0.5, annot=False)
plt.title(r'$R_\infty$')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\beta/\gamma$')
plt.yticks(rotation=0, fontsize=8)
plt.xticks(fontsize=8)
plt.savefig(os.path.join(w_dir, folder, 'R_inf_heatmap.png'), dpi=300)
plt.show()
