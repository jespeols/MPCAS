# %% 
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import tkinter as tk
import time as time
from prisoners_dilemma import get_best_strategy, play_von_Neumann_neighbors

w_dir = os.getcwd()

# Define parameters
T = 0     # Penalty for defection when the other player cooperates
P = 1     # Penalty for when both defect
R = 0.8   # Penalty for when both cooperate
S = 1.5   # Penalty for cooperation when the other player defects
punishments = [T, R, P, S]

N = 7
L = 30               # Size of the lattice
t_max = 201          # Number of time steps
mu = 0.01            # Mutation rate

save_plots = True
if save_plots:
    folder = 'Plots_Ex4/'+str(R)+'/' 
    if not os.path.exists(folder):
        os.makedirs(folder)

# Initialize lattice of strategies
strategies = np.random.randint(0, N+1, (L, L))      # Random strategies in full range

cmap_colors = ['red', 'brown', 'orange', 'yellow', 'blue', 'cyan', 'green', 'limegreen']
cmap = colors.ListedColormap(cmap_colors)

strategy_pop = np.zeros((N+1, t_max))
for t in range(t_max):
    # count strategy population
    for n in range(N+1):
        strategy_pop[n, t] = np.sum(strategies == n)

    if t in [0, 10, 20, 50, 100]:
        plt.figure()
        plt.imshow(strategies, cmap=cmap)
        plt.title('t =' + str(t) + ', R = ' + str(R))
        plt.colorbar()
        savename = 'strategies_t' + str(t) + '_R' + str(R) + '.png'
        plt.savefig(os.path.join(w_dir, folder, savename), dpi=300) if save_plots else None
        plt.show()

    prison_times = np.zeros((L, L))
    # all players play their neighbors
    for i in range(L):
        for j in range(L):
            prison_times[i,j] = play_von_Neumann_neighbors(N, punishments, strategies, i, j)
    # Update strategies
    new_strategies = get_best_strategy(strategies, prison_times)
    strategies = new_strategies

    if mu > 0:
        # Randomly mutate strategies
        r_matrix = np.random.rand(L, L)
        strategies[r_matrix < mu] = np.random.choice([0,N], (L, L))[r_matrix < mu]

# plot final strategies
plt.figure()
plt.imshow(strategies, cmap=cmap)
plt.title('final strategies after t = ' + str(t) + ', R = ' + str(R))
plt.colorbar()
savename = 'strategies_final_R' + str(R) + '.png'
plt.savefig(os.path.join(w_dir, folder, savename), dpi=300) if save_plots else None
plt.show()

# plot strategy population
plt.figure()
for n in range(N+1):
    plt.plot(strategy_pop[n, :], label='strategy ' + str(n), c=cmap_colors[n])
plt.title('Strategy population, R = ' + str(R))
plt.xlabel('t')
plt.ylabel('strategy population, n')
# place legend below plot
# plt.legend([('n = ' + str(n)) for n in range(N+1)], loc='upper center', bbox_to_anchor=(0.5, -0.125), ncols=8)
plt.legend([str(n) for n in range(N+1)], loc='upper center', bbox_to_anchor=(1.07, 1.02), ncols=1)
savename = 'strategy_pop_R' + str(R) + '.png'
plt.savefig(os.path.join(w_dir, folder, savename), dpi=300) if save_plots else None
plt.show()

print('Done')