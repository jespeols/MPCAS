# %% a)
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time as time
from prisoners_dilemma import get_best_strategy, play_von_Neumann_neighbors

start_time = time.time()

w_dir = os.getcwd()

# Define parameters
T = 0     # Penalty for defection when the other player cooperates
P = 1     # Penalty for when both defect

N = 7
L = 30               # Size of the lattice
t_max = 500          # Number of time steps
t_cutoff = 100        # Time step to start counting the population
mu = 0.01            # Mutation rate

save_plots = False
if save_plots:
    folder = 'Plots_Ex5' 
    if not os.path.exists(folder):
        os.makedirs(folder)

R_list = list(np.linspace(0, 1, 11)[1:-1])  # List of R values to test
S_list = list(np.linspace(1, 3, 11)[1:-1])  # ensure T < R < P < S
variance_list = np.zeros((N+1,len(R_list),len(S_list)))  # 3D array to store the variance for each strategy

for R in R_list:
    for S in S_list:
        # Initialize lattice of strategies
        strategies = np.random.randint(0, N+1, (L, L))      # Random strategies in full range

        print('R =', R, ', S =', S)
        punishments = [T, R, P, S]
        strategy_pop = np.zeros((N+1, t_max))
        for t in range(t_max):
            # count strategy population
            for n in range(N+1):
                strategy_pop[n, t] = np.sum(strategies == n)

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

        # Calculate the variance for each strategy
        for n in range(N+1):
            variance_list[n,R_list.index(R),S_list.index(S)] = np.var(strategy_pop[n,t_cutoff:])

runtime = (time.time() - start_time)/60
print('Runtime: ' + str(runtime) + ' minutes')

# %% plot variances
save_plots = True
folder = 'Plots_Ex5'
if not os.path.exists(folder) and save_plots:
    os.makedirs(folder)

for n in range(N+1):
    plt.figure()
    plt.imshow(np.flipud(variance_list[n,:,:]))
    plt.colorbar()
    plt.xticks(np.arange(len(S_list)), [round(val,1) for val in S_list])
    plt.yticks(np.arange(len(R_list)), [round(val,2) for val in R_list])
    plt.xlabel('S')
    plt.ylabel('R')
    plt.title('Variance of strategy n = ' + str(n))
    savename = 'variance_n' + str(n) + '.png'
    plt.savefig(os.path.join(w_dir, folder, savename), dpi=300) if save_plots else None
    plt.show()

# plot sum of variances
variance_sum = np.sum(variance_list, axis=0)
# plot as heatmap
plt.imshow(np.flipud(variance_sum))
plt.colorbar()
plt.xticks(np.arange(len(S_list)), [round(val,2) for val in S_list])
plt.yticks(np.arange(len(R_list)), [round(val,2) for val in R_list])
plt.xlabel('S')
plt.ylabel('R')
plt.title('Sum of variances of all strategies')
savename = 'heatmap_variance_sum.png'
plt.savefig(os.path.join(w_dir, folder, savename), dpi=300) if save_plots else None
plt.show()

# # plot phase diagram
variance_threshold = 20e3
phase_diagram = np.zeros((len(R_list), len(S_list)))
phase_diagram[variance_sum > variance_threshold] = 1

plt.figure()
plt.imshow(np.flipud(phase_diagram), cmap=colors.ListedColormap(['red', 'green']))
plt.colorbar()
plt.xticks(np.arange(len(S_list)), [round(val,2) for val in S_list])
plt.yticks(np.arange(len(R_list)), [round(val,2) for val in R_list])
plt.xlabel('S')
plt.ylabel('R')
plt.title('Phase diagram of variance, threshold = ' + str(int(variance_threshold)))
savename = 'phase_diagram_threshold='+str(variance_threshold)+'.png'
plt.savefig(os.path.join(w_dir, folder, savename), dpi=300) if save_plots else None
plt.show()
