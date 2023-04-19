import os
import numpy as np
import time
import matplotlib.pyplot as plt
from forest_fires_functions import *

start_time = time.time()

N = 128
p = 0.01
f = 0.05
T = int(3e4)

# create forest
S = np.zeros((N,N))     # State of cells: 0=no tree, 1=tree, 2=burned, 3=burning
    
fire_sizes = []
fire_sizes_rand = []

iteration = 0
run_simulation = True
while run_simulation:
    print("Iteration: " + str(iteration))
    
    # Apply tree growth
    S[(np.random.rand(N,N) < p) & (S == 0)] = 1
    
    lightning_strikes = np.random.rand() < f    # check if lightning strikes
    if lightning_strikes:
        strike_position = np.random.randint(0, N, 2)
        
        tree_struck = S[strike_position[0], strike_position[1]] == 1
        if tree_struck:
            print('Number of fires: ' + str(len(fire_sizes)+1))
            # Generate fire in random forest with same number of trees and record its size
            tree_cells = np.where(S==1)
            number_of_trees = np.shape(tree_cells)[1]
            
            fire_size_rand = simulate_fire_in_random_forest(N, number_of_trees)
            fire_sizes_rand.append(fire_size_rand)
                        
            fire_cluster = []
            S[strike_position[0], strike_position[1]] = 3      # The struck tree is burning
            while sum(sum(S==3)) > 0:    # while the fire is spreading
                burning_trees = np.where(S==3)
                burning_tree_locations = zip(burning_trees[0].tolist(), burning_trees[1].tolist())
                for i,j in burning_tree_locations:   # loop over burning trees
                    # check upward expansion
                    if i == 0 and S[N-1,j] == 1:   # periodic boundary conditions
                        S[N-1,j] = 3
                    elif S[i-1,j] == 1:
                        S[i-1,j] = 3
                    # check downward expansion
                    if i == N-1 and S[0,j] == 1:
                        S[0,j] = 3
                    elif (i+1) < N-1 and S[i+1,j] == 1:
                        S[i+1,j] = 3
                    # check rightward expansion
                    if j == N-1 and S[i,0] == 1:
                        S[i,0] = 3
                    elif (j+1) < N-1 and S[i,j+1] == 1:
                        S[i,j+1] = 3
                    # check leftward expansion
                    if j == 0 and S[i,N-1] == 1:
                        S[i,N-1] = 3
                    elif S[i,j-1] == 1:
                        S[i,j-1] = 3
                    
                    fire_cluster.append([i,j]) if [i,j] not in fire_cluster else fire_cluster
                    S[i,j] = 2      # tree is burned
                
            # Determine size of fire:
            fire_size = len(fire_cluster)
            print('Fire size: ' + str(fire_size))
            fire_sizes.append(fire_size)
    
    S[S==2] = 0     # Burned trees become empty cells
        
    iteration += 1
    if iteration == T:
        run_simulation = False
        
stop_time = time.time()
runtime = stop_time - start_time
print('Runtime: ' + str(runtime/60) + ' min')
    
# %% Plot data

number_of_fires = len(fire_sizes)

rel_fire_sizes_sorted_rand = get_sorted_rel_fire_sizes(N, fire_sizes_rand)
C_rand = get_C_vector(fire_sizes_rand)

rel_fire_sizes_sorted = get_sorted_rel_fire_sizes(N, fire_sizes)
C = get_C_vector(fire_sizes)

plt.loglog(rel_fire_sizes_sorted_rand, C_rand, 'bo', label="random forest")
plt.loglog(rel_fire_sizes_sorted, C, 'ro', label="forest grown with fires")
plt.xlabel('n/N$^2$')
plt.ylabel('C(n)')
plt.title('N = '+str(N)+", p = "+str(p)+", f = "+str(f)+", T = "+str(T)+", "+str(number_of_fires)+" fires")
plt.legend()
plt.xticks([10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**0])
w_dir = os.getcwd()
plot_name = "cCDF_N"+str(N)+"_p"+str(p)+"_f"+str(f)+"_T"+str(T)
plt.savefig(w_dir+"/Plots_Ex3.5/"+plot_name+".png", dpi=300)