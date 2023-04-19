import numpy as np
from itertools import product


def random_power_law_numbers(alpha, k_correlation, sequence_length):
    # Generate sequence of uniform random numbers
    r_list = np.random.rand(sequence_length)
    n_list = []
    
    for r_i in r_list:
        n_i = round((r_i/k_correlation)**(1/(1-alpha)))
        n_list.append(n_i)
        
    return n_list


def get_sorted_rel_fire_sizes(N, fire_sizes):
    fire_sizes_sorted = sorted(fire_sizes)
    rel_fire_sizes_sorted = [s / N**2 for s in fire_sizes_sorted]
    
    return rel_fire_sizes_sorted


def get_C_vector(fire_sizes):
    k = len(fire_sizes)
    C = []
    for i in range(k):
        C.append((k-i)/k)
    
    return C


def forest_fires_simulator(N, p, f, mode, termination_critera):
    # create forest
    S = np.zeros((N,N))     # State of cells: 0=no tree, 1=tree, 2=burned, 3=burning
        
    fire_sizes = []
    
    iteration = 0
    run_simulation = True
    while run_simulation:
        if iteration in range(0, int(1e6), 100):
            print("Iteration: " + str(iteration))
        
        # Apply tree growth
        S[(np.random.rand(N,N) < p) & (S == 0)] = 1
        
        lightning_strikes = np.random.rand() < f    # check if lightning strikes
        if lightning_strikes:
            strike_position = np.random.randint(0, N, 2)
            
            tree_struck = S[strike_position[0], strike_position[1]] == 1
            if tree_struck:
                print('Number of fires: ' + str(len(fire_sizes)+1))
                
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
                # print('Fire size: ' + str(fire_size))
                fire_sizes.append(fire_size)
        
        S[S==2] = 0     # Burned trees become empty cells
        
        if mode == 'max_iter':
            if termination_critera == iteration:
                run_simulation = False
        elif mode == 'max_fire_count':
            if termination_critera == len(fire_sizes):
                run_simulation = False
        else:
            print("enter valid termination mode: 'max_iter' or 'max_fire_count'")
            
        iteration += 1
            
    return fire_sizes


def simulate_fire_in_random_forest(N, number_of_trees):
    S = np.zeros((N,N))     # create random forest
    
    S_indices = list(product(range(0,N), range(0,N)))   # create list of all cell locations
    # choose a random set of cells without replacement
    random_rows = np.random.choice(np.arange(len(S_indices)), number_of_trees, replace=False)
    rand_tree_locations = [S_indices[i] for i in random_rows]
    
    for tree_loc in rand_tree_locations:
        S[tree_loc[0], tree_loc[1]] = 1

    rand_index = np.random.randint(0, len(rand_tree_locations))
    rand_strike_position = rand_tree_locations[rand_index]
    S[rand_strike_position[0], rand_strike_position[1]] = 3      # The struck tree is burning

    fire_cluster = []
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
    print('Fire size (random forest): ' + str(fire_size))
        
    return fire_size
