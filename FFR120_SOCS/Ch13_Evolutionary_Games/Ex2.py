# %% a)-b)
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import time as time
from prisoners_dilemma import get_best_strategy, play_von_Neumann_neighbors

# Define parameters
T = 0     # Penalty for defection when the other player cooperates
P = 1     # Penalty for when both defect
R = 0.9   # Penalty for when both cooperate
S = 1.5   # Penalty for cooperation when the other player defects
punishments = [T, R, P, S]

N = 7
n_list = [0, N]
L = 30                  # Size of the lattice
t_max = 21              # Number of time steps
mu = 0                  # Mutation rate

window_res = 600
# Initialize animation window & canvas
animation_window = tk.Tk()
animation_window.title("")
animation_window.geometry(str(window_res)+'x'+str(window_res))
canvas = tk.Canvas(animation_window)
canvas.configure(bg="white")
canvas.pack(expand=True, fill="both")

# Initialize lattice of strategies
# strategies = np.random.randint(0, N, (L, L))      # Random strategies in full range
# strategies = np.random.choice([0, N], (L, L))     # Choose from two strategies
strategies = np.ones((L, L))*N                      # All start as cooperators
strategies[L//2-9, L//2+9] = 0
strategies[L//2-3, L//2+3] = 0   
strategies[L//2+3, L//2-3] = 0                       
strategies[L//2+9, L//2-9] = 0                      

for t in range(t_max):
    # count defectors
    n_defectors = np.sum(strategies == 0)

    canvas.delete("all")
    # draw players
    for i in range(L):
        for j in range(L):
            if strategies[i, j] == 0:
                canvas.create_rectangle(i*window_res/L, j*window_res/L, (i+1)*window_res/L, (j+1)*window_res/L, fill="red")
            else:
                canvas.create_rectangle(i*window_res/L, j*window_res/L, (i+1)*window_res/L, (j+1)*window_res/L, fill="green")

    animation_window.title('t =' + str(t) + ', ' + str(n_defectors) + ' defectors')
    animation_window.update()

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
        strategies[r_matrix < mu] = np.random.randint(0, N, (L, L))[r_matrix < mu]
    if t == 0:
        time.sleep(1.5)
    else:
        time.sleep(0.1)

animation_window.mainloop() 
# %% d)
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import time as time
from prisoners_dilemma import get_best_strategy, play_von_Neumann_neighbors

# Define parameters
T = 0     # Penalty for defection when the other player cooperates
P = 1     # Penalty for when both defect
R = 0.84   # Penalty for when both cooperate
S = 1.5   # Penalty for cooperation when the other player defects
punishments = [T, R, P, S]

N = 7
n_list = [0, N]
L = 30                  # Size of the lattice
t_max = 21              # Number of time steps
mu = 0                  # Mutation rate

window_res = 600
# Initialize animation window & canvas
animation_window = tk.Tk()
animation_window.title("")
animation_window.geometry(str(window_res)+'x'+str(window_res))
canvas = tk.Canvas(animation_window)
canvas.configure(bg="white")
canvas.pack(expand=True, fill="both")

# Initialize lattice of strategies
# strategies = np.random.randint(0, N, (L, L))      # Random strategies in full range
# strategies = np.random.choice([0, N], (L, L))     # Choose from two strategies
strategies = np.zeros((L, L))                        # All start as defectors
strategies[L//2-2:L//2+2, L//2-2:L//2+2] = N 

for t in range(t_max):
    # count defectors
    n_defectors = np.sum(strategies == 0)

    canvas.delete("all")
    # draw players
    for i in range(L):
        for j in range(L):
            if strategies[i, j] == 0:
                canvas.create_rectangle(i*window_res/L, j*window_res/L, (i+1)*window_res/L, (j+1)*window_res/L, fill="red")
            else:
                canvas.create_rectangle(i*window_res/L, j*window_res/L, (i+1)*window_res/L, (j+1)*window_res/L, fill="green")

    animation_window.title('t =' + str(t) + ', ' + str(n_defectors) + ' defectors')
    animation_window.update()

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
        strategies[r_matrix < mu] = np.random.randint(0, N, (L, L))[r_matrix < mu]
    if t == 0:
        time.sleep(1.5)

animation_window.mainloop() 