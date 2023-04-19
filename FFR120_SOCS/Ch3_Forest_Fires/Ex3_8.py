# %% Generate data
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from forest_fires_functions import *

start_time = time.time()

p = 0.01
f = 0.3

n_realizations = 3
N_list = [16, 32, 64, 128, 256, 512]
x1_reg_list = [5e-2, 1.4e-2, 3e-3, 7e-4, 1.5e-4, 4e-5]
x2_reg_list = [0.2, 0.15, 0.1, 0.125, 0.1, 0.125]
T_list = [5e4, 5e4, 3e4, 8e3, 7e3, 5e3]


alpha_list = np.zeros((n_realizations, len(N_list)))
n_fires = np.zeros((n_realizations, len(N_list)))

for idx, N in enumerate(N_list):
    T = T_list[idx]
    alpha_list_current = []
    for iteration in range(n_realizations):
        print("iteration " + str(iteration) + " for N = " + str(N))
        # run sim and estimate alpha
        fire_sizes = forest_fires_simulator(N, p, f, 'max_iter', T)
        
        number_of_fires = len(fire_sizes)
        n_fires[iteration, idx] = number_of_fires

        rel_fire_sizes_sorted = get_sorted_rel_fire_sizes(N, fire_sizes)
        C = get_C_vector(fire_sizes)

        x = rel_fire_sizes_sorted.copy()
        x1_reg = x1_reg_list[idx]
        x2_reg = x2_reg_list[idx]
        x_trunc = [value for value in x if value > x1_reg and value < x2_reg]  # N=128, p=0.01, f=0.2

        trunc_indices = [x.index(value) for value in x if value in x_trunc]
        C_trunc = [C[idx] for idx in trunc_indices]

        x_trunc_log = np.log(x_trunc)
        C_trunc_log = np.log(C_trunc)

        lin_model = np.poly1d(np.polyfit(x_trunc_log, C_trunc_log, 1))
        alpha = 1 - lin_model[1]
        
        alpha_list[iteration, idx] = alpha
    
stop_time = time.time()
runtime = stop_time - start_time
print('Runtime: ' + str(runtime/(60*60)) + ' h')
  
# %% Plot average alpha vs 1/N to find alpha at N --> inf

average_alpha_list = []
for j in range(len(N_list)):
    alpha_list_N = alpha_list[:, j]
    average_alpha = np.mean(alpha_list_N)

    average_alpha_list.append(average_alpha)

print(average_alpha_list)

N_list_inverse = [1/N for N in N_list]

model = np.poly1d((np.polyfit(N_list_inverse, average_alpha_list, 1)))
x_plot = np.linspace(0, N_list_inverse[0]*1.1)
alpha_inf = model[0]
print("\u03B1_inf = "+str(alpha_inf))

plt.plot(x_plot, model(x_plot), '--', color="k", label="fit, \u03B1_inf = "+str(round(alpha_inf,3)))
plt.plot(N_list_inverse, average_alpha_list, 'ro', label="average \u03B1")
xticks_list = [r"$\frac{1}{"+str(N)+"}$" for N in N_list]
#xticks_list[-1] = ''
#xticks_list[-3] = ''
plt.xticks(N_list_inverse, labels=xticks_list)
plt.yticks([1.1, 1.2, 1.3, 1.4, 1.5])
plt.xlim([0, 1/16*1.1])
plt.legend()
w_dir = os.getcwd()
plt.savefig(w_dir+"/Plots_Ex3.8/alpha_inf_estimate.png", dpi=300)

plt.show()
# %%
