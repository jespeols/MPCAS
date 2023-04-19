import os
import numpy as np
import time
import matplotlib.pyplot as plt
from forest_fires_functions import *

start_time = time.time()

N = 256
p = 0.01
f = 0.4
T = int(1.5e4)
        
fire_sizes = forest_fires_simulator(N, p, f, 'max_iter', T)
        
stop_time = time.time()
runtime = stop_time - start_time
print('Runtime: ' + str(runtime/60) + ' min')
    
# %% Plot data & fit line

number_of_fires = len(fire_sizes)

rel_fire_sizes_sorted = get_sorted_rel_fire_sizes(N, fire_sizes)
C = get_C_vector(fire_sizes)

x = rel_fire_sizes_sorted.copy()
x1_reg = 1e-4
x2_reg = 0.3
x_trunc = [value for value in x if value > x1_reg and value < x2_reg]

trunc_indices = [x.index(value) for value in x if value in x_trunc]
C_trunc = [C[idx] for idx in trunc_indices]

x_trunc_log = np.log(x_trunc)
C_trunc_log = np.log(C_trunc)

lin_model = np.poly1d(np.polyfit(x_trunc_log, C_trunc_log, 1))
alpha = 1 - lin_model[1]
print('\u03B1 = ' + str(alpha))
x_plot = np.linspace(np.log(x)[0], np.log(x)[-1])

plt.loglog(x, C, 'o', color='orangered', markersize=3, label="forest grown with fires")
plt.loglog(np.exp(x_plot), np.exp(lin_model(x_plot)), '--', color="gold", label="fit, \u03B1 = "+str(round(alpha,3)))
plt.loglog((x1_reg, x1_reg), (min(C), max(C)*1.2), 'k--', linewidth=1, label="regression interval")
plt.loglog((x2_reg, x2_reg), (min(C), max(C)*1.2), 'k--', linewidth=1)

plt.xlabel('n/N$^2$')
plt.ylabel('C(n)')
plt.title('N = '+str(N)+", p = "+str(p)+", f = "+str(f)+", T = "+str(T)+", "+str(number_of_fires)+" fires")
plt.legend(loc='best', fontsize=6)
plt.xticks([10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**0])
w_dir = os.getcwd()
plot_name = "cCDF_N"+str(N)+"_p"+str(p)+"_f"+str(f)+"_T"+str(T)
plt.savefig(w_dir+"/Plots_Ex3.6/"+plot_name+".png", dpi=300)