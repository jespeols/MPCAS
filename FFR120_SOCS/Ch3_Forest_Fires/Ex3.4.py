import matplotlib.pyplot as plt
import time
from forest_fires_simulator import *

start_time = time.time()

p = 0.01
f = 0.2

N1 = 16
fire_sizes_1 = forest_fires_simulator(N1, p, f, 'max_fire_count', 5e3)
rel_fire_sizes_sorted_1 = get_sorted_rel_fire_sizes(N1, fire_sizes_1)
C1 = get_C_vector(fire_sizes_1)

N2 = 256
fire_sizes_2 = forest_fires_simulator(N2, p, f, 'max_fire_count', 4.5e3)
rel_fire_sizes_sorted_2 = get_sorted_rel_fire_sizes(N2, fire_sizes_2)
C2 = get_C_vector(fire_sizes_2)

stop_time = time.time()
runtime = stop_time - start_time
print('Runtime: ' + str(runtime/60) + ' min')

# %% Generate plots

fig, ax = plt.subplots(1,2)

ax[0].loglog(rel_fire_sizes_sorted_1, C1, 'bo')
ax[0].set(xlabel='n/N$^2$', ylabel='C(n)')
ax[0].set_title('N = ' + str(N1))
ax[0].set_xticks([10**(-2), 10**(-1), 10**0])

ax[1].loglog(rel_fire_sizes_sorted_2, C2, 'bo')
ax[1].set(xlabel='n/N$^2$', ylabel='C(n)')
ax[1].set_title('N = ' + str(N2))
ax[1].set_xticks([10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**0])


fig.tight_layout()
fig.savefig("cCDF_N" + str(N1) + '_' + "cCDF_N" + str(N2) + ".png", dpi=300)
plt.show()
