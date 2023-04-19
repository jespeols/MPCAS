import matplotlib.pyplot as plt
import time
from forest_fires_functions import forest_fires_simulator

start_time = time.time()

# Define parameters
N = 256      # lattice size
p = 0.01    # growth probability
f = 0.2     # lightning strike probability
max_iter = int(1e4)

fire_sizes = forest_fires_simulator(N, p, f, 'max_iter', max_iter)

fig, (ax1, ax2) = plt.subplots(2,1)

bins = (0, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 25000, 30000, 35000, 40000)
ax1.hist(fire_sizes, edgecolor='white', bins=bins)
ax1.set_ylabel("count", fontsize=8)
ax1.set_xlabel("number of trees burned, $\cdot\:10^3$", fontsize=8)
ax1.set_xticks(bins)
new_labels = [int(s/1000) for s in ax1.get_xticks()]
ax1.set_xticklabels(new_labels, fontsize=5)

bins = range(0, 1050, 50)
ax2.hist(fire_sizes, edgecolor='white', bins=bins)
ax2.set_xlabel("number of trees burned", fontsize=8)
ax2.set_ylabel("count", fontsize=8)
ax2.set_xticks(bins)
ax2.set_xticklabels(ax2.get_xticks(), fontsize=5, rotation=90)

fig.tight_layout()
fig.savefig("histograms.png", dpi=300)

plt.show()

stop_time = time.time()
runtime = stop_time - start_time
print('Runtime: ' + str(runtime/60) + ' min')