from forest_fires_functions import *
import numpy as np
import matplotlib.pyplot as plt

alpha = 1.15
sequence_length = int(5e3)
k = 1
N = 256

r_list = np.random.rand(sequence_length)
n_list = []

for r_i in r_list:
    n_i = round((r_i/k)**(1/(1-alpha)))
    n_list.append(n_i)
    
n_list_sorted = sorted(n_list)
rel_n_list_sorted = [s / N**2 for s in n_list_sorted]
C = get_C_vector(rel_n_list_sorted)

print(n_list)
print(n_list_sorted)
print(rel_n_list_sorted)

plt.loglog(n_list_sorted, C, 'o')
plt.xlim([0.8, 1e6])
plt.ylim([5e-2, 1.2])
plt.xlabel('n')
plt.ylabel('C(n)')

plt.show()