# %% a)
import os
import numpy as np
import matplotlib.pyplot as plt
from prisoners_dilemma import play_prisoners_dilemma

# Define parameters
N = 10    # Number of rounds
T = 0     # Penalty for defection when the other player cooperates
R = 0.5   # Penalty for when both cooperate
P = 1     # Penalty for when both defect
S = 1.5   # Penalty for cooperation when the other player defects 
punishments = [T, R, P, S]
m = 5            # strategy of player 1
n_list = list(range(N+1))     # strategy of player 2

# initialize punishment histories
p_hist = []
for n in n_list:
    total_p = play_prisoners_dilemma(N, punishments, n, m)
    p_hist.append(total_p)

# Plot
w_dir = os.getcwd()
folder = 'Plots_Ex1'
if not os.path.exists(folder):
    os.makedirs(folder)

plt.figure()
best_n = n_list[p_hist.index(min(p_hist))]
best_p = min(p_hist)
plt.plot(n_list, p_hist, 'o', label='Player 1')
plt.plot(best_n, best_p, 'o', c='limegreen', label='Best strategy')
plt.vlines(m, min(p_hist)*0.95, max(p_hist), 'black', '--', label='Player 2 strategy')
plt.title("total punishment for player 1 with m = " + str(m) + ', N = ' + str(N))
plt.xlabel("Player 1 strategy, n")
plt.ylabel("total punishment (yrs)")
plt.xticks(range(N+1))
plt.legend()
savename = 'player1_punishments'+ '_N' + str(N) + '.png'
plt.savefig(os.path.join(w_dir, folder, savename), dpi=300)

plt.show()

# %% b)
import os
import numpy as np
import matplotlib.pyplot as plt
from prisoners_dilemma import play_prisoners_dilemma
import pandas as pd
import seaborn as sns

# Define parameters
N = 10    # Number of rounds
T = 0     # Penalty for defection when the other player cooperates
R = 0.5   # Penalty for when both cooperate
P = 1     # Penalty for when both defect
S = 1.5   # Penalty for cooperation when the other player defects 
punishments = [T, R, P, S]

m_list = list(range(N+1))     # strategy of player 1
n_list = list(range(N+1))     # strategy of player 2

# initialize punishment histories
p_hist = np.zeros((len(n_list),len(m_list)))
for n in n_list:
    for m in m_list:
        total_p = play_prisoners_dilemma(N, punishments, n, m)
        p_hist[n,m] = total_p

# Plot
w_dir = os.getcwd()
folder = 'Plots_Ex1'
if not os.path.exists(folder):
    os.makedirs(folder)

x = m_list
y = n_list
z = p_hist
# create a dataframe with the data
df = pd.DataFrame(np.flipud(z), index=reversed(y), columns=x)
# plot the heatmap
plt.figure()
sns.heatmap(df, cmap='magma_r', linewidths=0.01, annot=False, cbar_kws={'label': 'total punishment (yrs)'})
x_plot = [1, 11.5]
y_plot = [11, 0.5]
plt.plot(x_plot, y_plot, 'r--')
plt.title('best strategy across the space, N = ' + str(N))
plt.xlabel("Player 2 strategy, m")
plt.ylabel("Player 1 strategy, n")
plt.yticks(rotation=0)
savename = 'strat_map'+ '_N' + str(N) + '.png'
plt.savefig(os.path.join(w_dir, folder, savename), dpi=300)

plt.show()

# %% c)
import os
import numpy as np
import matplotlib.pyplot as plt
from prisoners_dilemma import play_prisoners_dilemma_sort
import pandas as pd
import seaborn as sns

# Define parameters
N = 10    # Number of rounds
T = 0     # Penalty for defection when the other player cooperates
R = 1.5   # Penalty for when both cooperate
P = 1     # Penalty for when both defect
S = 2     # Penalty for cooperation when the other player defects 
punishments = [T, R, P, S]

m_list = list(range(N+1))     # strategy of player 1
n_list = list(range(N+1))     # strategy of player 2

# c1, c2 are choices of player 1 and 2, respectively
# True = cooperate, False = defect

# initialize punishment histories
p_hist = np.zeros((len(n_list),len(m_list)))
for n in n_list:
    for m in m_list:
        total_p = play_prisoners_dilemma_sort(N, punishments, n, m, False)
        p_hist[n,m] = total_p

# Plot

w_dir = os.getcwd()
folder = 'Plots_Ex1/vary_R_S'
if not os.path.exists(folder):
    os.makedirs(folder)

x = m_list
y = n_list
z = p_hist
# create a dataframe with the data
df = pd.DataFrame(np.flipud(z), index=reversed(y), columns=x)
# plot the heatmap
plt.figure()
sns.heatmap(df, cmap='magma_r', linewidths=0.01, annot=True, cbar_kws={'label': 'total punishment (yrs)'})
x_plot = [1, 11.5]
y_plot = [11, 0.5]
plt.plot(x_plot, y_plot, 'r--')
plt.title('best strategy across the space, S = ' + str(S) + ', R = ' + str(R))
plt.xlabel("Player 2 strategy, m")
plt.ylabel("Player 1 strategy, n")
savename = 'strat_map'+ '_R' + str(R) + '_S' + str(S) + '.png'
plt.savefig(os.path.join(w_dir, folder, savename), dpi=300)

plt.show()