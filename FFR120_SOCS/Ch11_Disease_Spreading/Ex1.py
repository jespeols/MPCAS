# %% a)
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from agents import Agent
import time as time

# Define parameters
N = 100       # lattice size
n = 1000      # number of agents
d = 0.8       # diffusion probability
beta = 0.75    # infection probability
gamma = 0.01  # recovery probability
fraction_infected = 0.01  # fraction of agents that are initially infected

window_res = 600
# Initialize animation window & canvas
animation_window = tk.Tk()
animation_window.title("0 infected")
animation_window.geometry(str(window_res)+'x'+str(window_res))
canvas = tk.Canvas(animation_window)
canvas.configure(bg="white")
canvas.pack(expand=True, fill="both")

# Initialize lattice
S = np.zeros((N,N))     # State of cells: 0=healthy, 1=infected, 2=recovered

beta_list = [0.19, 0.2, 0.21]
w_dir = os.getcwd()
folder = 'Plots_Ex1/'
if not os.path.exists(folder):
    os.makedirs(folder)

# Initialize agents
agents = [Agent(np.random.randint(N), np.random.randint(N), N, 0) for _ in range(n)]
# choose 1% random agents to be infected
initial_infected = np.random.choice(range(n), int((fraction_infected*n)), replace=False)
for i in initial_infected:
    agents[i].set_state(1)
infected_counter = len(initial_infected)

time_step = 0
infected_hist = [infected_counter]
healthy_hist = [n-infected_counter]
recovered_hist = [0]

while infected_counter > 0:
    time_step += 1
    if time_step % 500 == 0:
        print('Time step:', time_step)
        print('Infected:', infected_counter)
        print('Healthy:', n-infected_counter)
        print('Recovered:', recovered_counter)
        print('--------------------------------')

    # Random walk
    for agent in agents:
        if np.random.rand() < d:
            agent.random_walk(agent.x, agent.y)
    
    # Check for spread of infection
    infected_agents = [agent for agent in agents if agent.get_state() == 1]
    healthy_agents = [agent for agent in agents if agent.get_state() == 0]
    for agent in infected_agents:
        for other_agent in healthy_agents:
            if agent.get_position() == other_agent.get_position():
                if np.random.rand() < beta:     # infection
                    other_agent.set_state(1)

        # check for recovery
        if np.random.rand() < gamma:
            agent.set_state(2)

    # Update counters
    healthy_counter = 0
    infected_counter = 0
    recovered_counter = 0
    for agent in agents:
        if agent.state == 1:
            infected_counter += 1
        elif agent.state == 2:
            recovered_counter += 1
        else:
            healthy_counter += 1
    infected_hist.append(infected_counter)
    healthy_hist.append(healthy_counter)
    recovered_hist.append(recovered_counter)
    
    # draw agents
    canvas.delete("all")
    for agent in agents:
        if agent.state == 0:
            canvas.create_rectangle(agent.x*window_res/N, agent.y*window_res/N, (agent.x+1)*window_res/N, (agent.y+1)*window_res/N, fill="green")
        elif agent.state == 1:
            canvas.create_rectangle(agent.x*window_res/N, agent.y*window_res/N, (agent.x+1)*window_res/N, (agent.y+1)*window_res/N, fill="red")
        else:
            canvas.create_rectangle(agent.x*window_res/N, agent.y*window_res/N, (agent.x+1)*window_res/N, (agent.y+1)*window_res/N, fill="blue")

    animation_window.title('t =' + str(time_step) + ', healthy: ' + str(healthy_counter) + ', infected: ' + str(infected_counter) + ', recovered: ' +str(recovered_counter))
    animation_window.update()
    time.sleep(0.05)

# close animation window
animation_window.destroy()
print('Simulation finished after', time_step, 'time steps')

# plot results
plt.figure()
plt.plot(infected_hist, label='infected', color='red')
plt.plot(healthy_hist, label='healthy', color='green')
plt.plot(recovered_hist, label='recovered', color='blue')
plt.axhline(y=n/2, color='black', linestyle='--')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), frameon=False, ncol=3)
plt.xlabel('time steps')
plt.ylabel('number of agents')
plt.title(label=r'state of agents over time, $\beta$ = ' + str(beta) + ', $\gamma$ = ' + str(gamma))
plt.ylim([0, n])
plt.xlim([0, time_step*1.011])

savename = 'beta_' + str(beta) +'_' + str(iter)
plt.savefig(w_dir+'/'+folder+'/'+savename+'.png', dpi=300)
plt.show()

animation_window.mainloop() 

# %% b)-c)
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from agents import Agent
import time as time

# Define parameters
N = 100       # lattice size
n = 1000      # number of agents
d = 0.8       # diffusion probability
gamma = 0.02  # recovery probability
fraction_infected = 0.01  # fraction of agents that are initially infected
beta_list = [0.6, 0.7, 0.8]  # infection probability

w_dir = os.getcwd()
folder = 'Plots_Ex1/gamma_'+str(gamma)
if not os.path.exists(folder):
    os.makedirs(folder)

for beta in beta_list:
    print('beta =', beta)
    for iter in range(4):
        # Initialize agents
        agents = [Agent(np.random.randint(N), np.random.randint(N), N, 0) for _ in range(n)]
        # choose 1% random agents to be infected
        initial_infected = np.random.choice(range(n), int((fraction_infected*n)), replace=False)
        for i in initial_infected:
            agents[i].set_state(1)
        infected_counter = len(initial_infected)

        time_step = 0
        infected_hist = [infected_counter]
        healthy_hist = [n-infected_counter]
        recovered_hist = [0]

        while infected_counter > 0:
            time_step += 1

            # Random walk
            for agent in agents:
                if np.random.rand() < d:
                    agent.random_walk(agent.x, agent.y)
            
            # Check for spread of infection
            infected_agents = [agent for agent in agents if agent.get_state() == 1]
            healthy_agents = [agent for agent in agents if agent.get_state() == 0]
            for agent in infected_agents:
                for other_agent in healthy_agents:
                    if agent.get_position() == other_agent.get_position():
                        if np.random.rand() < beta:     # infection
                            other_agent.set_state(1)

                # check for recovery
                if np.random.rand() < gamma:
                    agent.set_state(2)

            # Update counters
            healthy_counter = 0
            infected_counter = 0
            recovered_counter = 0
            for agent in agents:
                if agent.state == 1:
                    infected_counter += 1
                elif agent.state == 2:
                    recovered_counter += 1
                else:
                    healthy_counter += 1
            infected_hist.append(infected_counter)
            healthy_hist.append(healthy_counter)
            recovered_hist.append(recovered_counter)
            
        print('Simulation finished after', time_step, 'time steps')

        # plot results
        plt.figure()
        plt.plot(infected_hist, label='infected', color='red')
        plt.plot(healthy_hist, label='healthy', color='green')
        plt.plot(recovered_hist, label='recovered', color='blue')
        plt.axhline(y=n/2, color='black', linestyle='--')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), frameon=False, ncol=3)
        plt.xlabel('time steps')
        plt.ylabel('number of agents')
        plt.title(label=r'state of agents over time, $\beta$ = ' + str(beta) + ', $\gamma$ = ' + str(gamma))
        plt.ylim([0, n])
        plt.xlim([0, time_step*1.011])

        savename = 'beta_' + str(beta) +'_' + str(iter)
        plt.savefig(w_dir+'/'+folder+'/'+savename+'.png', dpi=300)
        plt.show()

print('Done')