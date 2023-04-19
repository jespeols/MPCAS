import os
import numpy as np
import matplotlib.pyplot as plt
from agents import Agent
import time as time


def SIR_simulation(gamma, beta, number_of_iterations):    

    # hardcoded parameters
    N = 100       # lattice size
    n = 1000      # number of agents
    d = 0.8       # diffusion probability
    fraction_infected = 0.01  # fraction of agents that are initially infected

    R_inf_list = []
    for iter in range(number_of_iterations):
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

        # print('Simulation finished after', time_step, 'time steps')

        R_inf = recovered_hist[-1]  # number of recovered agents
        R_inf_list.append(R_inf)

    average_R_inf = np.mean(R_inf_list)
    print('average_R_inf =', average_R_inf)
    return average_R_inf

def SIRD_simulation(gamma, beta, mu, number_of_iterations):    

    # hardcoded parameters
    N = 100       # lattice size
    n = 1000      # number of agents
    d = 0.8       # diffusion probability
    fraction_infected = 0.01  # fraction of agents that are initially infected

    D_inf_list = []
    for iter in range(number_of_iterations):
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
        dead_hist = [0]
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

                # check for death or recovery
                if np.random.rand() < mu:       # death
                    agent.set_state(3)
                elif np.random.rand() < gamma:  # recovery
                    agent.set_state(2)

            # Update counters
            healthy_counter = 0
            infected_counter = 0
            recovered_counter = 0
            dead_counter = 0
            for agent in agents:
                if agent.state == 1:
                    infected_counter += 1
                elif agent.state == 2:
                    recovered_counter += 1
                elif agent.state == 3:
                    dead_counter += 1
                else:
                    healthy_counter += 1
            infected_hist.append(infected_counter)
            healthy_hist.append(healthy_counter)
            recovered_hist.append(recovered_counter)
            dead_hist.append(dead_counter)

        # print('Simulation finished after', time_step, 'time steps')

        D_inf = dead_hist[-1]  # number of dead agents at the end
        D_inf_list.append(D_inf)

    average_D_inf = np.mean(D_inf_list)
    print('average_D_inf =', average_D_inf)
    return average_D_inf

def SIRI_simulation(gamma, beta, alpha, maxIter):    

    # hardcoded parameters
    N = 100       # lattice size
    n = 1000      # number of agents
    d = 0.8       # diffusion probability
    fraction_infected = 0.01  # fraction of agents that are initially infected

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

    while infected_counter > 0 and time_step < maxIter:
        time_step += 1
        if time_step % 250 == 0:
            print('Time step:', time_step)
            print('Infected:', infected_counter)
            print('Healthy:', n-infected_counter)
            print('Recovered:', recovered_counter)
            print('--------------------------------')

        # Random walk
        for agent in agents:
            if np.random.rand() < d:
                agent.random_walk(agent.x, agent.y)
        
        # Check for lost immunity
        recovered_agents = [agent for agent in agents if agent.get_state() == 2]
        for agent in recovered_agents:
            if np.random.rand() < alpha:
                agent.set_state(0)

        # Check for spread of infection
        infected_agents = [agent for agent in agents if agent.get_state() == 1]
        healthy_agents = [agent for agent in agents if agent.get_state() == 0]
        for agent in infected_agents:
            for other_agent in healthy_agents:
                if agent.get_position() == other_agent.get_position():
                    if np.random.rand() < beta:     # infection
                        other_agent.set_state(1)

            # check for recovery
            if np.random.rand() < gamma:  # recovery
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

    return infected_hist, healthy_hist

def SIRI_simulation_2(gamma, beta, alpha, fraction_infected, maxIter):    

    # hardcoded parameters
    N = 100       # lattice size
    n = 1000      # number of agents
    d = 0.8       # diffusion probability

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

    while infected_counter > 0 and time_step < maxIter:
        time_step += 1
        if time_step % 250 == 0:
            print('Time step:', time_step)
            print('Infected:', infected_counter)
            print('Healthy:', n-infected_counter)
            print('Recovered:', recovered_counter)
            print('--------------------------------')

        # Random walk
        for agent in agents:
            if np.random.rand() < d:
                agent.random_walk(agent.x, agent.y)
        
        # Check for lost immunity
        recovered_agents = [agent for agent in agents if agent.get_state() == 2]
        for agent in recovered_agents:
            if np.random.rand() < alpha:
                agent.set_state(0)

        # Check for spread of infection
        infected_agents = [agent for agent in agents if agent.get_state() == 1]
        healthy_agents = [agent for agent in agents if agent.get_state() == 0]
        for agent in infected_agents:
            for other_agent in healthy_agents:
                if agent.get_position() == other_agent.get_position():
                    if np.random.rand() < beta:     # infection
                        other_agent.set_state(1)

            # check for recovery
            if np.random.rand() < gamma:  # recovery
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

    return infected_hist, healthy_hist