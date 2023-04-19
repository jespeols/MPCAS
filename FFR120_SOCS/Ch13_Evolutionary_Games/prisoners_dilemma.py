import numpy as np
import matplotlib.pyplot as plt

def get_punishments(c1, c2, punishments:list, sort:bool=True):
    if sort:
        punishments_sorted = sorted(punishments)
        T = punishments_sorted[0]
        R = punishments_sorted[1]
        P = punishments_sorted[2]
        S = punishments_sorted[3]
    else:
        T = punishments[0]
        R = punishments[1]
        P = punishments[2]
        S = punishments[3]

    if c1 and c2:
        return (R, R)
    elif c1 and not c2:
        return (S, T)
    elif not c1 and c2:
        return (T, S)
    else:
        return (P, P)

def play_prisoners_dilemma(N:int, punishments:list, n1:int, n2:int):
    # Initialize lists with initial choices
    c1 = [False] if n1 == 0 else [True]
    c2 = [False] if n2 == 0 else [True]
    p1, p2 = get_punishments(c1[0], c2[0], punishments)
    p_hist1 = [p1]
    p_hist2 = [p2]

    # Play N rounds
    for i in range(1,N):
        # Player 1
        if i >= n1 or not c2[i-1]:
            c1.append(False)
        else:
            c1.append(True)
        # Player 2
        if i >= n2 or not c1[i-1]:
            c2.append(False)
        else:
            c2.append(True)

        # Get the punishments for both players
        p1, p2 = get_punishments(c1[i], c2[i], punishments)
        p_hist1.append(p1)
        p_hist2.append(p2)

    total_p1 = np.sum(p_hist1)
    total_p2 = np.sum(p_hist2)

    return total_p1

def play_prisoners_dilemma_sort(N:int, punishments:list, n1:int, n2:int, sort:bool):
    # Initialize lists with initial choices
    c1 = [False] if n1 == 0 else [True]
    c2 = [False] if n2 == 0 else [True]
    p1, p2 = get_punishments(c1[0], c2[0], punishments)
    p_hist1 = [p1]
    p_hist2 = [p2]

    # Play N rounds
    for i in range(1,N):
        # Player 1
        if i >= n1 or not c2[i-1]:
            c1.append(False)
        else:
            c1.append(True)
        # Player 2
        if i >= n2 or not c1[i-1]:
            c2.append(False)
        else:
            c2.append(True)

        # Get the punishments for both players
        p1, p2 = get_punishments(c1[i], c2[i], punishments, sort)
        p_hist1.append(p1)
        p_hist2.append(p2)

    total_p1 = np.sum(p_hist1)
    total_p2 = np.sum(p_hist2)

    return total_p1

def play_von_Neumann_neighbors(N:int, punishments:list, strategies:np.ndarray, i:int, j:int):
    lattice_width = strategies.shape[0]
    lattice_height = strategies.shape[1]
    n0 = strategies[i, j]
    # Get the strategies of the von Neumann neighbors, periodic boundary conditions
    n1 = strategies[(i+1)%lattice_height, j]  # right
    n2 = strategies[(i-1), j]  # left
    n3 = strategies[i, (j+1)%lattice_width] # up
    n4 = strategies[i, (j-1)] # down
    # Play the game
    total_p0_1 = play_prisoners_dilemma(N, punishments, n0, n1)
    total_p0_2 = play_prisoners_dilemma(N, punishments, n0, n2)
    total_p0_3 = play_prisoners_dilemma(N, punishments, n0, n3)
    total_p0_4 = play_prisoners_dilemma(N, punishments, n0, n4)

    prison_times = [total_p0_1, total_p0_2, total_p0_3, total_p0_4]
    total_time = np.sum(prison_times)

    return total_time

def get_best_strategy(strategies, prison_times):
    L = strategies.shape[0] 
    best_strategies = np.zeros((L, L))
    for i in range(strategies.shape[0]):
        for j in range(strategies.shape[1]):
            neighbor_strats =  np.array([strategies[i,j], strategies[(i+1)%L, j], strategies[(i-1), j], 
                        strategies[i, (j+1)%L], strategies[i, (j-1)]])
            neighbor_p_times = np.array([prison_times[i,j], prison_times[(i+1)%L, j], prison_times[(i-1), j], 
                        prison_times[i, (j+1)%L], prison_times[i, (j-1)]])
            
            best_neighbor_strats = np.where(neighbor_p_times == np.min(neighbor_p_times))[0].tolist()
            best_strategies[i,j] = neighbor_strats[np.random.choice(best_neighbor_strats)]

    return best_strategies