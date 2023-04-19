import numpy as np

class Agent:
    def __init__(self, x, y, N, initial_state):
        self.x = x
        self.y = y
        self.state = initial_state      # 0=healthy, 1=infected, 2=recovered, 3=dead, 4=immune   
        self.N = N                      # size of the lattice

    def __eq__(self, other):
        assert isinstance(other, Agent)
        return self.x == other.x

    def get_position(self):
        return self.x, self.y

    def get_state(self):
        return self.state
    
    def set_state(self, state):
        self.state = state

    def random_walk(self, x, y):
        random_tile = np.random.randint(0, 4)
        if random_tile == 0:                    # move up
            self.x += 1
        elif random_tile == 1: 
            self.x -= 1
        elif random_tile == 2:                  # move right
            self.y += 1
        elif random_tile == 3:
            self.y -= 1
        
        # check if the agent is outside the lattice, peiodiiic boundary conditions
        if x < 0:
            self.x = self.N - 1
        elif x > self.N - 1:
            self.x = 0
        if y < 0:
            self.y = self.N - 1
        elif y > self.N - 1:
            self.y = 0
    