import numpy as np
import matplotlib
import time


start_time = time.time()
# define functions


def calculate_weights(patterns, index_neuron):
    
    N = np.size(patterns, 1)

    # print(patterns[index_neuron, :])
    # print(patterns.T.shape)
    weights = 1/N * np.matmul(patterns[index_neuron, :], patterns.T)
    # print(weights.shape)
    weights[index_neuron] = 0

    return weights


def generate_random_patterns(p, N):
    
    random_patterns = np.random.randint(2, size=(N, p))
    random_patterns[random_patterns == 0] = -1
    
    return random_patterns


### Main ###

# define parameters
n_patterns = [12, 24, 48, 70, 100, 120]  # number of patterns
N = 120
n_trials = 1e5

# run trials for the different values of p
error_probabilities = []
for p in n_patterns:
    error_counter = 0
    for i in np.arange(n_trials):
        # generate random patterns & randomly select pattern, neuron
        random_patterns = generate_random_patterns(p, N)
        index_selected_pattern = np.random.randint(p)
        index_selected_neuron = np.random.randint(N)
        
        # store patterns by generating weights
        weights = calculate_weights(random_patterns, index_selected_neuron)
        
        # update selected neuron
        selected_neuron_state = np.sign(random_patterns[index_selected_neuron, index_selected_pattern])
        updated_neuron_state = np.sign(np.matmul(weights, random_patterns[:, index_selected_pattern]))
        if updated_neuron_state == 0:
            updated_neuron_state = 1
            
        # check for errors
        if updated_neuron_state != selected_neuron_state:
            error_counter += 1
            
    p_one_step_error = error_counter/n_trials
    error_probabilities.append(p_one_step_error)
    
print('One-step error probability for ', str(n_patterns), ' stored patterns')
print(error_probabilities)

print("--- %s seconds ---" % (time.time() - start_time))

            
