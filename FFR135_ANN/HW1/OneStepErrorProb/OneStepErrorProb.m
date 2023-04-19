% Runs the trials and calculates the one-step error probability

clc, clear

% Define parameters
p = [12, 24, 48, 70, 100, 120];
N = 120;

errorProbabilities = zeros(1, length(p));

% Run the trials for the different values of p
for n = 1:length(p)
    errorCounter = 0; 
    for i = 1:1e5
        % Generate a set of random pattern then randomly select one pattern & one neuron.
        randomPatterns = GenerateRandomPatterns(p(n), N);
        iSelectedPattern = randi(p(n));
        iSelectedNeuron = randi(N);

        % Store patterns into the network by calculating weights
        weights = CalculateWeights(randomPatterns, iSelectedNeuron);

        % Update the selected neuron
        selectedNeuronState = sign(randomPatterns(iSelectedNeuron, iSelectedPattern));
        updatedNeuronState = sign(weights*randomPatterns(:,iSelectedPattern));

        if updatedNeuronState == 0
            updatedNeuronState = 1;
        end
        
        % Check if there has been an error
        if updatedNeuronState ~= selectedNeuronState
            errorCounter = errorCounter + 1;
        end
    end

    pOneStepError = errorCounter/1e5;
    errorProbabilities(n) = pOneStepError;  
end

disp("One-step error probability for [12, 24, 48, 70, 100, 120] stored patterns: ")
disp(errorProbabilities);