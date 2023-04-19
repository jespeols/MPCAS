clc, clear

% Define parameters
n = 5; 
nTrials = 10^4;
nEpochs = 20;
eta = 0.05;

counter = 0;

% Constructs the matrix of boolean inputs
booleanInputs = ff2n(n)';
booleanInputs(booleanInputs==0) = -1;

usedBooleanFunctions = [];

for trial = 1:nTrials
    % Sample the boolean function
    booleanOutputs = randi([0 1], 2^n, 1);
    booleanOutputs(booleanOutputs==0) = -1;
    string_copy = strjoin(string(booleanOutputs));
    
    if ~any(strcmp(string_copy,usedBooleanFunctions))
        W = randn(1, n)*1/sqrt(n);
        theta = 0;
    
        for epoch = 1:nEpochs
            totalError = 0;
            for mu = 1:2^n
                % compute output
                y = sign(dot(W,booleanInputs(:,mu)) - theta);
                if y == 0
                    y = 1;
                end
                
                error = booleanOutputs(mu) - y;
                % learning rules
                deltaW = eta*(error)*booleanInputs(:,mu)';
                deltaTheta = -eta*(error);
    
                % update neuron and weights
                W = W + deltaW;
                theta = theta + deltaTheta;
    
                totalError = totalError + abs(error);
            end
            if totalError == 0 % Linearly separable function found
                counter = counter + 1;
                break;
            end
        end     
        % Add booleanOutputs to usedBooleanFunctions
        usedBooleanFunctions = [usedBooleanFunctions string_copy];
    end
end
disp("Number of functions tried: ")
disp(length(usedBooleanFunctions))
disp("Number of linearly separable functions found: ")
disp(counter)