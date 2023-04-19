clc, clear

% Import raw data
rawTrainingSet = importdata("training_set.csv");
rawValidationSet = importdata("validation_set.csv");

% Normalize data
normTrainingSet = rawTrainingSet;
normValidationSet = rawValidationSet;
for i = 1:2
    meanValue = mean(rawTrainingSet(:,i));
    stdev = std(rawTrainingSet(:,i));
    normTrainingSet(:, i) = (rawTrainingSet(:,i) - meanValue)/stdev;
    normValidationSet(:, i) = (rawValidationSet(:,i) - meanValue)/stdev;
end

% Define parameters
M1 = 15;
eta = 0.008;
pTrain = size(normTrainingSet,1);
pVal = size(normValidationSet, 1);
classificationError = 100; % percent

% Initialize weights & thresholds
hiddenWeights = randn(M1,2);
outputWeights = randn(M1,1);
hiddenThresholds = zeros(M1,1);
outputThreshold = 0;

inputs = normTrainingSet(:,[1 2]);
targets = normTrainingSet(:,3);

inputsVal = normValidationSet(:,[1 2]);
targetsVal = normValidationSet(:,3);

errors = [];
errorThreshold = 0.12;
maxEpochs = 1000;
nEpochs = 0;
while nEpochs <= maxEpochs
        for i = 1:pTrain
            % choose random mu
            mu = randi(pTrain);

            % Compute hidden neuron outputs for the training set
            localFieldHidden = -hiddenThresholds + hiddenWeights*inputs(mu,:)';
            hiddenOutput = tanh(localFieldHidden);

            % Compute network outputs
            localFieldOutput = -outputThreshold + dot(outputWeights, hiddenOutput);
            networkOutput = tanh(localFieldOutput);
            
            % Calculate errors and weight/threshold increments
            outputError = (targets(mu) - networkOutput)*(1-tanh(localFieldOutput).^2); % generates values for all mu
            outputDeltaW = eta*outputError*hiddenOutput;
            outputDeltaTheta = -eta*outputError; % corresponds to sum over all mu
    
            hiddenError = outputError*(1-tanh(localFieldHidden).^2).*outputWeights; 
            hiddenDeltaW = eta*hiddenError*inputs(mu,:);
            hiddenDeltaTheta = -eta*hiddenError;
    
            % Update weights and thresholds
            hiddenWeights = hiddenWeights + hiddenDeltaW;
            hiddenThresholds = hiddenThresholds + hiddenDeltaTheta;
            outputWeights = outputWeights + outputDeltaW;
            outputThreshold = outputThreshold + outputDeltaTheta; 
        end

        % Compute hidden neuron outputs for the validation set 
        networkOutputVal = zeros(pVal, 1);
        for k = 1:pVal
                localFieldHiddenVal = -hiddenThresholds + hiddenWeights*inputsVal(k,:)';
                hiddenOutputVal = tanh(localFieldHiddenVal);

                % Compute network outputs
                localFieldOutputVal = -outputThreshold + outputWeights'*hiddenOutputVal;
                networkOutputVal(k) = tanh(localFieldOutputVal);
        end
            classificationError = 1/(2*pVal)*sum(abs(sign(networkOutputVal)-targetsVal))
            if classificationError < errorThreshold
                break;
            end
            %errors = [errors classificationError];
        nEpochs = nEpochs + 1
end

% Create .csv files
writematrix(hiddenWeights,"w1.csv");
writematrix(outputWeights,"w2.csv");
writematrix(hiddenThresholds,"t1.csv");
writematrix(outputThreshold,"t2.csv");
