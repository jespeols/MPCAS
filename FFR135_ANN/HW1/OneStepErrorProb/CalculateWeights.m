% Generate the weights using Hebb's rule

function weights = CalculateWeights(patterns, iNeuron)

    N = size(patterns, 1);
    weights = 1/N*patterns(iNeuron,:)*patterns';
    weights(iNeuron) = 0; % sets diagonal element to zero

end