% Note: Each component of x should take values in [-a,a], where a = maximumVariableValue.

function x = DecodeChromosome(chromosome,numberOfVariables,maximumVariableValue)

    chromosomeSize = size(chromosome, 2);
    numberOfGenes = chromosomeSize/numberOfVariables;
    x=zeros(1, numberOfVariables);

    for i = 1:numberOfVariables
        x(i) = 0.0;
        for j = 1:numberOfGenes
            x(i) = x(i) + chromosome(j + (i-1)*numberOfGenes)*2^(-j);
        end
        x(i) = -maximumVariableValue + 2*maximumVariableValue*x(i)/(1 - 2^(-numberOfGenes));  
    end
end