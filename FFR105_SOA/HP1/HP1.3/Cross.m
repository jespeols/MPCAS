% Carries out crossover between two chromomes and returns the resulting "new" individuals

function newIndividuals = Cross(individual1, individual2)

    numberOfGenes = size(individual1, 2); 
    crossoverPoint = 1 + fix(rand*(numberOfGenes-1));
    
    newIndividuals = zeros(2, numberOfGenes);
    for j = 1:numberOfGenes
        if j <= crossoverPoint
            newIndividuals(1, j) = individual1(j);
            newIndividuals(2, j) = individual2(j);
        else
            newIndividuals(1, j) = individual2(j);
            newIndividuals(2, j) = individual1(j);
        end
    end
    
end