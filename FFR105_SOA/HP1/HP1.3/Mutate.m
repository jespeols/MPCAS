% Carries out mutations on an individual and returns the mutated one

function mutatedIndividual = Mutate(individual, mutationProbability)

    numberOfGenes = size(individual, 2);
    mutatedIndividual = individual;
    for j = 1:numberOfGenes
        r = rand;
        if r < mutationProbability 
            mutatedIndividual(j) = 1 - individual(j); 
        end
    end

end