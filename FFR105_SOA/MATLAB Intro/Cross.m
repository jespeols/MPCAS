% Performs crossover between two chromosomes to for a new chromosome pair

function newChromosomePair = Cross(chromosome1, chromosome2)

    nGenes = size(chromosome1, 2); % Both chromosomes must have same length!

    crossoverPoint = 1 + fix(rand*(nGenes-1)); % nGenes-1 since there must be a gene (at the end)
                                               % to be crossed over
    
    newChromosomePair = zeros(2, nGenes);
    for j = 1:nGenes
        if j <= crossoverPoint
            newChromosomePair(1, j) = chromosome1(j);
            newChromosomePair(2, j) = chromosome2(j);
        else
            newChromosomePair(1, j) = chromosome2(j);
            newChromosomePair(2, j) = chromosome1(j);
        end
    end

end