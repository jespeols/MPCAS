function mutatedChromosome = Mutate(chromosome, mutationProbability)

    nGenes = size(chromosome, 2);
    mutatedChromosome = chromosome;
    for j = 1:nGenes
        r = rand;
        if r < mutationProbability % perform mutation
            mutatedChromosome(j) = 1 - chromosome(j); % gives opposite value, since the encoding is binary
        end
    end

end