% Assigns random values to all genes in the chromosomes contained in the
% population

function population = InitializePopulation(populationSize, nGenes)
    
    population = zeros(populationSize, nGenes); % each row is an individual
    for i = 1:populationSize % encode variables (genes) in the chromosomes
        for j = 1:nGenes % binary encoding
            s = rand;
            if s < 0.5
                population(i,j) = 0;
            else
                population(i,j) = 1;
            end
        end
    end
    
end