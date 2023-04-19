% Obtains values for variables x1, x2

function x = DecodeChromosome(chromosome, variableRange)
    
    nGenes = size(chromosome, 2); % number of columns in the chromosome
    nHalf = fix(nGenes/2);

    x(1) = 0.0;
    for j = 1:nHalf % first half of genes decode x1
        x(1) = x(1) + chromosome(j)*2^(-j);
    end
    % scale to variable range
    x(1) = -variableRange + 2*variableRange*x(1)/(1 - 2^(-nHalf));  

    x(2) = 0.0;
    for j=1:nHalf
        x(2) = x(2) + chromosome(j+nHalf)*2^(-j);
    end
    x(2) = -variableRange + 2*variableRange*x(2)/(1 - 2^(-nHalf));

end