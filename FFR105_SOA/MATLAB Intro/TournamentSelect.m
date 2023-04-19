% Selects individuals using the Tournament Selection method
% Tournament size will be 2

function iSelected = TournamentSelect(fitness, pTournament)
    
    populationSize = size(fitness, 1);
    iTmp1 = 1 + fix(rand*populationSize);
    iTmp2 = 1 + fix(rand*populationSize);

    r = rand;

    if r < pTournament % pick better individual
        if fitness(iTmp1) > fitness(iTmp2)
            iSelected = iTmp1;
        else
            iSelected = iTmp2;
        end
    else % pick the worse individual
        if fitness(iTmp1) > fitness(iTmp2)
            iSelected = iTmp2;
        else
            iSelected = iTmp1;
        end
    end

end