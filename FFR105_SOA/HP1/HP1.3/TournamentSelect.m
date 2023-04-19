% Returns the index of the selected individual. Selection is carried via tournament selection.

function selectedIndividualIndex = TournamentSelect(fitnessList, tournamentProbability, tournamentSize)

    populationSize = size(fitnessList, 2);
    tournamentPopulation = zeros(2, tournamentSize);

    for i = 1:tournamentSize
        tournamentPopulation(1, i) = fitnessList(i);
        tournamentPopulation(2, i) = 1 + fix(rand*populationSize);
    end
    
    [B, I] = sort(tournamentPopulation(1,:), "descend");
    sortedTournamentPopulation = tournamentPopulation(:, I);
    
    selectedIndividualIndex = 0;
    for j = 1:length(tournamentSize)
        r = rand;
        if r < tournamentProbability
            selectedIndividualIndex = sortedTournamentPopulation(2, j);
            break;
        end
    end

    if selectedIndividualIndex == 0
        selectedIndividualIndex = sortedTournamentPopulation(2, end);
    end


end