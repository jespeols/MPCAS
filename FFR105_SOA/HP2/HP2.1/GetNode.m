function chosenCity = GetNode(probability, unvisitedCities)

    r = rand;
    probabilitySum = 0;
    for i=1:length(unvisitedCities)
        iCity = unvisitedCities(i);
        if i==1 && i~=length(unvisitedCities)
            if r <= probability(i)
                chosenCity = iCity;
                break;
            else
                probabilitySum = probabilitySum + probability(i);
            end
        elseif i==length(unvisitedCities) 
            chosenCity = iCity;
        else
            if r <= (probabilitySum + probability(i)) && r > probabilitySum
               chosenCity = iCity;
               break;
            else
                probabilitySum = probabilitySum + probability(i);
            end
        end
    end

end