function path = GeneratePath(pheromoneLevel, visibility, alpha, beta)
    
    numberOfCities = size(pheromoneLevel,1);
    startingCity = randi(50);
    tabuList = [startingCity];
    unvisitedCities = setdiff(1:numberOfCities, tabuList);
    
    tourFinished = false;
    while tourFinished ~= true
        probability = zeros(1,length(unvisitedCities));
    
        j = tabuList(end); % Current node
        denominatorSum = dot(pheromoneLevel(unvisitedCities,j).^alpha,  visibility(unvisitedCities,j).^beta);
        for i = 1:length(unvisitedCities)
            iCity = unvisitedCities(i);
            probability(i) = pheromoneLevel(iCity,j)^alpha*visibility(iCity,j)^beta/denominatorSum; 
        end    
        chosenCity = GetNode(probability,unvisitedCities);
    
        tabuList = [tabuList, chosenCity];
        unvisitedCities = setdiff(1:numberOfCities, tabuList); % Removes visited cities
        
        if length(tabuList) == 50
            tourFinished = true;
        end
    end
    path = tabuList;

end