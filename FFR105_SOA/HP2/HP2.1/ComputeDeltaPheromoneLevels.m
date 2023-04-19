function deltaPheromoneLevel = ComputeDeltaPheromoneLevels(pathCollection, pathLengthCollection)
    
    numberOfCities = length(pathCollection(1,:));
    deltaPheromoneLevel = zeros(numberOfCities);
    numberOfAnts = length(pathLengthCollection);

    for k = length(numberOfAnts) % Calculate contribution of each ant
        currentPath = pathCollection(k,:);
        currentPathLength = pathLengthCollection(k);

        for city1 = 1:numberOfCities
            for city2 =  1:numberOfCities
                currentEdge = [city1 city2]; % from city1 to city2
                if city2 == currentPath(1) && city1 == currentPath(end)
                    deltaPheromoneLevel(city2,city1) =  deltaPheromoneLevel(city2,city1) + 1/currentPathLength;
                else
                    for i = 2:length(currentPath) % check which edge where traversed
                        if currentPath(i-1) == currentEdge(1) && currentPath(i) == currentEdge(2) % Ant k has traversed the current edge
                            deltaPheromoneLevel(city2,city1) = deltaPheromoneLevel(city2,city1) + 1/currentPathLength;
                        else
                            deltaPheromoneLevel(city2,city1) = deltaPheromoneLevel(city2,city1) + 0;
                        end
                    end
                end
            end
        end
    end

end