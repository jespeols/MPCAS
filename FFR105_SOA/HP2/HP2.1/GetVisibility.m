function visibility = GetVisibility(cityLocation)
    
    numberOfCities = length(cityLocation);
    distance = zeros(numberOfCities);
    for i = 1:numberOfCities
        for j = 1:numberOfCities
            xDistance = cityLocation(i,1)-cityLocation(j,1);
            yDistance = cityLocation(i,2)-cityLocation(j,2);
            distance(i,j) = sqrt(xDistance^2 + yDistance^2); 
            if distance(i,j) == 0
                distance(i,j) = 1e-15; % Avoid division by zero
            end
        end
    end
    visibility = 1./distance;

end