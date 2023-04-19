function pathLength = GetPathLength(path, cityLocation)
    
    distance = zeros(1, length(path)+1);
    for i = 2:(length(path) + 1)
        if i == (length(path) + 1) % calculate distance to starting city
            FirstCity = path(1);
            LastCity = path(end);       
            locFirstCity = cityLocation(FirstCity,:);
            locLastCity = cityLocation(LastCity,:);

            xDistance = locLastCity(1) - locFirstCity(1);
            yDistance = locLastCity(2) - locFirstCity(2); 
            distance(i-1) = sqrt(xDistance^2 + yDistance^2);
        else
            City1 = path(i-1);
            City2 = path(i);
            loc1 = cityLocation(City1,:);
            loc2 = cityLocation(City2,:);
        
            xDistance = loc2(1) - loc1(1);
            yDistance = loc2(2) - loc1(2);
            distance(i-1) = sqrt(xDistance^2 + yDistance^2);
        end        
    end
    pathLength = sum(distance);

end