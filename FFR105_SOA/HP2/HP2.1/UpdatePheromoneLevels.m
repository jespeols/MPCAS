function outputPheromoneLevel = UpdatePheromoneLevels(inputPheromoneLevel,deltaPheromoneLevel,rho)
    
    outputPheromoneLevel = (1 - rho)*inputPheromoneLevel + deltaPheromoneLevel;

    checkLB = outputPheromoneLevel < 1e-15;
    outputPheromoneLevel(checkLB) = 1e-15;
    
end