cityLocation = LoadCityLocations;
nCities = size(cityLocation,1);
path = randperm(nCities);                
tspFigure = InitializeTspPlot(cityLocation,[0 20 0 20]); 
connection = InitializeConnections(cityLocation); 
% run("BestResultFound.m") % REMOVE
PlotPath(connection,cityLocation,path);
% PlotPath(connection,cityLocation,bestPath); % REMOVE
