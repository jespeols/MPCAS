%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter specifications
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numberOfRuns = 100;                % Do NOT change
populationSize = 100;              % Do NOT change
maximumVariableValue = 5;          % Do NOT change (x_i in [-a,a], where a = maximumVariableValue)
numberOfGenes = 50;                % Do NOT change
numberOfVariables = 2;		   % Do NOT change
numberOfGenerations = 300;         % Do NOT change
tournamentSize = 2;                % Do NOT change
tournamentProbability = 0.75;      % Do NOT change
crossoverProbability = 0.8;        % Do NOT change


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Batch runs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define more runs here (pMut < 0.02) ...

mutationProbability = 0;
sprintf('Mutation rate = %0.5f', mutationProbability)
maximumFitnessList0 = zeros(numberOfRuns,1);
for i = 1:numberOfRuns 
 [maximumFitness, bestVariableValues]  = RunFunctionOptimization(populationSize, numberOfGenes, numberOfVariables, maximumVariableValue, tournamentSize, ...
                                       tournamentProbability, crossoverProbability, mutationProbability, numberOfGenerations);
 sprintf('Run: %d, Score: %0.10f', i, maximumFitness)
  maximumFitnessList0(i,1) = maximumFitness;
end

mutationProbability = 0.02;
sprintf('Mutation rate = %0.5f', mutationProbability)
maximumFitnessList002 = zeros(numberOfRuns,1);
for i = 1:numberOfRuns 
 [maximumFitness, bestVariableValues]  = RunFunctionOptimization(populationSize, numberOfGenes, numberOfVariables, maximumVariableValue, tournamentSize, ...
                                       tournamentProbability, crossoverProbability, mutationProbability, numberOfGenerations);
 sprintf('Run: %d, Score: %0.10f', i, maximumFitness)
  maximumFitnessList002(i,1) = maximumFitness;
end

mutationProbability = 0.05;
sprintf('Mutation rate = %0.5f', mutationProbability)
maximumFitnessList005 = zeros(numberOfRuns,1);
for i = 1:numberOfRuns 
 [maximumFitness, bestVariableValues]  = RunFunctionOptimization(populationSize, numberOfGenes, numberOfVariables, maximumVariableValue, tournamentSize, ...
                                       tournamentProbability, crossoverProbability, mutationProbability, numberOfGenerations);
 sprintf('Run: %d, Score: %0.10f', i, maximumFitness)
  maximumFitnessList005(i,1) = maximumFitness;
end

mutationProbability = 0.1;
sprintf('Mutation rate = %0.5f', mutationProbability)
maximumFitnessList01 = zeros(numberOfRuns,1);
for i = 1:numberOfRuns 
 [maximumFitness, bestVariableValues]  = RunFunctionOptimization(populationSize, numberOfGenes, numberOfVariables, maximumVariableValue, tournamentSize, ...
                                       tournamentProbability, crossoverProbability, mutationProbability, numberOfGenerations);
 sprintf('Run: %d, Score: %0.10f', i, maximumFitness)
  maximumFitnessList01(i,1) = maximumFitness;
end

mutationProbability = 0.2;
sprintf('Mutation rate = %0.5f', mutationProbability)
maximumFitnessList02 = zeros(numberOfRuns,1);
for i = 1:numberOfRuns 
 [maximumFitness, bestVariableValues]  = RunFunctionOptimization(populationSize, numberOfGenes, numberOfVariables, maximumVariableValue, tournamentSize, ...
                                       tournamentProbability, crossoverProbability, mutationProbability, numberOfGenerations);
 sprintf('Run: %d, Score: %0.10f', i, maximumFitness)
  maximumFitnessList02(i,1) = maximumFitness;
end

mutationProbability = 0.25;
sprintf('Mutation rate = %0.5f', mutationProbability)
maximumFitnessList025 = zeros(numberOfRuns,1);
for i = 1:numberOfRuns 
 [maximumFitness, bestVariableValues]  = RunFunctionOptimization(populationSize, numberOfGenes, numberOfVariables, maximumVariableValue, tournamentSize, ...
                                       tournamentProbability, crossoverProbability, mutationProbability, numberOfGenerations);
 sprintf('Run: %d, Score: %0.10f', i, maximumFitness)
  maximumFitnessList025(i,1) = maximumFitness;
end

mutationProbability = 0.35;
sprintf('Mutation rate = %0.5f', mutationProbability)
maximumFitnessList035 = zeros(numberOfRuns,1);
for i = 1:numberOfRuns 
 [maximumFitness, bestVariableValues]  = RunFunctionOptimization(populationSize, numberOfGenes, numberOfVariables, maximumVariableValue, tournamentSize, ...
                                       tournamentProbability, crossoverProbability, mutationProbability, numberOfGenerations);
 sprintf('Run: %d, Score: %0.10f', i, maximumFitness)
  maximumFitnessList035(i,1) = maximumFitness;
end

mutationProbability = 0.45;
sprintf('Mutation rate = %0.5f', mutationProbability)
maximumFitnessList045 = zeros(numberOfRuns,1);
for i = 1:numberOfRuns 
 [maximumFitness, bestVariableValues]  = RunFunctionOptimization(populationSize, numberOfGenes, numberOfVariables, maximumVariableValue, tournamentSize, ...
                                       tournamentProbability, crossoverProbability, mutationProbability, numberOfGenerations);
 sprintf('Run: %d, Score: %0.10f', i, maximumFitness)
  maximumFitnessList045(i,1) = maximumFitness;
end

mutationProbability = 0.6;
sprintf('Mutation rate = %0.5f', mutationProbability)
maximumFitnessList06 = zeros(numberOfRuns,1);
for i = 1:numberOfRuns 
 [maximumFitness, bestVariableValues]  = RunFunctionOptimization(populationSize, numberOfGenes, numberOfVariables, maximumVariableValue, tournamentSize, ...
                                       tournamentProbability, crossoverProbability, mutationProbability, numberOfGenerations);
 sprintf('Run: %d, Score: %0.10f', i, maximumFitness)
  maximumFitnessList06(i,1) = maximumFitness;
end

mutationProbability = 0.8;
sprintf('Mutation rate = %0.5f', mutationProbability)
maximumFitnessList08 = zeros(numberOfRuns,1);
for i = 1:numberOfRuns 
 [maximumFitness, bestVariableValues]  = RunFunctionOptimization(populationSize, numberOfGenes, numberOfVariables, maximumVariableValue, tournamentSize, ...
                                       tournamentProbability, crossoverProbability, mutationProbability, numberOfGenerations);
 sprintf('Run: %d, Score: %0.10f', i, maximumFitness)
  maximumFitnessList08(i,1) = maximumFitness;
end
% ... and here (pMut > 0.02)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Summary of results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add more results summaries here (pMut < 0.02) ...

average0 = mean(maximumFitnessList0);
median0 = median(maximumFitnessList0);
std0 = sqrt(var(maximumFitnessList0));
sprintf('PMut = 0.01: Median: %0.10f, Average: %0.10f, STD: %0.10f', median0, average0, std0)

average002 = mean(maximumFitnessList002);
median002 = median(maximumFitnessList002);
std002 = sqrt(var(maximumFitnessList002));
sprintf('PMut = 0.02: Median: %0.10f, Average: %0.10f, STD: %0.10f', median002, average002, std002)

average005 = mean(maximumFitnessList005);
median005 = median(maximumFitnessList005);
std005 = sqrt(var(maximumFitnessList005));
sprintf('PMut = 0.05: Median: %0.10f, Average: %0.10f, STD: %0.10f', median005, average005, std005)

average01 = mean(maximumFitnessList01);
median01 = median(maximumFitnessList01);
std01 = sqrt(var(maximumFitnessList01));
sprintf('PMut = 0.1: Median: %0.10f, Average: %0.10f, STD: %0.10f', median01, average01, std01)

average02 = mean(maximumFitnessList02);
median02 = median(maximumFitnessList02);
std02 = sqrt(var(maximumFitnessList02));
sprintf('PMut = 0.2: Median: %0.10f, Average: %0.10f, STD: %0.10f', median02, average02, std02)

average025 = mean(maximumFitnessList025);
median025 = median(maximumFitnessList025);
std025 = sqrt(var(maximumFitnessList025));
sprintf('PMut = 0.25: Median: %0.10f, Average: %0.10f, STD: %0.10f', median025, average025, std025)

average035 = mean(maximumFitnessList035);
median035 = median(maximumFitnessList035);
std035 = sqrt(var(maximumFitnessList035));
sprintf('PMut = 0.35: Median: %0.10f, Average: %0.10f, STD: %0.10f', median035, average035, std035)

average045 = mean(maximumFitnessList045);
median045 = median(maximumFitnessList045);
std045 = sqrt(var(maximumFitnessList045));
sprintf('PMut = 0.45: Median: %0.10f, Average: %0.10f, STD: %0.10f', median045, average045, std045)

average06 = mean(maximumFitnessList06);
median06 = median(maximumFitnessList06);
std06 = sqrt(var(maximumFitnessList06));
sprintf('PMut = 0.6: Median: %0.10f, Average: %0.10f, STD: %0.10f', median06, average06, std06)

average08 = mean(maximumFitnessList08);
median08 = median(maximumFitnessList08);
std08 = sqrt(var(maximumFitnessList08));
sprintf('PMut = 0.8: Median: %0.10f, Average: %0.10f, STD: %0.10f', median08, average08, std08)
% ... and here (pMut > 0.02)

PmutValues = [0.0 0.02 0.05 0.1 0.2 0.25 0.35 0.45 0.6 0.8];
medianValues = [median0 median002 median005 median01 median02 median025 median035 median045 median06 median08];

plot(PmutValues, medianValues, "ob", MarkerFaceColor="b")
grid on
xlabel("p_{mut}")
ylabel("Median performance, F")
