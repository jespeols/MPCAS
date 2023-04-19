clc, clear
format long

maxIterations = 15;
xList = [];
fList = [];
for iter = 1:maxIterations
    duplicateExists = false;
    [xBest, fVal] = RunPSO();
    for i = 1:size(xList,1)
        if round(xBest,1) == round(xList(i,:),1)
            duplicateExists = true;
        end
    end
    if duplicateExists ~= true
        xList = [xList; xBest];
        fList = [fList; fVal];
    end
    
end
disp("Unique local minima identified:")
disp([xList fList])