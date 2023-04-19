function [xMinima, minVal] = RunPSO()

    objectiveFunction = @(x) (x(:,1).^2 + x(:,2) - 11).^2 + (x(:,1) + x(:,2).^2 - 7).^2;
    
    % Parameters
    N = 30; 
    n = 2; 
    xMin = -5;
    xMax = -xMin;
    alpha = 1; 
    timeStep = 1; 
    vMax = (xMax-xMin)/timeStep;
    c1 = 2;
    c2 = 2;
    inertiaWeight = 1.4; % starting value
    beta = 0.95; 
    inertiaWeightLB = 0.4;
    
    x = zeros(N,n);
    v = zeros(N,n);
    % Initialize positions and velocities
    for i = 1:N
        for j = 1:n
            r = rand;
            x(i,j) = xMin + r*(xMax - xMin);
        end
        for j=1:n
            r = rand;
            v(i,j) = alpha/timeStep*(xMin + r*(xMax-xMin));
        end
    end
    
    % Define initial bests
    xPB = x; 
    PB = objectiveFunction(xPB); 
    initialValues = objectiveFunction(x);
    initialMax = max(initialValues);
    xSB = x(initialValues==initialMax,:); 
    SB = objectiveFunction(xSB); 
    
    maxIter = 500;
    for iter = 1:maxIter 
        values = objectiveFunction(x);   
        for i = 1:N 
            if values(i) < PB
                PB = values(i);
                xPB(i,:) = x(i,:);
            end
            if values(i) < SB
                SB = values(i);
                xSB = x(i,:);
            end
        end
        r = rand;
        q = rand;
        for i = 1:N 
            for j = 1:n 
                v(i,j) = inertiaWeight*v(i,j) + c1*q*(xPB(i,j)-x(i,j))/timeStep + c2*r*(xSB(j)-x(i,j))/timeStep;
                if v(i,j) > vMax 
                    v(i,j) = vMax;
                elseif v(i,j) < -vMax
                    v(i,j) = -vMax;
                end
                x(i,j) = x(i,j) + v(i,j)*timeStep;
            end
        end
        
        if inertiaWeight > inertiaWeightLB
            inertiaWeight = beta*inertiaWeight;
        end
    end
    xMinima = xSB;
    minVal = objectiveFunction(xMinima);

end