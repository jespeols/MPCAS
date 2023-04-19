% Generate and return random patterns

function randomPatterns = GenerateRandomPatterns(p, N)

    tempMatrix = randi([0 1], N, p);
    for i=1:N
        for j = 1:p
            if tempMatrix(i,j) == 0
                tempMatrix(i,j) = -1;
            end
        end
    end
    randomPatterns = tempMatrix;

end