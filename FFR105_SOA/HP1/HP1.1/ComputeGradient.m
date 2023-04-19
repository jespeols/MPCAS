% This function should return the gradient of f_p = f + penalty.
% You may hard-code the gradient required for this specific problem.

function gradF = ComputeGradient(x,mu)
    
    constrainedGradient = [2*x(1)-2, 4*x(2)-8];
    if (x(1)^2 + x(2)^2 - 1 <= 0) 
        gradF = constrainedGradient;
    else
        unconstrainedGradient = [4*mu*x(1)*(x(1)^2+x(2)^2-1), 4*mu*x(2)*(x(1)^2+x(2)^2-1)];
        gradF = constrainedGradient + unconstrainedGradient;
    end
end


