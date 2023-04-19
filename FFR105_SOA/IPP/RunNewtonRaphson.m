% This function should run the Newton-Raphson method, making use of the
% other relevant functions (StepNewtonRaphson, DifferentiatePolynomial, and
% GetPolynomialValue). Before returning iterationValues any non-plottable values 
% (e.g. NaN) that can occur if the method fails (e.g. if the input is a
% first-order polynomial) should be removed, so that only values that
% CAN be plotted are returned. Thus, in some cases (again, the case of
% a first-order polynomial is an example) there may be no points to plot.

function iterationValues = RunNewtonRaphson(polynomialCoefficients, startingPoint, tolerance)

if length(polynomialCoefficients) < 3
    disp("The polynomial must be of a degree larger than 2.")
else
    fPrime = DifferentiatePolynomial(polynomialCoefficients, 1);
    fDoublePrime = DifferentiatePolynomial(polynomialCoefficients, 2);
    
    iterates = [startingPoint];
    absDifference = tolerance * 10;
    i = 1;
    
    while absDifference > tolerance
        fPrimeValue = GetPolynomialValue(iterates(i), fPrime);
        fDoublePrimeValue = GetPolynomialValue(iterates(i), fDoublePrime);
        iterates(i+1) = StepNewtonRaphson(iterates(end), fPrimeValue, fDoublePrimeValue);
        
        absDifference = abs(iterates(i + 1) - iterates(i));
        i = i + 1;
    
        if i == 15
            error("Did not converge")
        end

    end 
    iterationValues = iterates(~isnan(iterates)); % Removes NaN values
end




