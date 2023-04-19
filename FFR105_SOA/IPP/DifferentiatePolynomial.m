% This method should return the coefficients of the k-th derivative (defined by
% the derivativeOrder) of the polynomial given by the polynomialCoefficients (see also GetPolynomialValue)

function derivativeCoefficients = DifferentiatePolynomial(polynomialCoefficients, derivativeOrder)

if derivativeOrder == 0
derivativeCoefficients = polynomialCoefficients;
else
    for j = 1:derivativeOrder
        tempDerivative = zeros(1, length(polynomialCoefficients) - 1);
        for i = 1:length(polynomialCoefficients) - 1
            tempDerivative(i) = polynomialCoefficients(i + 1) * i; 
        end
        polynomialCoefficients = tempDerivative;
    end
    derivativeCoefficients = tempDerivative;
end