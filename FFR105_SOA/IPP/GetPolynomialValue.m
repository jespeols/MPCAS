% This function should return the value of the polynomial f(x) = a0x^0 + a1x^1 + a2x^2 ...
% where a0, a1, ... are obtained from polynomialCoefficients.

function value = GetPolynomialValue(x, polynomialCoefficients)

sum = 0;
for i=1:(length(polynomialCoefficients) - 1)
    sum = sum + polynomialCoefficients(i + 1) * x.^i;
end
value = polynomialCoefficients(1) + sum;

end