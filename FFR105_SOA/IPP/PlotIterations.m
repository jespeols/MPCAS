% This method should plot the polynomial and the iterates obtained
% using NewtonRaphsonStep (if any iterates were generated).

function PlotIterations(polynomialCoefficients, iterationValues)

plotSpacing = 0.2*(max(iterationValues)-min(iterationValues));

x = linspace(min(iterationValues)-plotSpacing, max(iterationValues)+plotSpacing);
y = GetPolynomialValue(x, polynomialCoefficients);

yIterates = zeros(1, length(iterationValues));
for j = 1:length(iterationValues)
    yIterates(j) = GetPolynomialValue(iterationValues(j), polynomialCoefficients);
end

plot(x, y, "b"), hold on
plot(iterationValues, yIterates, "ok")
