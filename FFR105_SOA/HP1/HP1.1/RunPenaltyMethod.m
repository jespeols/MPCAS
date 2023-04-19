%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Penalty method for minimizing
%
% (x1-1)^2 + 2(x2-2)^2, s.t.
%
% x1^2 + x2^2 - 1 <= 0.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The values below are suggestions - you may experiment with
% other values of eta and other (increasing) sequences of the
% µ parameter (muValues).

muValues = [1 10 100 500 1000];
eta = 0.0001;
xStart =  [1 2];
gradientTolerance = 1E-5;

hold on, grid on
for i = 1:length(muValues)
 mu = muValues(i);
 x = RunGradientDescent(xStart,mu,eta,gradientTolerance);
 sprintf('x(1) = %3f, x(2) = %3f mu = %d',x(1),x(2),mu)

 plot(mu, x(1),"ob",MarkerFaceColor="b")
 plot(mu, x(2),"or",MarkerFaceColor="r")
end
yline(x(1),"b"), yline(x(2),"r")
xlabel("\mu")
ylabel("x-values")
legend("x_1","x_2")

