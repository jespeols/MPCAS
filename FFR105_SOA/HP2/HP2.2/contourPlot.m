clc, clear, clf

x = linspace(-5,5);
y = linspace(-5,5);
[X,Y] = meshgrid(x,y);

objectiveFunction = @(x,y) (x.^2 + y - 11).^2 + (x + y.^2 - 7).^2;

a = 0.01; 
contour(X,Y,log(a + objectiveFunction(X,Y)))

% Add local minima markers
xCoordinates = [-2.80 2.98 -3.79 3.59];
yCoordinates = [3.14 2.02 -3.28 -1.87];
hold on
scatter(xCoordinates,yCoordinates,"filled","MarkerFaceColor","r")
xlabel("x"), ylabel("y"), title("log(0.01 + f(x,y))")