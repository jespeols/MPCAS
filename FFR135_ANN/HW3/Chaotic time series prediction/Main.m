clc, clear

% load data
training_data = importdata("training-set.csv");
test_data = importdata("test-set-6.csv");

% parameters
N = 3;
M = 500; % reservoir neurons

w_in = normrnd(0,sqrt(0.002),[M N]);
w_reservoir = normrnd(0,sqrt(2/500),M);

% training
k = 0.01;
r = zeros(M,1);
R = zeros(M, length(training_data));

for t = 1:(length(training_data)-1)
    x = training_data(:,t);
    R(:,t) = r;

    r = tanh(w_reservoir*r + w_in*x);
end

% calculate output weights
I = eye(M);
w_out = training_data*R' * (R*R' + k*I)^(-1);

% feed the test data
for t = 1:(length(test_data)-1)
    x = test_data(:,t);
    R(:,t) = r;

    r = tanh(w_reservoir*r + w_in*x);
end
O = w_out*r;

% predict next 500 time steps
for t = 1:500
    r = tanh(w_reservoir*r + w_in*O);
    O = w_out*r;

    predictions(:,t) = O;
end
% export y-predictions to .csv file
y_predictions = predictions(2,:);
writematrix(y_predictions,"prediction.csv")

