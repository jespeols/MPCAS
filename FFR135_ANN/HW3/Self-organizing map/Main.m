clc, clear, clf

iris_data = importdata("iris-data.csv");
iris_labels = importdata("iris-labels.csv");

iris_data_norm = iris_data/max(max(iris_data));

% Parameters 
m = 40;
n = 4;
p = size(iris_data_norm,1);
eta_0 = 0.1;
d_eta = 0.01;
sigma_0 = 10;
d_sigma = 0.05;
n_epochs = 10;

w_untrained = rand([m m n]);
w = w_untrained;

% train the network
for epoch = 1:n_epochs 
    % update eta & sigma
    eta = eta_0*exp(-d_eta*epoch);
    sigma = sigma_0*exp(-d_sigma*epoch);

    for mu = 1:p
        d_sum = zeros(size(w,[1 2]));
        delta_w = zeros(1,4);

        r = randi(p);
        x = iris_data_norm(r,:);

        % calculate distances
        for k = 1:length(x)
            d_sum = d_sum + (w(:,:,k) - x(k)).^2;
        end
        distance = sqrt(d_sum);
        min_distance = min(distance,[],'all');
        [i_min, j_min] = find(distance==min_distance);

        % update winning neuron
        h = exp(0); % zero for winning neuron
        for k = 1:n
            delta_w(k) = eta*h*(x(k) - w(i_min,j_min,k));
            w(i_min,j_min,k) = w(i_min,j_min,k) + delta_w(k);
        end

        % find and update neighboring neurons
        for i = 1:m
            for j = 1:m
                neuron_distance = calc_euclid_dist([i j],[i_min j_min]);
                if neuron_distance < 3*sigma
                    h = exp(-1/(2*sigma^2)*neuron_distance);
                    for k = 1:n
                        delta_w(k) = eta*h*(x(k) - w(i,j,k));
                        w(i,j,k) = w(i,j,k) + delta_w(k);
                    end
                end
            end
        end
    end
end

% feed all inputs and calculate distances
winning_neurons = zeros(p,2);
winning_neurons_untrained = zeros(p,2);
for mu = 1:p
    d_sum = zeros(size(w,[1 2]));
    d_sum_untrained = zeros(size(w,[1 2]));

    x = iris_data_norm(mu,:);
    % calculate distances
    for k = 1:length(x)
        d_sum = d_sum + (w(:,:,k) - x(k)).^2;
        d_sum_untrained = d_sum_untrained + (w_untrained(:,:,k) - x(k)).^2;
    end
    distance = sqrt(d_sum);
    distance_untrained = sqrt(d_sum_untrained);
    min_distance = min(distance,[],'all');
    min_distance_untrained = min(distance_untrained,[],"all");

    [winning_neurons(mu,1),winning_neurons(mu,2)] = find(distance==min_distance);
    [winning_neurons_untrained(mu,1),winning_neurons_untrained(mu,2)] = find(distance_untrained==min_distance_untrained);
end
winning_neurons = [winning_neurons iris_labels];
winning_neurons_untrained = [winning_neurons_untrained iris_labels];

% add noise to output data
r = normrnd(0,2e-2,[p 2]);
winning_neurons(:,1) = winning_neurons(:,1) + r(:,1);
winning_neurons(:,2) = winning_neurons(:,2) + r(:,2);

% plot results
pointer_size = 25;

colorID = zeros(p,3);
colorID(winning_neurons(:,3)==0,1) = 1; 
colorID(winning_neurons(:,3)==1,2) = 1;
colorID(winning_neurons(:,3)==2,3) = 1;

figure(1); 
scatter(winning_neurons(:,1),winning_neurons(:,2),pointer_size,colorID,"filled")
axis([0 m 0 m])
xlabel("x"), ylabel("y"), title("Trained weights")

% do the same for untrained data
r = normrnd(0,2e-1,[p 2]);
winning_neurons_untrained(:,1) = winning_neurons_untrained(:,1) + r(:,1);
winning_neurons_untrained(:,2) = winning_neurons_untrained(:,2) + r(:,2);

colorID = zeros(p,3);
colorID(winning_neurons_untrained(:,3)==0,1) = 1; 
colorID(winning_neurons_untrained(:,3)==1,2) = 1;
colorID(winning_neurons_untrained(:,3)==2,3) = 1;

figure(2);
scatter(winning_neurons_untrained(:,1),winning_neurons_untrained(:,2),pointer_size,colorID,'filled')
axis([0 m 0 m])
xlabel("x"), ylabel("y"),title("Initial weights")