tic
clc, clear, clf

% Define patterns
allPatterns = [-1 -1 -1;
                1 -1 1;
               -1 1 1;
                1 1 -1;
               -1 -1 1;
                1 1 1;
               -1 1 -1;
                1 -1 -1;];
XORPatterns = allPatterns(1:4,:);

% Define network and other parameters
N = 3; % visible neurons
eta = 0.002;
k = 250;

nTrials = 1000;
miniBatchSize = 20;
N_outer = 3000;
N_inner = 2000;

% Define Boltzmann probability function
BoltzmannProb = @(b) 1/(1 + exp(-2*b));

DKL_list = [];
Pb_list = [];
for exp_nHiddenNeurons = 1:4

    M = 2^(exp_nHiddenNeurons-1); % Number of hidden neurons
    
    visibleOutput = zeros(N,k+1);
    hiddenOutput = zeros(M,k+1);
    localfieldHidden = zeros(M,k+1);
    localfieldVisible = zeros(N,k+1);
    samplePattern = zeros(1,size(XORPatterns,2));

    % Initialize weights and thresholds
    weights = normrnd(0,1,M,N);
    visibleThresholds = zeros(N, 1);
    hiddenThresholds = zeros(M, 1);

    for trial = 1:nTrials
        deltaW = zeros(size(weights));
        deltaThetaVisible = zeros(size(visibleThresholds));
        deltaThetaHidden = zeros(size(hiddenThresholds));
    
        for mu = 1:miniBatchSize
            rndIndx = randi(4);
            samplePattern = XORPatterns(rndIndx, :);
            
            % Initialize network at the sample pattern
            visibleOutput(:,1) = samplePattern;

            % update hidden neurons
            localfieldHidden(:,1) = weights*visibleOutput(:,1) - hiddenThresholds;
            for i = 1:M % assign states to hidden neurons stochastically
                r = rand;
                prob = BoltzmannProb(localfieldHidden(i,1));
                if r <= prob
                    hiddenOutput(i,1) = 1;
                else
                    hiddenOutput(i,1) = -1;
                end
            end
            
            for t = 2:k+1 % CD-k algorithm
                % update visible neurons              
                localfieldVisible(:,t-1) = (hiddenOutput(:,t-1)'*weights)' - visibleThresholds;
                for j = 1:N 
                    r = rand;
                    prob = BoltzmannProb(localfieldVisible(j,t-1));
                    if r <= prob
                        visibleOutput(j,t) = 1;
                    else
                        visibleOutput(j,t) = -1;
                    end
                end
                
                localfieldHidden(:,t) = weights*visibleOutput(:,t) - hiddenThresholds;
                % update hidden neurons
                for i = 1:M 
                    r = rand;
                    prob = BoltzmannProb(localfieldHidden(i,t));
                    if r <= prob
                        hiddenOutput(i,t) = 1;
                    else
                        hiddenOutput(i,t) = -1;
                    end
                end
            end
            % compute weight and threshold increments
            deltaW = deltaW + eta*(tanh(localfieldHidden(:,1))*visibleOutput(:,1)' - tanh(localfieldHidden(:,k+1)*visibleOutput(:,k+1)'));
            deltaThetaVisible = deltaThetaVisible - eta*(visibleOutput(:,1) - visibleOutput(:,k+1));
            deltaThetaHidden = deltaThetaHidden - eta*(tanh(localfieldHidden(:,1)) - tanh(localfieldHidden(:,k+1)));     
        end
        % update weights and thresholds
        weights = weights + deltaW;
        visibleThresholds = visibleThresholds + deltaThetaVisible;
        hiddenThresholds = hiddenThresholds + deltaThetaHidden;
    end
    
    % Count patterns to compute sampled Kullback-Leibler divergence
    counters = zeros(1, size(allPatterns,1));
    
    visibleOutput = zeros(N,N_inner+1);
    hiddenOutput = zeros(M,N_inner+1);
    localfieldHidden = zeros(M,N_inner+1);
    localfieldVisible = zeros(N,N_inner+1);

    for n = 1:N_outer
        selectedIndx = randi(8);
        selectedPattern = allPatterns(selectedIndx,:)';
    
        visibleOutput(:,1) = selectedPattern;
        localfieldHidden(:,1) = weights*visibleOutput(:,1) - hiddenThresholds;
        for i = 1:M % assign states to hidden neurons stochastically
            r = rand;
            prob = BoltzmannProb(localfieldHidden(i,1));
            if r < prob
                hiddenOutput(i,1) = 1;
            else
                hiddenOutput(i,1) = -1;
            end
        end
        for j=2:N_inner+1
            % update visible neurons
            localfieldVisible(:,j-1) = (hiddenOutput(:,j-1)'*weights)' - visibleThresholds;
            for k = 1:N 
                r = rand;
                prob = BoltzmannProb(localfieldVisible(k,j-1));
                if r < prob
                    visibleOutput(k,j) = 1;
                else
                    visibleOutput(k,j) = -1;
                end
            end
            % update hidden neurons
            localfieldHidden(:,j) = weights*visibleOutput(:,j) - hiddenThresholds;
            for i = 1:M 
                r = rand;
                prob = BoltzmannProb(localfieldHidden(i,j));
                if r < prob
                    hiddenOutput(i,j) = 1;
                else
                    hiddenOutput(i,j) = -1;
                end
            end
            for p = 1:8 % check for matches
                if visibleOutput(:,j) == allPatterns(p,:)'
                    %P_b(p) = P_b(p) + 1/(N_outer*N_inner);
                    counters(p) = counters(p) + 1;
                end
            end
        end
    end
    % Compute Boltzmann probability
    P_b = counters./(N_outer*N_inner);
    Pb_list = [Pb_list; P_b];
    disp("Probabilities for " + M + " neurons:")
    disp(P_b)

    % Compute DKL estimate
    DKL_est = sum(0.25.*log(0.25./P_b(1:4)));
    DKL_list = [DKL_list, DKL_est];
end
disp("Probabilites: ")
disp(Pb_list)
disp("Measured DKL: ")
disp(DKL_list)

plot([1 2 4 8], DKL_list, "b --o")
xlabel("Number of hidden neurons, M"), ylabel("D_{KL}")

% Compute the theoretical Kullback-Leibler divergence
M_t = 1:8;
DKL = zeros(1, length(M_t));
for i = 1:length(M_t)
    if M_t(i) < 2^(N-1)-1
        DKL(i) = log(2)*(N - fix(log2(M_t(i)+1)) - (M_t(i)+1)./2.^fix((log2(M_t(i)+1))));
    else
        DKL(i) = 0;
    end
end
hold on
plot(M_t, DKL, "r --o")
legend("Estimated D_{KL}", "Theoretical D_{KL}")

runtime = toc;
disp("Time = " + runtime/60 + " min")