clear all, close all

% ======================== Parameter Setup =========================%

n=7;
Ntrain=100;
Ntest=10000;
lower_noise_multiplier = 10^-3;
upper_noise_multiplier = 10^3;
x_mu_max=10;
totalExp = 100;
betasPerExperiment = 100;
maxExpPerJob = 10;
maxJobs = floor(totalExp/maxExpPerJob);
betaMin = -3;
betaMax = 2;

% ======================== Data Generation =========================%

[x, gm] = generateRandGMMData(1, n, Ntrain+Ntest, x_mu_max);
wtrue = rand(n,1)*100;

% alpha = trace(gm.Sigma(:,:,1))/n*10.^(linspace(-2,2, totalExp));
alpha = logspace(-3,3, totalExp);
expRange = floor(linspace(0,totalExp-1,maxExpPerJob));

chosenBetas = zeros(1,totalExp); 
mse = zeros(1,totalExp); 
data_likelyhood = zeros(1,totalExp); 

% ============= Experiment Runs at Increasing Alphas ===============%

for exp = 1:totalExp

exp,

% ============= Generate Noisy Data ===============%

[y, x] = noisy_model(n, Ntrain+Ntest, wtrue, x, alpha(exp));
data = segment_data(y, x, Ntrain);

% ============= Use Kfold to optimize L2 Regularizer for Wmap ===============%

[bestBeta, wfit] = runkfoldexperiment_ridgeregression_bestbeta(Ntrain, 10, betaMin, betaMax, betasPerExperiment, data.train, 1);


experiment{exp}.data = data;
chosenBetas(exp) = bestBeta;
chosenWfits{exp} = wfit;

% ============= Evaluate MSE for optimally L2 Regularizer ===============%

[~, total_mse, ~] = calc_MSE_noisy_linear_model(data.test.y,data.test.x,wfit);
mse(exp) = total_mse;

data_likelyhood(exp) = data_log_likelyhood(data.test.y,data.test.x, wfit, 1);
end

figure(1)
subplot(3,1,1), semilogx(alpha, chosenBetas)
title('Chosen weight regularization value beta for noise (z) variance alpha');
xlabel('alpha'),ylabel('beta'),
hold on, subplot(3,1,2), semilogx(alpha, mse)
title('MSE of assumed model vs real data distorted by noise (z) with variance alpha');
xlabel('alpha'),ylabel('mse'),
hold on, subplot(3,1,3), semilogx(alpha, data_likelyhood)
title('log likelyhood of data using assumed model vs alpha');
xlabel('alpha'),ylabel('P(D|{\theta})'),

figure(2), 
subplot(2,1,1),
plot_experiment_data(10, alpha, experiment, chosenWfits)
title(['Model Output @ alpha = ', num2str(alpha(10))]);
xlabel('Data Index'),ylabel('y'), legend('yTrue', 'yModel');

subplot(2,1,2),
plot_experiment_data(80, alpha, experiment, chosenWfits)
title(['Model Output @ alpha = ', num2str(alpha(80))]);
xlabel('Data Index'),ylabel('y'), legend('yTrue', 'yModel');

function plot_experiment_data(exp, alpha, experiment, chosenWfits)
noise_alpha = alpha(exp);
noise_alpha
[pred, total_mse, ~] = calc_MSE_noisy_linear_model(experiment{exp}.data.test.y,experiment{exp}.data.test.x,chosenWfits{exp});
total_mse
plot_data_and_noise_model_pred(experiment{exp}.data.test.y, pred)
end


function [pred, total_mse, mse] = calc_MSE_noisy_linear_model(y,x,wfit)
    N = size(x,2);
    b = [ones(1,N); x];
    pred = wfit'*b;
    mse = (y-wfit'*b).^2;
    total_mse = mean(mse);
end

function [bestBeta, wfit] = runkfoldexperiment_ridgeregression_bestbeta(N, K, betaMin, betaMax, betaCount, dataset, model_noise_variance)
    partitions_idx_start = ceil(linspace(0,N,K+1));
    indPartitionLimits = zeros(K, 2);
    for k = 1:K
        indPartitionLimits(k,:) = [partitions_idx_start(k)+1,partitions_idx_start(k+1)];
    end
    loglikelyood_Beta = zeros(1,betaCount);
    beta = 10.^linspace(betaMin, betaMax, betaCount);
    for betaIdx = 1:betaCount
        loglikelyhood = zeros(1,K); 
        for k= 1:K
            indValidate = indPartitionLimits(k,1):indPartitionLimits(k,2);
            x = dataset.x;
            y = dataset.y;
            xValidate = x(:, indValidate); % Using folk k as validation set
            yValidate = y(:, indValidate);
            if k == 1
                indTrain = indPartitionLimits(k+1,1):N;
            elseif k == K
                indTrain = 1:indPartitionLimits(k-1,2);
            else
                indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
            end
            xTrain = x(:, indTrain); % using all other folds as training set
            yTrain = y(:, indTrain); % using all other folds as training set

            wfit = fit_noisy_linear_model(yTrain, xTrain, beta(betaIdx), model_noise_variance);
%             [~, total_mse, ~] = calc_MSE_noisy_linear_model(yValidate,xValidate,wfit);
%             loglikelyhood(k) = total_mse;

            loglikelyhood(k) = data_log_likelyhood(yValidate,xValidate, wfit, model_noise_variance);
        end
        loglikelyood_Beta(betaIdx) = mean(loglikelyhood);
    end
    [~, bestBetaIdx] = max(loglikelyood_Beta);
%     [~, bestBetaIdx] = min(loglikelyood_Beta);
    bestBeta = beta(bestBetaIdx);
    wfit = fit_noisy_linear_model(dataset.y, dataset.x, bestBeta, model_noise_variance);
end

function plot_data_and_noise_model_pred(y, pred)
    plot(y,'b'), hold on
    plot(pred, 'g'),
end

function data = segment_data(y, x, Ntrain)
    data.train.x = x(:, 1:Ntrain);
    data.train.y = y(:, 1:Ntrain);
    data.test.x = x(:, Ntrain+1:end);
    data.test.y = y(:, Ntrain+1:end);
end

function [y, x] = noisy_model(n, N, wtrue, x, alpha)
z = generateRandWhiteNoise(1, n, N, alpha);
v = generateRandWhiteNoise(1, 1, N, 1);
y = wtrue'*(x+z)+v;
end

function log_likelyhood = data_log_likelyhood(y, x, w, noise_variance)
    N = size(x,2);
    b = [ones(1,N); x];
%     log_likelyhood = sum(log((1/(sqrt(2*pi)*noise_variance))*exp(-(y-w'*b)/(2*(noise_variance)^2))));
    likelyhoods = zeros(1,N);
    mus = w'*b;
    for i = 1:N
        likelyhoods(i) = evalGaussian(y(i), mus(i), noise_variance.^2);
    end
    log_likelyhood = sum(log(likelyhoods));
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
    [n,N] = size(x);
    C = ((2*pi)^n * det(Sigma))^(-1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
end


function wfit = fit_noisy_linear_model(y,x, beta, noise_variance)
    N = size(x,2);
    dim = size(x,1)+1;
    b = [ones(1,N); x];
    y = repmat(y, dim, 1);
    R = b*b';
    Q = sum(y.*b, 2);
    wfit = inv(R+(noise_variance^2/beta).*eye(dim))*Q;
end

function [data, gmdist] = generateRandWhiteNoise(Order, dim, N, alpha)
    sigma = zeros(dim,dim,Order);
    mu = zeros(Order, dim);
    for m = 1:Order
        sigma(:,:,m) = alpha.*eye(dim);
    end
    gmdist = gmdistribution(mu,sigma,ones(1,Order)/Order);
    data = random(gmdist,N)';
end

function [data, gmdist] = generateRandGMMData(Order, dim, N, mu_max)
    sigma = zeros(dim,dim,Order);
    mu = zeros(Order, dim);
    for m = 1:Order
        sigma(:,:,m) = rand(1,1).*eye(dim);
        mu(m, :) = rand(1,dim)*mu_max;
    end
    gmdist = gmdistribution(mu,sigma,ones(1,Order)/Order);
    data = random(gmdist,N)';
end
