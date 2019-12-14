% Data, Inference & Applied Machine Learning
% DIAML: Tuorial 4b
% Copyright (c) 2017 Patrick E. McSharry (patrick@mcsharry.net)

% Kernel Density Estimation
% Generate a mixture of two normal distributions, and plot the
% estimated cumulative distribution at a specified set of values.
N = 100;
x = [-3+randn(N,1); 3+randn(N,1)];
xi = linspace(-10,10,201);
h = 0.4;
f = ksdensity(x,xi,'width',h);
F = ksdensity(x,xi,'width',h,'function','cdf');
figure
subplot(2,1,1)
plot(xi,0.5*pdf('norm',xi,-3,1)+0.5*pdf('norm',xi,3,1),'r')
hold
plot(x,zeros(size(x)),'k.')
plot(xi,f,'b');
ylabel('PDF')
subplot(2,1,2)
plot(xi,F,'b');
ylabel('CDF')
xlabel('x')


% KNN Classification

% Load the Fisher iris data.
load fisheriris
X = meas; % use all data for fitting
Y = species; % response data
varnames = {'SLength','SWidth','PLength','PWidth'};

% Construct the classifier using ClassificationKNN.fit.
mdl = ClassificationKNN.fit(X,Y)

% A default k-nearest neighbor classifier uses just the single nearest neighbor. 
% Often, a classifier is more robust with more neighbors than that. 


% Change the neighborhood size of mdl to 4, meaning mdl classifies using the four nearest neighbors:
mdl = ClassificationKNN.fit(X,Y,'NumNeighbors',4);

% Examine the resubstitution loss, which, by default, is the fraction of misclassifications from the predictions of mdl. (For nondefault cost, weights, or priors, see ClassificationKNN.loss.)
rloss = resubLoss(mdl)
% rloss = 0.0400
% The classifier predicts incorrectly for 4% of the training data.

% Construct a cross-validated classifier from the model.
cvmdl = crossval(mdl);
% Examine the cross-validation loss, which is the average loss of each 
% cross-validation model when predicting on data that is not used for training.

kloss = kfoldLoss(cvmdl)
% kloss = 0.0600
% The cross-validated classification accuracy resembles the resubstitution accuracy. 
% Therefore, you can expect mdl to misclassify approximately 5% of new data, 
% assuming that the new data has about the same distribution as the training data.


% KNN classification error versus number of neighbors
[N,D] = size(X);
K = round(logspace(0,log10(N),10)); % number of neighbors
cvloss = zeros(length(K),1);
for k=1:length(K)
    % Construct a cross-validated classification model
    mdl = ClassificationKNN.fit(X,Y,'NumNeighbors',K(k));
    % Calculate the in-sample loss
    rloss(k)  = resubLoss(mdl);
    % Construct a cross-validated classifier from the model.
    cvmdl = crossval(mdl);
    % Examine the cross-validation loss, which is the average loss of each cross-validation model when predicting on data that is not used for training.
    cvloss(k) = kfoldLoss(cvmdl);
end
[cvlossmin,icvlossmin] = min(cvloss);
kopt = K(icvlossmin);


% plot the accuracy versus k
figure; 
semilogx(K,rloss,'g.-');
hold
semilogx(K,cvloss,'b.-');
plot(K(icvlossmin),cvloss(icvlossmin),'ro')
xlabel('Number of nearest neighbors');
ylabel('Ten-fold classification error');
legend('In-sample','Out-of-sample','Optimum','Location','NorthWest')
title('KNN classification');


% feature selection
% 4 features 
count = 0;
for i=1:4
    ind = nchoosek([1:4],i);
    for j=1:size(ind,1)
        count = count+1;
        featsel{count} = ind(j,:);
    end
end

for i=1:length(featsel)
    for k=1:length(K)
        % Construct a cross-validated classification model
        mdl = ClassificationKNN.fit(X(:,featsel{i}),Y,'NumNeighbors',K(k));
        % Calculate the in-sample loss
        rloss(i,k)  = resubLoss(mdl);
        % Construct a cross-validated classifier from the model.
        cvmdl = crossval(mdl);
        % Examine the cross-validation loss, which is the average loss of each cross-validation model when predicting on data that is not used for training.
        cvloss(i,k) = kfoldLoss(cvmdl);
    end
end

% view the matrix of errors using a heat map
figure
imagesc(cvloss)
xlabel('log_2(Number of nearest neighbors)')
ylabel('Feature set')
colorbar;



% KNN Regression

% To create a regression tree for the carsmall data based on the Horsepower and Weight vectors for data, and MPG vector for response:
load carsmall % contains Horsepower, Weight, MPG
X = [Horsepower Weight];
y = MPG;

% learnig and testing datasets
indl = [1:50];
indt = [51:100];
Xl = X(indl,:);
yl = y(indl,:);
Xt = X(indt,:);
yt = y(indt);
nl = length(yl);
nt = length(yt);


K = 2.^[0:5];
for k=1:length(K)
    k
   [idx, dist] = knnsearch(Xl,Xt,'dist','seuclidean','k',K(k));
   %[idx, dist] = knnsearch(Xl,Xt,'dist','mahalanobis','k',K(k));
   ythat = nanmean(yl(idx),2);
   E = yt - ythat;
   RMSE(k) = sqrt(nanmean(E.^2));
end


figure
plot(K,RMSE,'k.-');
xlabel('Number of nearest neighbors')
ylabel('RMSE')
