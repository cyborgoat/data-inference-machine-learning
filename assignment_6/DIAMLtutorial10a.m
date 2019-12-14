% Data, Inference & Applied Machine Learning
% DIAML: Tuorial 10a
% Copyright (c) 2017 Patrick E. McSharry (patrick@mcsharry.net)

% Classification Tree for Fisher's Iris data
load fisheriris;
ctree = ClassificationTree.fit(meas,species);

% view the decision tree via its rules
view(ctree)

% create a visual graphic for the tree
view(ctree,'mode','graph')

% Pruning level can be adjusted directly on the graphic

% in-sample evaluation
resuberror = resubLoss(ctree);
% resuberror = 0.0200
% The tree classifies nearly all the Fisher iris data correctly.

% Cross Validation
% To get a better sense of the predictive accuracy of your tree for new data, cross validate the tree. 
% By default, cross validation splits the training data into 10 parts at random. 
% It trains 10 new trees, each one on nine parts of the data. It then examines the predictive accuracy 
% of each new tree on the data not included in training that tree. This method gives a good estimate 
% of the predictive accuracy of the resulting tree, since it tests the new trees on new data.
cvctree = crossval(ctree);
cvloss = kfoldLoss(cvctree)
% cvloss = 0.0533 is much larger than the in-sample error




% Classification problem with two classes
% Ionosphere data with Y = {'b','g'}
load ionosphere
ctree = ClassificationTree.fit(X,Y)

% in-sample evaluation
resuberror = resubLoss(ctree)

% cross-validation evaluation
cvctree = crossval(ctree);
cvloss = kfoldLoss(cvctree)
% again the cross-validation error is much larger than the in-sample error

% Selecting Appropriate Tree Depth
% Generate minimum leaf occupancies for classification trees from 10 to 100, spaced exponentially apart:
leafs = logspace(1,2,10);

% Create cross validated classification trees for the ionosphere data with minimum leaf occupancies from leafs:
N = numel(leafs);
err = zeros(N,1);
for n=1:N
    t = ClassificationTree.fit(X,Y,'crossval','on',...
        'minleaf',leafs(n));
    err(n) = kfoldLoss(t);
end
plot(leafs,err);
xlabel('Min Leaf Size');
ylabel('cross-validated error');

% The best leaf size is between about 20 and 50 observations per leaf.

% Compare the near-optimal tree with at least 40 observations per leaf with the default tree, 
% which uses 10 observations per parent node and 1 observation per leaf.
DefaultTree = ClassificationTree.fit(X,Y);
view(DefaultTree,'mode','graph')

OptimalTree = ClassificationTree.fit(X,Y,'minleaf',40);
view(OptimalTree,'mode','graph')

% calculate evaluation statistics
resubOpt = resubLoss(OptimalTree);
lossOpt = kfoldLoss(crossval(OptimalTree));
resubDefault = resubLoss(DefaultTree);
lossDefault = kfoldLoss(crossval(DefaultTree));
resubOpt,resubDefault,lossOpt,lossDefault
% the in-sample error is greater for the Optimal than the Default
% the cross-validated error for the Optimal is less than the Default

% Pruning a Classification Tree
% Find the optimal pruning level by minimizing cross-validated loss:
[~,~,~,bestlevel] = cvLoss(ctree,'subtrees','all','treesize','min')

% Prune the tree to use it for other purposes:
pctree = prune(ctree,'Level',6);
view(pctree,'mode','graph')



% Creating a Regression Tree

% To create a regression tree for the carsmall data based on the Horsepower and Weight vectors for data, and MPG vector for response:
load carsmall % contains Horsepower, Weight, MPG
X = [Horsepower Weight];
rtree = RegressionTree.fit(X,MPG)

% view the decision tree via its rules
view(rtree)

% create a visual graphic for the tree
view(rtree,'mode','graph')

% in-sample evaluation
resuberror = resubLoss(rtree)

% cross-validation evaluation
cvrtree = crossval(rtree);
cvloss = kfoldLoss(cvrtree)


% The resubstitution loss for a regression tree is the mean-squared error. 
% The resulting value indicates that a typical predictive error for the tree is about the square root of this.
% The cross-validated loss is approximately five times larger.
% This demonstrates that cross-validated loss is usually higher than simple resubstitution loss.

