% Data, Inference & Applied Machine Learning
% DIAML: Tuorial 11b
% Copyright (c) 2018 Patrick E. McSharry (patrick@mcsharry.net)

%%
% Simple Dendrogram with random data and 10 elements
% Generate sample data.
% rng('default') % For reproducibility
X = rand(10,2);
% Create a hierarchical binary cluster tree using linkage. Then, plot the dendrogram using the default options.
tree = linkage(X,'average');
figure()
dendrogram(tree)
%
% Dendrogram with random data and 100 elements
% Generate sample data.
rng('default') % For reproducibility
X = rand(100,2);
% There are 100 data points in the original data set, X.
% Create a hierarchical binary cluster tree using linkage. 
% Then, plot the dendrogram for the complete tree (100 leaf nodes) by setting the input argument P equal to 0.
tree = linkage(X,'average');

figure()
dendrogram(tree,0)

% Plot the dendrogram with only 25 leaf nodes. Return the mapping of the original data points to the leaf nodes shown in the plot.
figure()
[~,T] = dendrogram(tree,25);
% List the original data points that are in leaf node 7 of the dendrogram plot.
find(T==7)

% Horizontal Dendrogram
% Generate sample data.
rng('default') % For reproducibility
X = rand(10,3);
% Create a hierarchical binary cluster tree using linkage. 
% Plot the dendrogram with a vertical orientation, using the default color threshold. 
% Return handles to the lines so you can change the dendrogram line widths.

tree = linkage(X,'average');
figure()
H = dendrogram(tree,'Orientation','left','ColorThreshold','default');
set(H,'LineWidth',2)


% Hierarchical Clustering using dendrogram
% Randomly generate the sample data.
rng default; % For reproducibility
X = [randn(100,2)*0.75+ones(100,2);
    randn(100,2)*0.5-ones(100,2)];

tree = linkage(X,'average');
figure
dendrogram(tree,0)

% Select two clusters and plot
c = cluster(tree,'maxclust',2);
figure
scatter(X(:,1),X(:,2),10,c)


% Clustering using k-means
% Randomly generate the sample data.
rng default; % For reproducibility
X = [randn(100,2)*0.75+ones(100,2);
    randn(100,2)*0.5-ones(100,2)];

figure;
plot(X(:,1),X(:,2),'.');
title 'Randomly Generated Data';

% There appears to be two clusters in the data.
% Partition the data into two clusters, and choose the best arrangement out of five intializations. 
% Display the final output.

opts = statset('Display','final');
[idx,C] = kmeans(X,2,'Distance','cityblock','Replicates',5,'Options',opts);
% By default, the software initializes the replicates separatly using k-means++.

% Plot the clusters and the cluster centroids.
figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx','MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Centroids','Location','NW')
title 'Cluster Assignments and Centroids'
hold off


% Clustering Using Gaussian Mixture Distributions
% Gaussian mixture distributions can be used for clustering data, by realizing that the multivariate normal components of the fitted model can represent clusters.
% 
% To demonstrate the process, first generate some simulated data from a mixture of two bivariate Gaussian distributions using the mvnrnd function.
rng default;  % For reproducibility
mu1 = [1 2];
sigma1 = [3 .2; .2 2];
mu2 = [-1 -2];
sigma2 = [2 0; 0 1];
X = [mvnrnd(mu1,sigma1,200);mvnrnd(mu2,sigma2,100)];
scatter(X(:,1),X(:,2),10,'ko')


% Fit a two-component Gaussian mixture distribution. Here, you know the correct number of components to use. 
% In practice, with real data, this decision would require comparing models with different numbers of components.
% For Matlab 2015
%options = statset('Display','final');
%gm = fitgmdist(X,2,'Options',options);
gm = gmdistribution.fit(X,2);

% Plot the estimated probability density contours for the two-component mixture distribution. The two bivariate normal components overlap, but their peaks are distinct. This suggests that the data could reasonably be divided into two clusters.
hold on
ezcontour(@(x,y)pdf(gm,[x y]),[-8 6],[-8 6]);
hold off

% Partition the data into clusters using the cluster method for the fitted mixture distribution. 
% The cluster method assigns each point to one of the two components in the mixture distribution.
idx = cluster(gm,X);
cluster1 = (idx == 1);
cluster2 = (idx == 2);

scatter(X(cluster1,1),X(cluster1,2),10,'r+');
hold on
scatter(X(cluster2,1),X(cluster2,2),10,'bo');
hold off
legend('Cluster 1','Cluster 2','Location','NW')


% Each cluster corresponds to one of the bivariate normal components in the mixture distribution. 
% cluster assigns points to clusters based on the estimated posterior probability that a point came from a component. 
% E ach point is assigned to the cluster corresponding to the highest posterior probability. 
% The posterior method returns those posterior probabilities. 
% For example, plot the posterior probability of the first component for each point.
P = posterior(gm,X);

scatter(X(cluster1,1),X(cluster1,2),10,P(cluster1,1),'+')
hold on
scatter(X(cluster2,1),X(cluster2,2),10,P(cluster2,1),'o')
hold off
legend('Cluster 1','Cluster 2','Location','NW')
clrmap = jet(80); colormap(clrmap(9:72,:))
ylabel(colorbar,'Component 1 Posterior Probability')