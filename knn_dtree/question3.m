% ---------Question 3---------------
f = readtable('titanic3.csv')
% survived = f.survived
selected = f(:,{'pclass','age','sex','survived'})
selected = rmmissing(selected)
features = selected(:,{'pclass','age','sex'})
target = selected(:,{'survived'})

nrow = size(selected,1);
selected.sexval = zeros(nrow, 1);
for k=1:1046
    if strcmp(selected.sex(k),'male')
        selected.sexval(k)=1
    end
end
selected = selected(:,{'pclass','age','sexval'})


% features.pclass = categorical(features.pclass)
[N,D] = size(selected);
K = round(logspace(0,log10(N),10)); % number of neighbors
cvloss = zeros(length(K),1);
for k=1:length(K)
    % Construct a cross-validated classification model
    mdl = ClassificationKNN.fit(selected,target,'NumNeighbors',K(k));
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



model1 = ClassificationKNN.fit(selected,target,'NumNeighbors',5,'distance','euclidean');
cvmodel1 = crossval(model1);
euc_cvloss = kfoldLoss(cvmodel1);
rloss_1  = resubLoss(model1);

model2 = ClassificationKNN.fit(selected,target,'NumNeighbors',5,'distance','mahalanobis');
cvmodel2 = crossval(model2);
mahalanobis_cvloss = kfoldLoss(cvmodel2);
rloss_2  = resubLoss(model2);


model3 = ClassificationKNN.fit(selected,target,'NumNeighbors',5,'distance','chebychev');
cvmodel3 = crossval(model3);
chebychev_cvloss = kfoldLoss(cvmodel3);
rloss_3  = resubLoss(model3);


model4 = ClassificationKNN.fit(selected,target,'NumNeighbors',5,'distance','correlation');
cvmodel4 = crossval(model4);
correlation_cvloss = kfoldLoss(cvmodel4);
rloss_4  = resubLoss(model4);


model5 = ClassificationKNN.fit(selected,target,'NumNeighbors',5,'distance','seuclidean');
cvmodel5 = crossval(model5);
seuc_cvloss = kfoldLoss(cvmodel5);
rloss_5  = resubLoss(model5);

arr = (table2array(selected));
arr = double(arr);
arry = (table2array(target));
arry = categorical(arry);

[B,dev,stats] = mnrfit(arr,arry);

% rloss_lr  = resubLoss(modellr);
% cvmodellr = crossval(modellr);
% lr_cvloss = kfoldLoss(cvmodellr);

lr_predict = predict(mdl,X);

plotconfusion(YTest,YPredicted)
