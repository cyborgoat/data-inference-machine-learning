fred = readtable('winequality-red.csv');
fwhite = readtable('winequality-white.csv');
name = {'fixedAcidity','volatileAcidity','citricAcid','residualSugar' ...
    ,'chlorides','freeSulfurDioxide','totalSulfurDioxide','density','pH','sulphates','alcohol'};
red_features = fred(:,name);
white_features = fwhite(:,name);
red_Target = fred(:,{'quality'});
white_Target = fwhite(:,{'quality'});

X = double(table2array(white_features));
Y = double(table2array(white_Target));
[b1,fitinfo1] = lasso(X,Y,'CV',10,'PredictorNames',name);
lassoPlot(b1,fitinfo1,'PlotType','CV');
lassoPlot(b1,fitinfo1,'PlotType','Lambda','XScale','log')

legend('show')
idxLambdaMinMSE1 = fitinfo1.IndexMinMSE;
minMSEModelPredictors1 = fitinfo1.PredictorNames(b1(:,idxLambdaMinMSE1)~=0);

X2 = double(table2array(red_features));
Y2 = double(table2array(red_Target));
[b2,fitinfo2] = lasso(X2,Y2,'CV',10,'PredictorNames',name);
idxLambdaMinMSE2 = fitinfo2.IndexMinMSE;
minMSEModelPredictors2 = fitinfo2.PredictorNames(b2(:,idxLambdaMinMSE2)~=0)
lassoPlot(b2,fitinfo2,'PlotType','CV');
lassoPlot(b2,fitinfo2,'PlotType','Lambda','XScale','log')

legend('show')


% selectednames1 = {'volatileAcidity','residualSugar' ...
%     ,'chlorides','freeSulfurDioxide','totalSulfurDioxide','density','pH','sulphates','alcohol'}
selectednames1 = {'volatileAcidity','residualSugar' ...
     ,'chlorides','freeSulfurDioxide','totalSulfurDioxide','pH','sulphates','alcohol'};


% KNN Regression
indl = [1:800];
indt = [801:1599];
Xl = X2(indl,:);
yl = Y2(indl,:);
Xt = X2(indt,:);
yt = Y2(indt);
nl = length(yl);
nt = length(yt);


K = 2.^[0:5];
for k=1:length(K)
    k
   %[idx, dist] = knnsearch(Xl,Xt,'dist','seuclidean','k',K(k));
   [idx, dist] = knnsearch(Xl,Xt,'dist','mahalanobis','k',K(k));
   ythat = nanmean(yl(idx),2);
   E = yt - ythat;
   RMSE(k) = sqrt(nanmean(E.^2));
end


figure
plot(K,RMSE,'k.-');
xlabel('Number of nearest neighbors')
ylabel('RMSE')

