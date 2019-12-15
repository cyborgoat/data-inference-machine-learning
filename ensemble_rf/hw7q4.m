% Q4-2
fred = readtable('winequality-red.csv');
fwhite = readtable('winequality-white.csv');
name = {'fixedAcidity','volatileAcidity','citricAcid','residualSugar' ...
    ,'chlorides','freeSulfurDioxide','totalSulfurDioxide','density','pH','sulphates','alcohol'};
red_features = fred(:,name);
white_features = fwhite(:,name);
red_Target = fred(:,{'quality'});
white_Target = fwhite(:,{'quality'});

X = double(table2array(red_features));
Y = double(table2array(red_Target));


leaf = [5 10 20 50 100];
col = 'rbcmy';
figure
for i=1:length(leaf)
    b = TreeBagger(150,X,Y,'Method','R','OOBPrediction','On',...),...
            'MinLeafSize',leaf(i));
    plot(oobError(b),col(i))
    hold on
end
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Error') 
legend({'5 leaves' '10 leaves' '20 leaves' '50 leaves' '100 leaves'})
hold off

b = TreeBagger(150,X,Y,'OOBVarImp','On','MinLeafSize',5);
figure
bar(b.OOBPermutedVarDeltaError)
xlabel('Feature Index')
ylabel('Out-of-Bag Feature Importance')
xticklabels(name)
xtickangle(90)


boptim = TreeBagger(150,X,Y,'Method','R','OOBPrediction','On',...),...
            'MinLeafSize',5);
plot(oobError(boptim))
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Error') 
title('RF model with 5 leaves')
hold on

% MSE for KNN Regression:  0.5134459036898061
% R^2 for KNN Regression:  0.21221695344876634
% MSE for Linear Regression:  0.5288856404406953
% R^2 for Linear Regression:  0.18852767524407976

