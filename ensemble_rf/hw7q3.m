f = readtable('titanic3.csv');
% survived = f.survived
selected = f(:,{'pclass','age','sex','survived'});
selected = rmmissing(selected);
features = selected(:,{'pclass','age','sex'});
target = selected(:,{'survived'});
features.pclass = categorical(features.pclass);
features.sex = categorical(features.sex);


%Tr is the training samples
%cl1 is the class label for the training images
%Ts is the testing samples
%cl2 is the class label for the test images 
Tr = features;
cl1 = target;
Ts = features;
cl2 = target;
nTrees=50;
Mdl = TreeBagger(nTrees,Tr,cl1, 'Method', 'classification','OOBPrediction','on'); 
predChar1 = Mdl.predict(Ts);  % Predictions is a char though. We want it to be a number.
c = str2double(predChar1);

figure;
oobErrorBaggedEnsemble = oobError(Mdl);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';
hold off


%Q3-5

% Logit Regression
selected2 = f(:,{'pclass','age','sex','survived'});
selected2 = rmmissing(selected2);
features2 = selected2(:,{'pclass','age','sex'});
target2 = selected2(:,{'survived'});
nrow = size(selected,1);
selected2.sexval = zeros(nrow, 1);
for k=1:1046
    if strcmp(selected2.sex(k),'male')
        selected2.sexval(k)=1;
    end
end


% Classification Tree
selected2.age = double(selected2.age);
selected2.pclass = double(selected2.pclass);
selected2.sexval = double(selected2.sexval);
selected2 = selected2(:,{'pclass','age','sexval'});
arr = table2array(selected2);
targetarr = table2array(target2);

treemodel = fitctree(arr,targetarr);
[~,score] = resubPredict(treemodel);
diffscore = score(:,2);

[Xtree,Ytree,Ttree,AUCtree] = perfcurve(targetarr,diffscore,1);



logmdl = fitglm(arr,targetarr,'Distribution','binomial','Link','logit');
score_log = logmdl.Fitted.Probability;
[Xlog,Ylog,Tlog,AUClog] = perfcurve(targetarr,score_log,1);

% KNN
knnmdl = ClassificationKNN.fit(arr,targetarr,'NumNeighbors',5);
[~,score] = resubPredict(knnmdl);
diffscoreknn = score(:,2);
[Xknn,Yknn,Tknn,AUCknn] = perfcurve(targetarr,diffscoreknn,1);

% RF
[Yfit,scorerf] = predict(Mdl,Ts);
scorerf = scorerf(:,2)
cl2 = table2array(cl2);
[Xrf,Yrf,Trf,AUCrf] = perfcurve(cl2,scorerf,1);



plot(Xlog,Ylog)
hold on
plot(Xtree,Ytree)
hold on
plot(Xknn,Yknn)
hold on
plot(Xrf,Yrf)
hold off
legend('Logistic Regression','Classification Tree','KNN','Random Forest')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Logistic Regression, Classification Tree, KNN and Random Forest')
hold off