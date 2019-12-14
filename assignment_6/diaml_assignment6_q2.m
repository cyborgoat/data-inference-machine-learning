f = readtable('titanic3.csv')
% survived = f.survived
selected = f(:,{'pclass','age','sex','survived'})
selected = rmmissing(selected)
features = selected(:,{'pclass','age','sex'})
target = selected(:,{'survived'})
features.pclass = categorical(features.pclass)
features.sex = categorical(features.sex)
model = fitctree(features,target);
resuberror = resubLoss(model)
cvrtree = crossval(model);
cvloss = kfoldLoss(cvrtree)
cvloss_arr = []
s = 11;

for c = 1:s
    prune_tree = prune(model,'Level',c); 
    cvrtree = crossval(model);
    cvloss = kfoldLoss(cvrtree)
    cvloss_arr = [cvloss_arr,cvloss]
end