% ---------Question 2---------------
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

f2 = readtable('titanic3.csv');
selected2 = f2(:,{'pclass','age','sex','survived'});
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
selected2.age = double(selected2.pclass);
selected2.sexval = double(selected2.sexval);
selected2 = selected2(:,{'pclass','age','sexval'});
arr = table2array(selected2);
arr = transpose(arr);
targetarr = table2array(target2);
CVMdl = fitrlinear(arr,targetarr,'CrossVal','on','ObservationsIn','columns');
oofYHat = kfoldPredict(CVMdl);
ge = kfoldLoss(CVMdl)

