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