load BlueChipStockMoments

mret = MarketMean;
mrsk = sqrt(MarketVar);
cret = CashMean;
crsk = sqrt(CashVar);

% Q1-3
Corr = corrcov(AssetCovar);
[coeff,latent,explained] = pcacov(Corr);
% [coeff,latent,explained] = pca(Corr);

bar(coeff(:,1:2));
legend('Component 1','Component 2');
xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30])
xticklabels(AssetList);
xtickangle(90)


% Q1-4
Var1 = var(coeff(:,1));
Var2 = var(coeff(:,2));

figure();
pareto(explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');

% Q1-5
scatter(coeff(:,1),coeff(:,2));
xlabel('First Principal Component');
ylabel('Second Principal Comoponent');
% legend('Component 1','Component 2');
avg = mean(coeff(:,1:2));
distances = sqrt((coeff(:,1)-avg(1)).^2+ ((coeff(:,1)-avg(2)).^2));
[top_elements,top_idxs] = maxk(distances,3);