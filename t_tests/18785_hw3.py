import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats

#Q1
hypo_mean = 7725
calorie_stat = [5260, 5470, 5640, 6180, 6390, 6515, 6805, 7515, 7515, 8230, 8770]
sample_mean,sample_std,sample_ste = np.mean(calorie_stat),np.std(calorie_stat),stats.sem(calorie_stat)
# t_stat,pval = stats.ttest_1samp(calorie_stat,ret_mean)
t_stat = (sample_mean - hypo_mean) *1000
dof = len(calorie_stat)-1
pval =1 - stats.t.cdf(calorie_stat,df=dof)

print("Mean: {}  Standard Deviation: {}   Standard Error of the Mean: {}".format(sample_mean,sample_std,sample_ste))
print("Degrees of Freedom: {}   t-statistic: {}   P Value:{}  ".format(dof,t_stat,pval))