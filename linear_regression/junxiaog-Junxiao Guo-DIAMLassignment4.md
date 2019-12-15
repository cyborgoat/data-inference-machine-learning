</br></br></br></br></br></br></br></br>

<center>18785-Assignment 4</center></br>
<center>Name: Junxiao Guo</center></br>
<center>AndrewID: junxiaog</center></br>
<center>Date: 2019/10/13</center></br>
<center>Programming Language: Python</center></br>
<center>Libraries used: csv, pandas, matplotlib, scipy, sklearn, RegscorePy, Datetime</center></br>
<div style="page-break-after: always;"></div>
## Question 1: Linear regression with one explanatory variable
---

- **Regression Model Result**:  $$Y = 0.09324143X + 0.00404784$$
<p align="center">
  <img width="460" height="300" src="../images/hw4_imgs/h4_q1p1.png">
</p>
<center>Fig 1.1 Regression Model for FTSE100 Index and House Monthly Return</center>
- **Correlation Coefficients**: 0.026551295701909897
- **What Does the result tell us**: Since the Correlation Coefficient is very positively small, the result tells us that the FTSE index monthly return does not have a small positive relationship between FTSE100 index.
- **Hypothesis test to back up**: I used 2 Sample T-test to test the null hypothesis, the result of t-statistics is:
  - statistic=0.011406194343330636
  - p-value=0.9909030401555481
  - With low t-statistic and a p-value above significance for test, the null hypothesis is accepted.

<div style="page-break-after: always;"></div>
## Question 2: Linear regression with multiple explanatory variables

---

### a) Calculate the correlation coefficients of the aforementioned variables.
```
********* Correlation coefficients of the variables *********
               Apps    Enroll  Outstate  Top10perc  Top25perc  Grad.Rate
Apps       1.000000  0.846822  0.050159   0.338834   0.351640   0.146755
Enroll     0.846822  1.000000 -0.155477   0.181294   0.226745  -0.022341
Outstate   0.050159 -0.155477  1.000000   0.562331   0.489394   0.571290
Top10perc  0.338834  0.181294  0.562331   1.000000   0.891995   0.494989
Top25perc  0.351640  0.226745  0.489394   0.891995   1.000000   0.477281
Grad.Rate  0.146755 -0.022341  0.571290   0.494989   0.477281   1.000000
```

### b) Considering the graduation rate as dependent variable, use stepwise to build the linear regression model

$$Y = 0.0019X['Outstate'] + 0.2255X['Top25perc'] + 33.0860$$

```
Reult Model:                             OLS Regression Results                            
==============================================================================
Dep. Variable:              Grad.Rate   R-squared:                       0.378
Model:                            OLS   Adj. R-squared:                  0.376
Method:                 Least Squares   F-statistic:                     235.0
Date:                Sun, 13 Oct 2019   Prob (F-statistic):           1.82e-80
Time:                        21:07:53   Log-Likelihood:                -3127.2
No. Observations:                 777   AIC:                             6260.
Df Residuals:                     774   BIC:                             6274.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         33.0860      1.607     20.593      0.000      29.932      36.240
Outstate       0.0019      0.000     13.658      0.000       0.002       0.002
Top25perc      0.2255      0.028      7.995      0.000       0.170       0.281
==============================================================================
Omnibus:                       25.071   Durbin-Watson:                   1.945
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               48.404
Skew:                           0.189   Prob(JB):                     3.08e-11
Kurtosis:                       4.163   Cond. No.                     3.69e+04
==============================================================================
```



### c) Which predictor variables are useful in predicting the graduation rate? Explain how you got those variables.

- By using the P value and cutoff criterion from the step wise model, the useful variables are:

1. Outstate 	(p-value = 1.6289e-68)
2. Top25perc  (p-value = 4.695e-15)

### d) Would the set of predictor variables be useful in predicting the graduation rate if you were to use BIC to select the model? Why? 

Since Lower BIC value indicates lower penalty terms hence a better model. From the resulted BIC value:
```
BIC Top10Perc: 6418
BIC Apps: 6619
BIC Outstate: 6329
BIC Top25Perc: 6435
BIC Enroll: 6636
BIC Outstate + Top10Perc: 6283
BIC Outstate + Apps: 6320
BIC Outstate + Top25Perc: 6274
BIC Outstate + Enroll: 6331
BIC Outstate + Top25Perc + Top10Perc: 6280
BIC Outstate + Top25Perc + Apps: 6279
BIC Outstate + Top25Perc + Enroll: 6281
```

The set of predictor variables ("Outstate + Top25Perc") are **still useful** in predicting in graduation rate if were to use BIC to select the model, becuase it has the lowest BIC value comparing to the others

### e) Compare the accurate of the model using only useful predictors with the one of the model using all five predictors? 
```
mse_5variables: 180.89546889773055
mse_2variables: 183.3690660289117
```
Use 5 predictors have higher accuracy since it has lower Mean Square Error.

### f) Given a set of predictor corresponding to Carnegie Mellon University, what graduation rate value should the most accurate model predict? 

The most accurate model predicts as 89.20112305346848%


<div style="page-break-after: always;"></div>
## Question 3: Open study

### Data Set

[City of Chicago Speed Camera Violations](https://catalog.data.gov/dataset/speed-camera-violations-997eb) from data.gov (from 2014-07-01 to 2019-09-24 ), the data points I'm choosing are the time records and number of speed violations recored by the camera for every corresponding time.

### My Assumption

The monthly average speed violations decreases along with time.


### Mythology

Algorithm used: Linear Regression

**Procedure**
1. Used data from 2014-07 to 2016-03 as the training dataset
2. Used the trained model to predict the speed violations from 2016-01 to 2019-07

### Conclusions


<p align="center">
  <img width="460" height="300" src="../images/hw4_imgs/h4_q3p1.png">
</p>
<center>Fig 3.1 Trained Model using linear Regression</center>
<p align="center">
  <img width="460" height="300" src="../images/hw4_imgs/h4_q3p2.png">
</p>
<center>Fig 3.2 Test Result by Using Trained Model</center>
The trained model mostly correctly reflected the actual result, though for the time period from January 2019 to July 2019, there is a highly non-linear fluctuation of the speed violations, so fo that part the model didn't reflect the true result accurately

To sum up, my assumption holds true since the trend of monthly average speed violations for the city of Chicago decreased along with time.


<div style="page-break-after: always;"></div>
## Question 4

```
Predicted Unemployment Rate by year 2020:
2020-12-31 00:00:00 11.357757386015834%
2019-12-31 00:00:00 11.242362110712065%
2018-12-31 00:00:00 11.127282123045745%
2017-12-31 00:00:00 11.012202135379397%
2016-12-31 00:00:00 10.897122147713048%
2015-12-31 00:00:00 10.78172687240928%
2014-12-31 00:00:00 10.66664688474296%

Actual Unemployment Rate by year 2020:
2020-12-31 00:00:00 3.9739999999999998%
2019-12-31 00:00:00 3.9739999999999998%
2018-12-31 00:00:00 3.984%
2017-12-31 00:00:00 4.25%
2016-12-31 00:00:00 4.775%
2015-12-31 00:00:00 5.275%
2014-12-31 00:00:00 5.9%
```



**Method to evaluate accuracy of the estimate**
- Mean absolute percentage error by comparing with the predicted value and actual value.

- Mean absolute percentage error as percentage = 145.78642239%

<p align="center">
  <img width="460" height="300" src="../images/hw4_imgs/h4_q4p1.png">
</p>
<center>Fig 3.2 Predicted Israel Unemployment Rate</center>