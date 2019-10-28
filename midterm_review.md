# 18785 Midterm Review


## Week 1

- Statistics is the “science of making decisions in the face of uncertainty”

## Week2

- When using multiple data sources, we need to ensure that the time stamps are properly aligned
- Similarly inaccuracies in monthly data could cause problems with estimates sensitive to daily timescales

## Week3

- Theoretical Distributions
  - Normal Distribution - Heights
  - Log-norm distribution - Financial Returns
  - Poisson distribution - waiting times
  - Student's t-distribution - Hypothesis test

- Normal Distributions
  - $σ^2$ : Variance, $\mu$: Mean
  - If $x ~ N(0, σ_x^2 )$ and $y ~ N(0, σ_y^2 )$, then $x + y ~ N(0, σ_x^2 +σ_y^2)$
  - Central limit theorem: sum of a large number of IID random variables (with finite mean and variance) is normally distributed.
    - IID: independent and identically distributed
  - Linear Models
    - Normal distributions are preserved by principle of superposition
    - Normally distributed forecast errors: Maximum likelihood gives least squares
    - Useful for calculating prediction intervals
  - Non-Linear models
    - Use of normal distributions neglects possibility of asymmetric distributions
    - Fat tailed distributions imply larger probability of worse case scenarios (risk management)
  - Approx. 68% of values fall between mean and 1 std
  - Approx. 95% of values fall between mean and 2 std
  - Approx. 99.9% of values fall between mean and 3 std
- lognormal distribution
  - It has positive values
  - It creates right-skewed curve
  - Normal distribution cannot be used to model stock prices because it has a negative side, and stock prices cannot fall below zero.
  - Conversely, normal distribution works better when calculating total portfolio returns.
- Skilled Distribution
  - A distribution is negatively skewed if the scores fall toward the higher side of the scale and there are very few low scores.
  - A distribution is positively skewed if the scores fall toward the lower side of the scale and there are very few higher scores.
- Standard Deviation
  - measure of the amount of variation or dispersion of a set of values
  - $$\sqrt{\sum_{i=1}^N(x_i-\bar{x})^2\over{N-1}}$$
- Variance
  -  variance is the expectation of the squared deviation of a random variable from its mean.
  -  $$Var(x)=E[(X-\mu)^2] = E[(X-E[X])^2]$$
- Measure of variability
  - Standard deviation
  - Variance
  - Inter-quartile range (middle 50%)
  - 5% and 95%
  - 2.5% and 97.5%
- Quartile
  - $Q_1=CDF^{-1}(0.25)$
  - $Q_3=CDF^{-1}(0.75)$

## Week3

### Monitoring and evaluation

- M&E is a process that helps improve performance and achieve results. Its goal is to improve current and future management of outputs, outcomes and impact.
  - Evaluation: a systematic and objective examination concerning the relevance, effectiveness, efficiency and impact of activities in the light of specified objectives
  - Monitoring is a continuous assessment that aims at providing all stakeholders with early detailed information on the progress or delay of the ongoing assessed activities.
  - Although evaluations are often retrospective, their purpose is essentially forward looking

### T-test

- A t-test is a type of inferential statistic used to determine if there is a significant difference between the means of two groups, which may be related in certain features.
- Essentially, a t-test allows us to compare the average values of the two data sets and determine if they came from the same population. 
- Mathematically, the t-test takes a sample from each of the two sets and establishes the problem statement by assuming a null hypothesis that the two means are equal. Based on the applicable formulas, certain values are calculated and compared against the standard values, and the assumed null hypothesis is accepted or rejected accordingly.
- If the null hypothesis qualifies to be rejected, it indicates that data readings are strong and are not by chance.
- Calculating a t-test requires three key data values. They include the difference between the mean values from each data set (called the mean difference), the standard deviation of each group, and the number of data values of each group.
- Hypothesis testing aims to qualify how likely an outcome could occur by chance.
- Z-score: Another way to calculate probability
  - $$z=(\bar{x}-\mu_0)\over{\sigma/\sqrt{n}}$$
- P Value: if < than standard threshold, null hypothesis is not true. And vise versa
- Degree of freedom =  $n-1$
- One-tailed test
  - If you are using a significance level of 0.05, a two-tailed test allots half of your alpha to testing the statistical significance in one direction and half of your alpha to testing statistical significance in the other direction.  This means that .025 is in each tail of the distribution of your test statistic. When using a two-tailed test, regardless of the direction of the relationship you hypothesize, you are testing for the possibility of the relationship in both directions.
- Two-tailed test
  - If you are using a significance level of .05, a one-tailed test allots all of your alpha to testing the statistical significance in the one direction of interest.  This means that .05 is in one tail of the distribution of your test statistic. When using a one-tailed test, you are testing for the possibility of the relationship in one direction and completely disregarding the possibility of a relationship in the other direction.

### A/B Testing

- A/B testing is a way to compare two versions of a single variable, typically by testing a subject's response to variant A against variant B, and determining which of the two variants is more effective.
- Confidence Interval
  - A statistical method for calculating a confidence interval around the conversion rate is used for each variation in A/B testing.
- Confidence Rate
  - $$P\plusmn(SE(standard error) * 1.96)$$
  - 1.96: based on 95th percentile of standard normal distribution

## Week5

### Autocorrelation

- Autocorrelation, also known as serial correlation, is the correlation of a signal with a delayed copy of itself as a function of delay. Informally, it is the similarity between observations as a function of the time lag between them.
- correlation coefficient only detects linear relationships

# Week6

### Correlation

- The correlation coefficient is 1 for perfectly correlated variables, -1 for anti-correlation and 0 for no correlation

### Occam's Razor

- The principle states that a theory should rely on as few assumptions as possible, eliminating those that make no difference to the observable predictions of the theory

### Step-wise Variable Selection

- Similar to forward selection
- At each iteration variables which are made obsolete by new additions are removed
- The algorithm stops when nothing new is added or when a term is removed immediately after it was added
- Threshold p values are required for adding a variable (p = p enter ) and for removing a variable (p = p remove )

### AIC & BIC

- AIC and BIC differ by the way they penalize the number of parameters of a model. More precisely, BIC criterion will induce a higher penalization for models with an intricate parametrization in comparison with AIC criterion.

### $R^2$ and correlation

- $R^2$ is Coefficient of determination, it is related to the correlation coefficient
- The coefficient of determination, $R^2$ , measures the proportion of variability in a data set that is accounted for by a statistical model
- Both attempt to quantify how well a linear model fits to a data set
- The further the points are scattered from the line, the smaller is the value of $R^2$
- Adjusted $R^2$ : accounts for the fact that the $R^2$ tends to spuriously increase when extra explanatory variables are added to the model


### Error Evaluations

- Mean Squared Error
  - If the forecast errors are not normally distributed, MSE may give misleading results
  - It represents a measure of forecast performance which is analogous to the least squares parameter estimation technique
- Mean Absolute Error
  - This forecast measure focuses on the magnitude of the errors
  - It is more robust than MSE as the large errors are not squared
  - It is commonly used in wind energy forecasting and may be given as a fraction of the total energy being generated
- MAPE
  - Focusing on the percentage error is useful as a means of standardizing the result
  - It should only be used if the dependent variable is positive definitive
  - This measure is commonly used in energy forecasting