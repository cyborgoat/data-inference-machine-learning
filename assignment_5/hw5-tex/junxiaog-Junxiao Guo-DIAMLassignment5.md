## 1. Statistical learning (25 points)

1.1 Describe at least four steps to implementing a rule-based approach to decision-making and give an example. Is any domain knowledge required to establish a rule? Support your answer with an explanation.

#### **Steps**:

- A list of rules or rule base, which is a specific type of knowledge base.
- An inference engine or semantic reasoner, which infers information or takes action based on the interaction of input and the rule base. The interpreter executes a production system program by performing the following match-resolve-act cycle.
- Temporary working memory.
- A user interface or other connection to the outside world through which input and output signals are received and sent.

#### **Example**: 

1.2 Explain over-fitting and why it is a problem in statistical learning. If you have a small dataset containing ten data points, should you prefer a simple model with one parameter or a complex model with ten parameters? Support your answer with an explanation.

**Overfitting**: In statistics, overfitting is "the production of an analysis that corresponds too closely or exactly to a particular set of data, and may therefore fail to fit additional data or predict future observations reliably". An overfitted model is a statistical model that contains more parameters than can be justified by the data.The essence of overfitting is to have unknowingly extracted some of the residual variation (i.e. the noise) as if that variation represented underlying model structure.

**For small dataset**: If we only have a small dataset of ten data points, we **should use s simple model** because if we use a complex model, it will be easily get overfitted because complex model can easily be trained to fit every data-points from the small dataset, which will lose the generality of our model.



1.3 There are two commonly used approaches to avoid over-fitting; describe each one.

- **Simplifying the model**: make sure that the number of independent parameters in your fit is much smaller than the number of data points you have.  By independent parameters, I mean the number of coefficients in a polynomial or the number of weights and biases in a neural network, not the number of independent variables

- **Adding Regularizations**: Regularization attempts to reduce the variance of the estimator by simplifying it, something that will increase the bias, in such a way that the expected error decreases.

1.4 Provide two examples of metrics used to evaluate the performance of a model and give formula for each one. Give two examples of applications and appropriate metrics for each case.

**F1 Score**
- F1 is an overall measure of a model's accuracy that combines precision and recall, in that weird way that addition and multiplication just mix two ingredients to make a separate dish altogether.
$${F_1}= {(recall^{-1}+precision^{-1})\over{2}}^{-1}=2\cdot{precision\cdot recall\over{precision+recall}}$$

- For the precision and recall for F1 Score: 

$$Precision={TruePositive \over {TruePositive + FalsePostiive}}$$
$$Recall = {TruePositive \over {TruePositive+FalseNegative}}$$



**R-Squared**
- R-squared, also known as the coefficient of determination, is the statistical measurement of the correlation between an investment’s performance and a specific benchmark index. In other words, it shows what degree a stock or portfolio’s performance can be attributed to a benchmark index.
$$R^2=1-{MSE(model)\over{MSE(baseline)}}$$

MSE(model) = Mean Squared Error of the predictions against the actual values
$$MSE(model)=\sum_i^N(y_i-\hat{y})^2$$

MSE(baseline) = Mean Squared Error of  mean prediction against the actual values
$$MSE(model)=\sum_i^N(\bar{y_i}-\hat{y})^2$$



**Example Cases:**
1. Evaluation of named entity recognition and word segmentation: F1 Score
2. Representing how a funds movements correlates with a benchmark index: R-Squred"

1.5 Why are benchmarks useful in machine learning and give two examples.
**Why useful**: 
    1. Benchmark is standard against which you compare the solutions, to get a feel if the solutions are better or worse.
    2. When the benchmarks are “representative,” they allow engineering effort to be focused on a small but high-value and widely used set of targets. In the best cases, benchmarks initiate a virtuous circle, propelling a cycle of optimization and improved value for all members of a community
    
**Examples**:
- For newcomers, benchmarks provide a summary that helps them orient in a maze of new terms and data. 
- For sophisticates, benchmarks provide an easily portable and quick-to-collect baseline, where specific measurements will give more relevant data, and disagreement between the benchmark and the specific measurement suggests that more investigation is needed.

## 2. Machine Learning (25 points)

2.1 What is machine learning? Discuss its evolution over time and why is it popular?

**What is Machine Learning**

Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or infeasible to develop a conventional algorithm for effectively performing the task.

**Evolution overt time**

| Time | Name             | Description                                                                                                                               |
| ---- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 1950 | Alan Turing Test | A machine can actually learn, if when we communicate with it, we cannot distinguish it from another human.                                |
| 1952 | ELIZA            | Arthur Samuel (IBM) wrote the first game-playing program, ELIZA, for checkers, to achieve sufficient skill to challenge a world champion. |
| 1957 |                  |                                                                                                                                           |  |
| 1990 |                  |                                                                                                                                           |  |
| 2010 |                  |                                                                                                                                           |  |
| 2014 |                  |                                                                                                                                           |  |

2.2 Give three examples of machine learning techniques that can be viewed as either supervised or unsupervised approaches.

1. Decision Tree (Supervised)
2. Linear Regression (Supervised)
3. K-means (Unsupervised)

2.3 What is the difference between classification and regression?

- The main difference between them is that the output variable in regression is numerical (or continuous) while that for classification is categorical (or discrete)


2.4 What is the difference between supervised learning and unsupervised learning?

- Supervised: All data is labeled and the algorithms learn to predict the output from the input data. 
- Unsupervised: All data is unlabeled and the algorithms learn to inherent structure from the input data.

2.5 Give examples of successful applications of machine learning and explain what technique is appropriate and what type of learning is involved?

