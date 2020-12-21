# Introduction

* ***Qualitative*** variables are often referred to as **categorical**. 
* Approaches for predicting qualitative responses are known as ***classification***, and predicting a qualitative response for an observation can be referred to as *classifying* that observation.
* Classification techniques are called as ***classifiers***, three of which are most widely-used, including:
  * *logistic regression*,
  * *linear discriminant analysis*, and
  * *K-nearest neighbors*.

# 4.1 An Overview of Classification

In the classification setting we have a set of training observations $(x_1, y_1),\,...,\,(x_n, y_n)$ that we can use to build a classifier. The classifier usually first predicts the probability of each of the categories of a qualitative variable, as the basis for making the classification.

# 4.2 Why Not Linear Regression?

* In general, there is no natural way to convert a qualitative response variable with more than two levels into a quantitative response that is ready for linear regression.
* For a binary response with a 0/1 coding, regression by least squares does make sense, but if we use linear regression, some of the estimates might be outside the $[0,1]$ interval, making them hard to interpret as probabilities.

# 4.3 Logistic Regression

Logistic regression models the *probability* that $Y$ belongs to a particular category.

## 4.3.1 The logistic Model

*Logistic function*:
$$
p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}\label{ref4.2}\tag{4.2}
$$
To fit the model ($\ref{ref4.2}$), we use a method called ***maximum likelihood***. 

The logistic function will always produce an *S*-shaped curve of this form, and so regardless of the value of $X$, we will obtain a sensible prediction that gives outputs between 0 and 1 for all values of $X$.

A bit of manipulation of ($\ref{ref4.2}$):
$$
\frac{p(X)}{1 - p(X)} = e^{\beta_0 + \beta_1 X}\label{ref4.3}\tag{4.3}
$$
which is called the ***odds***, and can take on any value between 0 and $\infin$. Values of the *odds* close to 0 and $\infin$ indicate very low and very high probabilities respectively.

By taking the logarithm of both sides of ($\ref{ref4.3}$):
$$
log(\frac{p(X)}{1 - p(X)})=\beta_0 + \beta_1 X\label{ref4.4}\tag{4.4}
$$
where the left-hand side is called the ***log-odds*** or ***logit***. According to ($\ref{ref4.4}$), we see that the logistic regression model ($\ref{ref4.2}$) has a *logit* that is linear in $X$.

Regardless of the value of $X$,

* if $\beta_1$ is positive then increasing $X$ will be associated with increasing $p(X)$, and
* if $\beta_1$ is negative then increasing $X$ will be associated with decreasing $p(X)$.

## 4.3.2 Estimating the Regression Coefficients

To estimate the coefficients $\beta_0$ and $\beta_1$, we prefer the *maximum likelihood*, by which we try to find $\hat{\beta}_0$ and $\hat{\beta}_1$ such that plugging these estimates into the model for $p(X)$, given in ($\ref{ref4.2}$), yields a number close to one for all individuals who belongs to a specific category, and a number close to zero for all individuals who does not.

***Likelihood function***:
$$
\mathscr{L}(\beta_0, \beta_1)=\prod_{i:y_i=1} p(x_i)\,\prod_{i':y_{i'}=0} (1 - p(x_{i'}))\label{ref4.5}\tag{4.5}
$$
The estimates $\hat{\beta}_0$ and $\hat{\beta}_1$ are chosen to **maximize** this likelihood function.

For logistic regression, the *z*-statistic associated with $\beta_1$ is equal to $\hat{\beta}_1 / SE(\hat{\beta}_1)$, and so a large (absolute) value of the *z*-statistic indicates evidence against the null hypothesis $H_0:\,\beta_1=0$. 

Similarly, if the *p*-value associated with $\beta_1$ is tiny, we also reject $H_0$.

## 4.3.3 Making Predictions

One can use qualitative predictors with the logistic regression model using the dummy variable approach like that in the linear regression.

## 4.3.4 Multiple Logistic Regression

Multiple logistic model:
$$
log(\frac{p(X)}{1 - p(X)}) = \beta_0 + \beta_1 X_1 +\,\cdot\cdot\cdot\,+\beta_p X_p\label{ref4.6}\tag{4.6}
$$
where $X\,=\,(X_1,\,...\,,X_p)$ are *p* predictors.

As in the linear regression setting, the results obtained using one predictor may be quite different from those obtained using multiple predictors, especially when there is correlation among the predictors. This phenomenon is generally known as *confounding*.

## 4.3.5 Logistic Regression for >2 Response Classes

The two-class logistic regression  models have multiple-class extensions, but in practice they tend not to be used all that often. One of the reason is that ***discriminant analysis*** is popular for multiple-class classification.

# 4.4 Linear Discriminant Analysis

Compared to the logistic regression, which is used to model directly the conditional distribution of the response $Y$ given the predictor(s) $X$, we use the linear discriminant analysis to model the distribution of the predictors $X$ separately in each of the response classes (i.e. given $Y$), and then use Bayes' theorem to flip these around into estimates for $Pr(Y=k|X = x)$.

## 4.4.1 Using Bayes' Theorem for Classification







