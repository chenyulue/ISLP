***Linear regression*** is a very simple approach for supervised learning, which is particularly useful for predicting a quantitative response.

# 3.1 Simple Linear Regression

***Simple linear regression*** can be expressed as:

$$Y \approx \beta_0 + \beta_1 X$$

where $\beta_0$ and $\beta_1$ are the ***intercept*** and ***slope*** terms in the linear model, which are known as the model ***coefficients*** or ***parameters***.

After using the training data to produce estimates $\hat{\beta_0}$ and $\hat{\beta_1}$ for the model coefficients, we get the following equation to predict the response:

$$\hat{y} = \hat{\beta_0} + \hat{\beta_1} x$$

where $\hat{y}$ indicates a prediction of $Y$ on the basis of $X = x$.

## 3.1.1 Estimating the Coefficients

To estimate the coefficients, we want to find an intercept $\hat{\beta_0}$ and $\hat{\beta_1}$ such that the linear model is as close as possible to the real data set. The most common approach to measure *closeness* involves minimizing the ***least squares*** criterion.

***Residual sum of squares*** (**RSS**) is used to denote the closeness:

$$RSS = \sum_{i=1}^n {e_i}^2 = \sum_{i=1}^n (y_i - \hat{y_i})^2 = \sum_{i=1}^n (y_i - \hat{\beta_0}-\hat{\beta_1}x_i)^2$$

where $e_i$ represents the *i*th *residual*.

The least squares approach chooses $\hat{\beta_0}$ and $\hat{\beta_1}$ to minimize the $RSS$. The *least squares coefficient estimates* (the minimizers) for simple linear regression are:

$$\hat{\beta_1} = \frac{\sum_{i=1}^n (x_i - \overline{x})(y_i - \overline{y})}{\sum_{i=1}^n (x_i - \overline{x})^2}$$,

$$\hat{\beta_0} = \overline{y} - \hat{\beta_1} \overline{x}$$

where $\overline{y} = \frac{1}{n}\sum_{i=1}^n y_i$ and $\overline{x} = \frac{1}{n}\sum_{i=1}^n x_i$ are the sample means.

## 3.1.2 Assessing the Accuracy of the Coefficient Estimates

If $f$ is to be approximated by a linear function, we can write this relationship as 

$$Y = \beta_0 + \beta_1 X + \epsilon$$

where $\epsilon$ is a mean-zero random error term, which is used as a catch-all for what we miss with this simple model:

* the true relationship is probably not linear
* there may be other variables that cause variation in $Y$
* there may be measurement error
* ...

and typically we assume that the error term $\epsilon$ is **independent** of $X$. The model shown above defines the ***population regression line***, which is the best linear approximation to the true relationship between $X$ and $Y$ but ***unobserved***.  The least squares regression coefficient estimates $\hat{\beta_0}$ and $\hat{\beta_1}$ characterize the ***least squares line***.

Note the difference between the *population regression line* and the *least squares line*:

* Different data sets generated from the same true model result in slightly different least squares lines, but
* the unobserved population regression line does not change.

The above difference is related to the concept of *bias*, which means for one particular set of observations $y_1, y_2..., y_n$, the model coefficients estimates $\hat{\beta_0}$ and $\hat{\beta_1}$ obtained from the data set might overestimate the true coefficients $\beta_0$ and $\beta_1$, and for another particular data set, the estimates might underestimate the true coefficients. 

***Standard error***, which expresses how far off the single estimate of the model coefficients estimates will be from the true coefficients, are denoted as the following formulas:

$$Var(\hat{\beta_0}) = SE(\hat{\beta_0})^2 = \delta^2 [\frac{1}{n} + \frac{\overline{x}^2}{\sum_{i=1}^n(x_i - \overline{x})^2}]$$

$$Var(\hat{\beta_1}) = SE(\hat{\beta_0})^2 = \frac{\delta^2}{\sum_{i=1}^n(x_i - \overline{x})^2}$$

where $\delta$ is the standard deviation of each of the realizations $y_i$ of $Y$,  $\delta^2 = Var(\epsilon)$,  and the errors $\epsilon_i$ for each observation are assumed to be uncorrelated with common variance $\delta^2$ for these formulas to be strictly valid. 

From these formulas we can tell that:

* the more observations we have, that is, the larger the $n$, the smaller the standard error of the coefficients estimates,
* $SE(\hat{\beta_0})$ is smaller when the $x_i$  are more spread out

In general, $\delta^2$ is not known, but can be estimated from the data. The estimate of $\delta$ is the ***residual standard error***, which is given by the formula:

$$RSE = \sqrt{RSS/(n-2)}$$

* Standard errors can be used to compute ***confidence intervals***.
  * $\hat{\beta_1} \pm 2 \cdot SE(\hat{\beta_1})$
  * $\hat{\beta_0} \pm 2 \cdot SE(\hat{\beta_0})$
* Standard errors can also be used to perform ***hypothesis tests*** on the coefficients.
  * Test the *null hypothesis* of $H_0$(there is no relationship between $X$ and $Y$, which means $\beta_1 = 0$) versus the *alternative hypothesis* of $H_a$(there is some relationship between $X$ and $Y$, which means $\beta_1 \neq 0$).
  * Determine whether $\hat{\beta_1}$, our estimate for $\beta_1$, is sufficiently far from zero that we can be confident that $\beta_1$ is non-zero, which depends on $SE(\hat{\beta_1})$.
  * ***t-statistic***, given by:
    * $$t = \frac{\hat{\beta_1} - 0}{SE(\hat{\beta_1})}$$, which measures the number of standard deviations that $\hat{\beta_1}$ is away from 0, and has a *t*-distribution with $n-2$ degrees of freedom if no relationship exits between $X$ and $Y$.
  * ***p-value***, the probability of observing any number equal to $|t|$ or larger in absolute value, assuming $\beta_1 = 0$.
    * a small *p-value*, which means an association between the predictor and the response.
    * typical *p-value* cutoffs for *rejecting the null hypothesis* are 5 or 1%.

## 3.1.3 Assessing the Accuracy of the Model

To quantify *the extent to which the model fits the data*, we use two related quantities:

* the ***residual standard error*** (**RSE**)
* the **$R^2$ statistic**

### Residual Standard Error

