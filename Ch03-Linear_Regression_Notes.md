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

The **RSE** is an estimate of the standard deviation of the error term $\epsilon$, namely, it is the average amount that the the response will deviate from the true regression line.

$$RSE = \sqrt{\frac{1}{n-2}RSS} = \sqrt{\frac{1}{n-2} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

The **RSE** is considered a measure of the *lack of fit* of the model ($Y = \beta_0 + \beta_1 X + \epsilon$) to the data. If the RSE is small, we can conclude that the model fits the data very well.

### $R^2$ Statistic

The $R^2$ statistic takes the form of a *proportion*, which means it always takes on a value between 0 and 1, and is independent of the scale of $Y$.

$$R^2 = \frac{TSS - RSS}{TSS} = 1 - \frac{RSS}{TSS]}$$

where $TSS = \sum_{i=1}^n(y_i - \bar{y})^2$ is the *total sum of squares*.

* $TSS$ measures the total variance in the response $Y$ and can be thought of as the amount of variability inherent in the response before the regression is preformed. 

* $RSS$ measures the amount of variability that is left unexplained after performing the regression.
* $TSS-RSS$ measures the amount of variability in the response that is explained (or removed) by performing the regression.
* $R^2$ measures the *proportion of variability in $Y$ that can be explained using $X$*.

The $R^2$ statistic is a measure of the linear relationship between $X$ and $Y$. Unlike the *correlation*, $R^2$ can quantify the association between a larger number of variables.

# 3.2 Multiple Linear Regression

Given $p$ distinct predictors, the  multiple linear regression model is of the form:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdot\cdot\cdot + \beta_p X_p + \epsilon$$

where $X_j$ represents the *j*th predictors and $\beta_j$ quantifies the association between that variable and the response. 

$\beta_j$ is interpreted as the *average* effect on $Y$ of a one unit increase in $X_j$, *holding all other predictors fixed.*

## 3.2.1 Estimating the Regression Coefficients

Given estimates $\hat{\beta}_0, \hat{\beta}_1, ..., \hat{\beta}_p$, we can make predictions using the formula:

$$\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1 + \hat{\beta}_2 x_2 + ... + \hat{\beta}_p x_p$$

Using the same *least squares approach*, we choose $\beta_0, \beta_1, ..., \beta_p$ to minimize the sum of squared residuals:

$$RSS = \sum_{i=1}^n(y_i - \hat{y}_i)^2 = \sum_{i=1}^n(y_i - \hat{\beta}_0 - \hat{\beta}_1 x_{i1} - \hat{\beta}_2 x_{i2} - \cdot\cdot\cdot - \hat{\beta}_p x_{ip})^2$$

The values $\hat{\beta}_0, \hat{\beta}_1, ..., \hat{\beta}_p$ that minimize the $RSS$ are the multiple least squares regression coefficient estimates.

## 3.2.2 Some Import Questions

### One: Is There a Relationship Between the Response and Predictors?

Use a hypothesis test to answer this question as in the simple linear regression setting:

* Null hypothesis, $H_0$: $\beta_1 = \beta_2 = \cdot\cdot\cdot = \beta_p = 0$, versus
* the alternative, $H_a$: at least one $\beta_j$ is non-zero.

This hypothesis test is performed by computing the *F-statistic*:

$$F = \frac{(TSS - RSS)/p}{RSS/(n-p-1)}$$

where $TSS = \sum({y_i - \bar{y})^2}$ and $RSS = \sum(y_i - \hat{y}_i)^2$.

* When there is no relationship between the response and predictors, one would expect the *F-statistic* to take on a value close to 1.
* If $H_a$ is true, we expect $F$ to be greater than 1.
* How large does the F-statistic need to be before we can reject $H_0$ and conclude that there is a relationship?   => Depends on the values of $n$ and $p$
  * When $n$ is large, an F-statistic that is just a little larger than 1 might still provide evidence against $H_0$.
  * A larger F-statistic is needed to reject $H_0$ if $n$ is small.

To test that a particular subset of $q$ of the coefficients are zero, the corresponding null hypothesis is:

$$H_0$$: $\beta_{p-q+1} = \beta_{p-q+2} = \cdot\cdot\cdot = \beta_p = 0$

In this case we fit a second model that uses all the variables *except* those $q$ predictors, and compute the residual sum of squares for that model as $RSS_0$. Then the appropriate F-statistic is:

$$F = \frac{(RSS_0 - RSS)/q}{RSS/(n-p-1)}$$

Note:

* The approach of using an F-statistic to test for any association between the predictors and the response works when $p$ is smaller than $n$.
* If $p > n$ then we cannot even fit the multiple linear regression model using least squares, and the F-statistic cannot be used.

### Two: Deciding on Important Variables

