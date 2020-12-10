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

***Variable selection***: to determine which predictors are associated with the response, in order to fit a single model involving only those predictors.

Statistics that can be used to judge the quality of a model:

* *Mallow's $C_p$*
* *Akaike information criterion (AIC)*
* *Bayesian information criterion (BIC)*
* *Adjusted $R^2$*

If $p$ is small enough, we can consider all $2^p$ models and determine which is best; otherwise, we need an automated and efficient approach to choose a smaller set of models to consider:

* *Forward selection*, which begins with the *null model* and add the variable one by one that results in the lowest $RSS$. After adding one variable, the new model needs to be re-fit.
* *Backward selection*, which starts with all variables in the model and remove the variable one by one that has the largest *p-value*. After removing one variable, the new model needs to be re-fit. **Backward selection cannot be used if $p>n$**.
* *Mixed selection*, which is a combination of forward and backward selection. We start with no variables in the model, and as with forward selection, we add the variable that provides the best fit. If at any point the *p-value* for one of the variables in the model rises above a certain threshold, then we remove that variable from the model. Repeat these forward and backward steps. 

### Three: Model Fit

In multiple linear regression, $R^2$ equals $Cor(Y, \hat{Y})$, the square of the correlation between the response and the fitted linear model.

$R^2$ will always increase when more variables are added to the model, even if those variables are only weakly associated with the response.

However, if adding a variable to the model only leads to just a tiny increase in $R^2$, then the additional variable can be dropped from the model.

In general $RSE$ is defined as:

$$RSE = \sqrt{\frac{1}{n-p-1}RSS}$$

In addition to looking at the $RSE$ and $R^2$ statistics, we can also plot the data to look how well the model can fit.

### Four: Predictions

* The *least squares plane*

  $$\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X_1 + \cdot\cdot\cdot + \hat{\beta}_p X_p$$

  is only an estimate for the *true population regression plane*

  $$f(X) = \beta_0 + \beta_1 X_1 + \cdot\cdot\cdot + \beta_p X_p$$

  The inaccuracy in the coefficient estimates is related to the *reducible error*. We can compute a *confidence interval* to determine how close $\hat{Y}$ will be to $f(X)$.

* A linear model for $f(X)$ is an approximation of reality, which causes an additional source of potentially reducible error called as *model bias*.

* The random error $\epsilon$, called as the *irreducible error*, makes it impossible to predict the response value perfectly even if we knew the true values for $\beta_0, \beta_1, ..., \beta_p$. To measure how much will $Y$ vary from $\hat{Y}$, we use ***prediction intervals***, which are always wider than confident intervals.

# 3.3 Other Considerations in the Regression Model

## 3.3.1 Qualitative Predictors

### Predictors with Only Two Levels

If a qualitative predictor (also known as a *factor*) only has two *levels*, or possible values, incorporating it into a regression model needs to create an indictor or ***dummy variable*** that takes on two possible numerical values. 

### Qualitative Predictors with More than Two Levels

In this situation, we can create additional dummy variables, but there will always be one fewer dummy variable than the number of levels. The level with no dummy variable is known as the ***baseline***.

Using this dummy variable approach presents no difficulties when incorporating both quantitative and qualitative predictors.

## 3.3.2 Extensions of the Linear Model

The standard linear regression model makes several highly restrictive assumptions that are often violated in practice, two of which state that the relationship between the predictors and response are:

* ***Additive*** => the effect of changes in a predictor $X_j$ on the response $Y$ is independent of the values of the other predictors.
* ***linear*** => the change in the response $Y$ due to a one-unit change in $X_j$ is constant , regardless of the value of $X_j$.

### Removing the Additive Assumption

* ***Synergy effect***, or ***interaction effect***

One way of extending the linear model to allow for interaction effects is to include a third predictor, called an ***interaction term***, which is constructed by computing the product of $X_1$ and $X_2$.

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1 X_2 + \epsilon$$

The *hierarchical principle* states that *if we include an interaction in a model, we should also include the main effects, even if the p-values associated with their coefficients are not significant.*

The concept of interactions applies just as well to qualitative variables, or to a combination of quantitative and qualitative variables. An interaction variable can be created by multiplying the quantitative variable with the dummy variable for the qualitative variable.

### Non-linear Relationships

To extend the linear model to accommodate non-linear relationships, we can use ***polynomial regression***, which incorporates non-linear associations in a linear model by including transformed versions of the predictors in the model. The *polynomial regression* is still a **linear model!.**

## 3.3.3 Potential Problems

### Non-linearity of the Data

***Residual plots*** are a useful graphical tool for identifying non-linearity.

* For a simple linear regression model, plot the residuals, $e_i = y_i - \hat{y}_i$, versus the predictors $x_i$;
* For a multiple regression model, plot the residuals versus the predicted (or fitted) values $\hat{y}_i$.

Ideally, the residual plot will show no **discernible pattern**. The presence of a pattern may indicate a problem with some aspects of the linear model.

If the residual plot indicates that there are non-linear associations in the data, a simple approach is to use non-linear transformations of the predictors, such as $log X$, $\sqrt{X}$, and $X^2$, in the regression model.

### Correlation of Error Terms

An important assumption of the linear regression model is that the error terms, $\epsilon_1, \epsilon_2, \cdot\cdot\cdot, \epsilon_n$ are uncorrelated. The standard errors that are computed for the estimated regression coefficients or the fitted values are based on this assumption.

If there is correlation among the error terms, then

* the estimated standard errors will tend to underestimate the true standard errors
* confidence and prediction intervals will be narrower than they should be.
* p-values associated with the model will be lower than they should be.

In short, if the error terms are correlated, we may have an unwarranted sense of confidence in our model.

If the errors are uncorrelated, then there should be no discernible pattern among the adjacent data points in the residuals versus observations plots; otherwise, we may see *tracking* in the residuals. For example, adjacent residuals may have similar values if the error terms are positively correlated.

### Non-constant Variance of Error Terms

