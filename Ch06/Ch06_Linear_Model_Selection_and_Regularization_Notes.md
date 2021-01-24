Alternative fitting procedures can yield better *prediction accuracy* and *model interpretability*:

* *Prediction Accuracy*: 

  Provided that the true relationship between the response and the predictors is approximately linear, the least squares estimates will have low bias.

  * $n \gg p$:  the least squares estimates tend to also have low variance.
  * $n \simeq p$:  there can be a lot of variability in the least squares fit, resulting in overfitting and poor predictions on future observations.
  * $p>n$: the variance is *infinite* so the method cannot be used at all.
  
* Model Interpretability:

  *Feature selection* or *variable selection* to exclude irrelevant variables from a multiple regression model.

 Three important classes of methods:

* *Subset Selection*: identifying a subset of the $p$ predictors, and then fitting a model using least squares on the reduced set of variables.
* *Shrinkage*: fitting a model involving all $p$ predictors, and then the estimated coefficients are shrunken towards zero relative to the least squares estimates. Shrinkage (also known as *regularization*) can also perform variable selection.
* *Dimension Reduction*: *projecting* the $p$ predictors into a *M*-dimensional subspace, where $M < p$ by computing *M* different *linear combinations,* or *projections*, of the variables.

# 6.1 Subset Selection

## 6.1.1 Best Subset Selection

**Algorithm 6.1** Best subset selection

1. Let $M_0$ denote the *null model*, which contains no predictors $\Rightarrow$ predict the sample mean for each observation.

2. For $k=1,2,...,p$:

   (a) Fit all $\tbinom{p}{k}$ models that contain exactly $k$ predictors.

   (b) Pick the best among these $\tbinom{p}{k}$ models, and call it $M_k$. Here *best* is defined as having the smallest $RSS$, or equivalently largest $R^2$.

   $\Rightarrow$ choose the models with the lowest *test error* for different subsets of the $p$ predictors

3. Select a single best model from among $M_0, ..., M_p$ using cross-validated prediction error, $C_P$, AIC, BIC, or adjusted $R^2$ $\Rightarrow$ choose a model that has a lowest *test error*.

The best subset selection can apply to other types of models, such as logistic regression, in addition to the least squares regression.

 For the logistic regression, we use the *deviance* instead of $RSS$ to order models, which is negative two times the maximized log-likelihood. So the smaller the deviance, the better the fit.

Best subset selection is computationally infeasible for values of $p$ greater than around 40.

## 6.1.2 Stepwise Selection

For the best subset selection, 

* computational consumption is large, and

* the search space is larger, which can lead to overfitting and high variance of the coefficient estimates

$\Rightarrow$ *stepwise* methods is superior to best subset selection for both of these reasons.

### Forward Stepwise Selection

 **Algorithm 6.2** Forward stepwise selection

1. Let $M_0$ denote the *null* model, which contains no predictors

2. For $k=0,...,p-1$:

   (a) Consider all $p-k$ models that augment  the predictors in $M_k$ with one additional predictors.

   (b) Choose the *best* among these $p-k$ models, and call it $M_{k+1}$, which have smallest $RSS$ or highest $R_2$.

3. Select a single best model from among $M_0, ..., M_p$ using cross-validated prediction error, $C_p$, AIC, BIC, or adjusted $R^2$.

Forward stepwise selection is not guaranteed to find the best possible model out of all $2^p$ models containing subsets of the $p$ predictors.

Forward stepwise selection can be applied even in the high-dimensional setting where $n<p$, in this case, it is possible to construct sub-models $M_0,...,M_{n-1}$ only,  since each sub-models is fit using least squares, which will not yield a unique solution if $p \ge n$.

### Backward Stepwise Selection

**Algorithm 6.3** Backward stepwise selection

1. Let $M_p$ denote the *full* model, which contains all $p$ predictors.

2. For $k=p, p-1, ..., 1$:

   (a) Consider all $k$ models that contain all but one of the predictors in $M_k$, for a total of $k-1$ predictors.

   (b) Choose the *best* among these *k* models, and call it $M_{k-1}$, which have smallest $RSS$ or highest $R^2$.

3. Select a single best model from among $M_0,...,M_p$ using cross-validated prediction error, $C_p$, AIC, BIC, or adjusted $R^2$.

Backward selection requires that the number of samples $n$ is larger than the number of variables $p$ so that the full model can be fit.

In contrast, forward stepwise can be used  even when $n<p$.

### Hybrid Approaches

Hybrid versions of forward and backward stepwise selection are available, in which variables are added to the model sequentially, in analogy to forward selection. After adding each new variable, the method may also remove any variable that no longer provide an improvement in the model fit.

## 6.1.3 Choosing the Optimal Model

### Indirectly estimate by adjustment to the training error

Approaches for *adjusting* the training error for the model size to estimate the test error:

* $C_p$ (for a fitted least squares model containing $d$ predictors)
  $$C_p = \frac{1}{n} (RSS + 2d\hat{\sigma}^2) \label{ref6.2}\tag{6.2}
  $$
  where $\hat{\sigma}^2$ is an estimate of the variance of the error $\epsilon$ associated with each response measurement, which is estimated using the full model containing  all predictors.

  The $C_p$ statistic tends to take on a small value for models with a low test error, so when determining which of a set of models is best, we choose the model with the lowest $C_p$ value.

* AIC criterion (for a large class of models fit by maximum likelihood)
  $$
  AIC = \frac{1}{n\hat{\sigma}^2}(RSS + 2d\hat{\sigma}^2)
  $$
  For least squares models, $C_p$ and AIC are proportional to each other.

* BIC (derived from a Bayesian point of view)
  $$
  BIC = \frac{1}{n\hat{\sigma}^2}(RSS + log(n)d\hat{\sigma}^2) \label{ref6.3} \tag{6.3}
  $$
  Like $C_p$, the BIC will tend to take on a small value for a model with a low test error, and so generally we select the model that has the lowest BIC value.

  Since $log(n)>2$ for any $n>7$, the BIC statistic generally places a heavier penalty on models with many variables, and hence results in the selection of smaller models than $C_p$

* Adjusted $R^2$ (for a least squares model with $d$ variables)
  $$
  Adjusted\;R^2 = 1 - \frac{RSS/(n-d-1)}{TSS/(n-1)} \label{ref6.4} \tag{6.4}
  $$
  A *large* value of adjusted $R^2$ indicates a model with a small test error.

  The adjusted $R^2$ is not as well motivated in statistical theory as AIC, BIC, and $C_p$.

### Directly estimate using Validation and Cross-Validation

Advantages relative to AIC, BIC, $C_p$, and adjusted $R^2$:

* Direct estimate of the test error
* Fewer assumptions about the true underlying model
* Used in a wider range of model selection tasks

*One-standard-error* rule:

1. Calculate the standard error of the estimated test MSE for each model size
2. select the smallest model for which the estimated test error is within one standard error of the lowest point on the curve.

# 6.2 Shrinkage Methods

*Constraining* or *regularizing* the coefficient estimates is known as *shrinking* the coefficient estimates towards zero, which can significantly reduce their variance.

## 6.2.1 Ridge Regression

The ridge regression coefficient estimates $\hat{\beta}^R$ are the values that minimize
$$
\sum_{i=1}^n(y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij}) + \lambda\sum_{j=1}^p\beta_j^2 = RSS + \lambda\sum_{j=1}^p\beta_j^2 \label{ref6.5} \tag{6.5}
$$
where $\lambda \ge 0$ is a *tuning parameter*, and $\lambda\sum_{j}\beta_j^2$ is called a *shrinkage penalty*.

The tuning parameter $\lambda$ serves to control the relative impact of these two terms on the regression coefficient estimates:

* $\lambda=0$, the penalty term has no effect, and ridge regression will produce the least squares estimates.
* $\lambda \rightarrow \infty$, the impact of the shrinkage penalty grows, and the ridge regression coefficient estimates will approach zero.

Ridge regression will produce a different set of coefficient estimate, $\hat{\beta}_{\lambda}^R$, for each value of $\lambda$.

Since $X_j \hat{\beta}_{j,\lambda}^R$ will depend not only on the value of $\lambda$, but also on the scaling of the *j*th predictor, it is best to apply ridge regression after *standardizing the predictors*, using the formula
$$
\widetilde{x}_{ij} = \frac{x_{ij}}{\sqrt{\frac1n \sum_{i=1}^n} (x_{ij} - \bar{x}_j)^2} \label{ref6.6} \tag{6.6}
$$
so that they are all on the same scale. In ( $\ref{ref6.6}$), the denominator is the estimated standard deviation of the *j*th predictor.

### Why does Ridge Regression Improve Over Least Squares?

Ridge regression's advantage over least squares is rooted in the *bias-variance trade-off*.

As $\lambda$ increases, the flexibility of the ridge regression fit decreases, leading to decreased variance but increased bias.

Ridge regression works best in situations where the least squares estimates have high variance, such as $p \simeq n$ or $p > n$.

Ridge regression also has substantial computational advantages over best subset selection, which requires  searching through $2^p$ models.

## 6.2.2 The Lasso

The *lasso* overcomes the disadvantage of ridge regression that the penalty $\lambda \sum \beta_j^2$ will shrink all of the coefficients towards zero but will not set any of them exactly to zero (unless $\lambda = \infty$).

The lasso coefficients, $\hat{\beta}_{\lambda}^L$, minimize the quantity
$$
\sum_{i=1}^n \left(y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij} \right)^2 + \lambda \sum_{j=1}^p \left| \beta_j \right| = RSS + \lambda \sum_{j=1}^p \left| \beta_j \right| \label{ref6.7} \tag{6.7}
$$
The $\ell_1$ norm of a coefficient vector $\beta$ is given by $\left\|\beta\right\|_1 = \sum \left|\beta_j\right|$, that is, the lasso uses an $\ell_1$ penalty instead of an $\ell_2$ penalty.

The $\ell_1$ penalty has the effect of forcing some of the coefficient estimates to be exactly equal to zero when the tuning parameter $\lambda$ is sufficiently large; that is, the lasso performs *variable selection*.

In this setting, we way that the lasso yields *sparse* models – that is, models that involve only a subset of the variables.

### Another Formulation for Ridge Regression and the Lasso

The lasso and ridge regression coefficient estimates solve the problems
$$
\min\limits_{\beta} \left\{ \sum_{i=1}^n \left( y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij}\right)^2\right\} \;subject\;to \sum_{j=1}^p \left| \beta_j \right| \le s \label{ref6.8} \tag{6.8}
$$
and
$$
\min\limits_{\beta} \left\{ \sum_{i=1}^n \left( y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij}\right)^2\right\} \;subject\;to \sum_{j=1}^p \beta_j^2 \le s \label{ref6.9} \tag{6.9}
$$
When we perform the lasso or ridge regression we are trying to find the set of coefficient estimates that lead to the smallest $RSS$, subject to the **constraint** that there is a *budget s* for how large $\sum_{j=1}^p \left| \beta_j \right|$ or $\sum_{j=1}^p \beta_j^2$ can be.

### The Variable Selection Property of the Lasso

The lasso leads to feature selection when $p>2$ due to the sharp corners of the polyhedron or polytope.

### Comparing the Lasso and Ridge Regression

* The lasso has a major advantage over ridge regression, in that it produces simpler and more interpretable models that involve only a subset of the predictors.
* As for prediction accuracy, neither ridge regression nor the lasso will universally dominate the other.
  * The lasso performs better in a setting where a relatively small number of predictors have substantial coefficients, and the remaining predictors have coefficients that are very small or that equal zero.
  * Ridge regression will perform better when the response is a function of many predictors, all with coefficients of roughly equal size.
* A technique such as cross-validation can be used in order to determine which approach is better on a particular data set.

### A Simple Special Case for Ridge Regression and the Lasso

Ridge regression more or less shrinks every dimension of the data by the same proportion, whereas the lasso more or less shrinks all coefficients toward zero by a similar amount, and sufficiently small coefficients are shrunken all the way to zero.

### Bayesian Interpretation for Ridge Regression and the Lasso

From a Bayesian viewpoint, ridge regression and the lasso follow directly from assuming the usual linear model with **normal error**, together with a simple **prior distribution for $\beta$**.

* For ridge regression, it has a Gaussian prior distribution with mean zero and standard deviation a function of $\lambda$.
* For lasso regression, it has a double-exponential (Laplace) distribution with mean zero and scale parameter a function  of $\lambda$.

## 6.2.3 Selecting the Tuning Parameter

Implementing ridge regression and the lasso requires a method for selecting a value for the tuning parameter $\lambda$ in ($\ref{ref6.5}$) and ($\ref{ref6.7}$), or equivalently, the value of the constraint $s$ in ($\ref{ref6.9}$) and ($\ref{ref6.8}$).

We use cross-validation to tackle this problem:

* choose a grid of $\lambda$ values, and compute the cross-validation error for each value of $\lambda$;
* then select the tuning parameter value for which the cross-validation error is smallest;
* finally, the model is re-fit using all of the available observations and the selected value of the tuning parameter.

# 6.3 Dimension Reduction Methods

*Dimension reduction*: *transform* the predictors and then fit a least squares model using the transformed variables.

Let $Z_1, Z_2, ..., Z_M$ represent $M<p$ *linear combinations* of the original $p$ predictors:
$$
Z_m = \sum_{j=1}^p \phi_{jm} X_j \label{ref6.16} \tag{6.16}
$$
for some constants $\phi_{1m}, \phi_{2m},...,\phi_{pm}$, $m=1,2,...,M$. We can then fit the linear regression model
$$
y_i = \theta_0 + \sum_{m=1}^M \theta_m z_{im} + \epsilon_i, \; i=1,2,...,n \label{ref6.17}\tag{6.17}
$$
using least squares.

The term *dimension reduction* means this approach reduces the problem of estimating the $p+1$ coefficients $\beta_0, \beta_1, ..., \beta_p$ to the simpler problem of estimating the $M+1$ coefficients $\theta_0, \theta_1, ..., \theta_M$, where $M < p$.
$$
\sum_{m=1}^M \theta_m z_{im} = \sum_{m=1}^M \theta_m \sum_{j=1}^p \phi_{jm} x_{ij} = \sum_{j=1}^p \sum_{m=1}^M  \theta_m \phi_{jm} x_{ij} = \sum_{j=1}^p \beta_j x_{ij}
$$
where
$$
\beta_j = \sum_{m=1}^M \theta_m \phi_{jm} \label{ref6.18}\tag{6.18}
$$
So dimension reduction serves to constrain the estimated $\beta_j$ coefficients such that they take the form ($\ref{ref6.18}$).

All dimension reduction methods work in two steps:

1. the transformed predictors $Z_1, Z_2, ..., Z_M$ are obtained.
2. the model is fit using these $M$ predictors.

The choice of $Z_1, Z_2, ..., Z_M$, or equivalently, the selection of the $\phi_{jm}$'s, can be achieved in different ways:

* *principal components*
* *partial least squares*

## 6.3.1 Principal Components Regression

*Principal components analysis* (PCA) is an approach used as a dimension reduction for regression.

The *first principal component* direction of the data is that along which the observations *vary the most*. That is, out of every possible *linear combination of predictors* under some constraint, this particular linear combination yields the highest variance.

Another interpretation for PCA: the first principal component vector defines the line that is *as close as possible* to the data.

The first principal component appears to capture most of the information contained in the predictors.

One can construct up to $p$ distinct principal components. They would successively maximize variance, subject to the constraint of being uncorrelated with the preceding components.

If the principal component scores are much closer to zero, it indicates that this component captures far less information.

### The Principal Components Regression Approach

The *principal components regression* (PCR) approach involves constructing the first $M$ principal components, $Z_1, ..., Z_M$, and then using these components as the predictors in a linear regression model that is fit using least squares.

For PCR, we assume that *the directions in which $X_1, ..., X_p$ show the most variation are the directions that are associated with $Y$*. If this assumption underlying PCR holds, then fitting a least squares model to $Z_1, ..., Z_M$ will lead to better results than fitting a least squares model to $X_1, ..., X_p$.

PCR will tend to do well in cases when the first few principal components are sufficient to capture most of the variation in the predictors as well as the relationship with the response.

PCR provides a simple way to perform regression using $M<p$ predictors, but it is *not* a feature selection method. This is because each of the $M$ principal components used in the regression is a linear combination of all $p$ of the *original* features.

In this sense, PCR is more closely related to ridge regression than to the lasso. One can think of ridge regression as a continuous version of PCR.

When performing PCR, we generally recommend *standardizing* each predictor prior to generating the principal components, so that all variables are on the same scale.

However, if the variables are all measured in the same units (say, kilograms, or inches), then one might choose not to standardize them.

## Partial Least Squares



The linear combinations, or *directions*, that best represent the predictors $X_1, ..., X_p$, are identified in an *unsupervised* way.

Consequently, PCR has a drawback: there is no guarantee that the directions that best explain the predictors will also be the best directions to use for predicting the response.

*Partial least squares* (PLS) is a *supervised* alternative to PCR. PLS works almost the same way as PCR, but it identifies the new features in a *supervised* way. PLS makes use of the response $Y$ in order to identify new features that not only approximate the old features well, but also that *are related to the response.*

How the first PLS direction is computed:

1. standardize the $p$ predictors
2. set each $\phi_{j1}$ in ($\ref{ref6.16}$) equal to the coefficient from the simple linear regression of $Y$ onto $X_j$, which means, in computing $Z_1 = \sum_{j=1}^p \phi_{j1}X_j$, PLS places the highest *weight* on the variables that are most strongly related to the response.

To identify the second PLS direction:

1. *adjust* each of the variables for $Z_1$ by regressing each variable on $Z_1$ and taking *residuals*; these residuals can be interpreted as the remaining information that has not been explained by the first PLS direction.
2. compute $Z_2$ using this *orthogonalized* data in exactly the same fashion as $Z_1$ was computed based on the original data.

This iterative approach can be repeated $M$ times to identify multiple PLS components $Z_1, ..., Z_M$. Finally we use least squares to fit a linear model to predict $Y$ using $Z_1, ..., Z_n$ in exactly the same fashion as for PCR.

# 6.4 Considerations in High Dimensions

## 6.4.1 High-Dimensional Data

