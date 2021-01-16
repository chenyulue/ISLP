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

