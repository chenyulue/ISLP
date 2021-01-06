*Resampling methods* involve repeatedly drawing samples from a training set and refitting a model of interest on each sample in order to obtain additional information about the fitted model.

* Cross-validation: estimate the test error associated with a given statistical learning method in order to:
  * evaluate its performance (model assessment), or
  * select the appropriate level of flexibility (model selection).
* Bootstrap: provide a measure of accuracy of a parameter estimate or of a given statistical learning method.

# 5.1 Cross-Validation

* *test error rate*
  * making a mathematical adjustment to the training error rate in order to estimate the test error rate.
  * *holding out* a subset of the training observations from the fitting process, and then applying the statistical learning method to those held out observations.
* *training error rate*

The use of a particular statistical learning method is warranted if it results in a low test error.

## 5.1.1 The Validation Set Approach

It involves randomly dividing the available set of observations into two parts, a *training set* and a *validation set* or *hold-out set*.

The resulting validation set error rate – typically assessed using MSE in the case of a quantitative response – provides an estimate of the test error rate.

Two potential drawbacks:

* The validation estimate of the test error rate can be highly variable.
* The validation set error rate may tend to *overestimate* the test error rate for the model fit on the entire data set.

## 5.1.2 Leave-One-Out Cross-Validation

*Leave-one-out cross-validation* (LOOCV) involves splitting the set of observations into two parts:

* a single observation $(x_1, y_1)$ for the validation set
* the remaining observations $\{(x_2,y_2),...,(x_n, y_n)\}$ for the training set

After fitting on the $n-1$ training observations, compute the $MSE_1$ for the single observation $(x_1, y_1)$ by $MES_1 = (y_1 - \hat{y}_1)^2$ .

Repeat the above procedure $n$ times by selecting another single observation for the validation set by order, and compute the $MSE_i,\; where\;i=2,...n$. 

The LOOCV estimate for the test MSE is the average of these *n* test error estimates:
$$
CV_{(n)} = \frac{1}{n}\sum_{i=1}^nMSE_i\label{ref5.1}\tag{5.1}
$$
The pros and cons:

* Far less bias, tending not to overestimate the test error rate
* performing LOOCV multiple times will always yield the same results: there is no randomness in the training/validation set splits.
* expensive to implement, very time consuming if $n$ is large and if each individual model is slow to fit.

With least squares linear or polynomial regression, the cost of LOOCV is the same as that of a single model fit.
$$
CV_{(n)}=\frac1n\sum_{i=1}^n(\frac{y_i-\hat{y}_i}{1-h_i})^2\label{ref5.2}\tag{5.2}
$$
where $\hat{y}_i$ is the *i*th fitted value from the original least squares fit, and $h_i$ is the leverage.

## 5.1.3 k-Fold Cross-Validation

*k-fold cross-validation* involves randomly dividing the set of observations into *k* groups, or *folds*, of approximately equal size.

* the 1st fold is treated as a validation set, and the remaining *k - 1* folds are used to fit.
* compute the $MSE_1$ on the observation in the held-out fold.

This procedure is repeated *k* times; each time, a different group of observations is treated as a validation set, and we get *k* estimates of the test error, $MSE_1, MSE_2, ..., MSE_k$.

The *k*-fold CV estimate is computed by averaging these values:
$$
CV_{(k)} = \frac1k \sum_{i=1}^k MSE_i \label{ref5.3}\tag{5.3}
$$
In practice, one typically performs *k*-fold CV using $k=5$ or $k=10$.

Advantages:

* consuming less computation
* the bias-variance trade-off

## 5.1.4 Bias-Variance Trade-Off for k-Fold Cross-Validation

*k*-fold CV often gives more accurate estimates of the test error rate than does LOOCV, which has to do with a bias-variance trade-off.

* From the perspective of bias reduction, LOOCV is to be preferred to *k*-fold CV.
* LOOCV has higher variance than does *k*-fold CV with $k<n$.

The test error estimate resulting from LOOCV tends to have higher variance than does the test error estimate resulting from *k*-fold CV.

There is a bias-variance trade-off associated with the choice of *k* in *k*-fold cross-validation. Typically, we use $k=5$ or $k=10$.

## 5.1.5 Cross-Validation on Classification Problems

In this setting, we instead use the number of misclassified observations to quantify test error.

In the classification setting, the LOOCV error rate takes the form:
$$
CV_{(n)} = \frac1n \sum_{i=1}^n Err_i \label{ref5.4}\tag{5.4}
$$
where $Err_i = I(y_i \neq \hat{y}_i)$.

# 5.2 The Bootstrap

The *bootstrap* can be used to quantify the uncertainty associated with a given estimator or statistical learning method.

The *bootstrap* approach allows us to use a computer to emulate the process of obtaining new sample sets, so that we can estimate the variability of $\hat{\alpha}$ without generating additional samples. 

We obtain distinct data sets by repeatedly sampling observations *from the original data set*.

For B different bootstrap data sets, $Z^{*1}, Z^{*2}, ..., Z^{*B}$, we can produce a bootstrap estimate for $\alpha$ respectively, $\hat{\alpha}^{*1}, \hat{\alpha}^{*2}, ..., \hat{\alpha}^{*B}$, and then we can compute the standard error of these bootstrap estimates using the formula:
$$
SE_B(\hat{\alpha}) = \sqrt {\frac{1}{B-1} \sum_{r=1}^B (\hat{\alpha}^{*r} - \frac1B \sum_{r'=1}^B \hat{\alpha}^{*r'})^2} \label{ref5.8}\tag{5.8}
$$
This serves as an estimate of the standard error of $\hat{\alpha}$ estimated from the original data set.

