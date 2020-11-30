# 2.1 What Is Statistical Learning?

Statistical learning refers to a set of approaches for estimating $f$ in the following formula:

$$Y = f(X) + \epsilon$$

where 

* $Y$ is an *output variable*, which can also called as a *response* or *dependent variable*; 
* $X=(X_1, X_2, ..., X_p)$ are *input variables*, which can go by different names such as *predictors*, *independent variables*, *features*, or sometimes just *variables*; 
* $f$ is some fixed but **unknown** function of $X_1, ..., X_p$, which represents the *systematic information* that $X$ provides about $Y$; and 
* $\epsilon$ is a random *error term*, which is **independent of $X$** and has **mean zero**.   

## 2.1.1 Why Estimate $f$?

* Prediction
* Inference

### Prediction

* For situations where a set of inputs $X$ are readily available, but the output $Y$ cannot be easily obtained.

* $\hat{Y} = \hat{f}(X)$, where $\hat{f}$ represents the estimate for $f$, and $\hat{Y}$ the resulting prediction for $Y$.

* Factors that affect the accuracy of $\hat{Y}$:

  * Reducible error  => describes how perfect the estimate is for $f$ and can be reduced by using the most appropriate statistical learning technique to estimate $f$
  * Irreducible error  => comes from the term $\epsilon$, which cannot be predicted by using $X$. 

* $E(Y - \hat{Y})^2 = E[f(X) + \epsilon - \hat{f}(X)]^2  = \underbrace{[f(X) - \hat{f}(X)]^2}_\text{Reducible} + \underbrace{Var(\epsilon)}_\text{Irreducible}$

  * where $E(Y - \hat{Y})^2$ represents the average, or *expected value*, of the squared difference between the predicted and actual value of $Y$, and $Var(\epsilon)$ the *variance* associated with the error term $\epsilon$
  * Estimating $f$ with the aim of minimizing the *reducible error*
  * Irreducible error provides an upper bound on the accuracy of the prediction for $Y$

### Inference


Understand the relationship between $X$ and $Y$, or more specifically, to understand how $Y$ changes as a function of $X_1, X_2, ..., X_p$
* Identify the *important* predictors that are associated with the response $Y$
* Identify the relationship between the response and each predictor
* Summarize the relationship between $Y$ and each predictor adequately using a linear equation, or a more complicated formula

## 2.1.2 How do We Estimate $f$?

We usually use *training data* to train, or teach, our method how to estimate $f$.

*training data*: $\{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$, where $x_i = (x_{i1}, x_{i2}, ..., x_{ip})^T$

* Parametric Methods: reduce the problem of estimating $f$ down to one of estimating a set of parameters
  * steps:
    * Make an assumption about the functional form, or shape, of $f$.
    * Choose a procedure that uses the training data to *fit* or *train* the model.
  * advantages: easier to estimate a set of parameters
  * disadvantages: bad model that is too far from the true $f$
  * *flexible* models => *overfitting* (following the error, or *noise*, too closely)
* Non-parametric methods: do not make explicit assumptions about the functional form of $f$
  * advantages: have the potential to accurately fit a wider range of possible shapes for $f$.
  * disadvantages: require a very large number of observations

## 2.1.3 The Trade-Off Between Prediction Accuracy and Model Interpretability

Usually,

* a less flexible or more restrictive method can be more interpretable, which is usually suitable for the inference setting;
* a more flexible or less restrictive method can be more accurate for prediction.

But because of the overfitting in highly flexible methods, sometimes we will obtain more accurate predictions using a less flexible method.

## 2.1.4 Supervised Versus Unsupervised Learning

* Supervised Learning: for each observation of the predictor measurements $x_i, i = 1,...,n$, there is an associated response measurement $y_i$, which we use to fit a model that relates the response to the predictors.
  * many fitting model: linear regression, logistic regression, GAM, boosting, SVM
* Unsupervised Learning: for every observation $i = 1,...,n$, we only observe a vector of measurements $x_i$, but no associated response $y_i$, in which we are in some sense working blind because of lacking a response variable that can supervise our analysis.
  * learning tools: cluster analysis
* Semi-supervised Learning: for one part of the observations, there are corresponding response variables, but for another, there are no response measurements.  => Beyond the scope of this book

## 2.1.5 Regression Versus Classification Problems

Variables can be characterized as either

* quantitative, which take on numerical values, or
* qualitative (also known as *categorical*), which take on values in one of **K** different *classes*, or categories.

We tend to refer to problems with a quantitative response as

* *regression* problems, 

while those involving a qualitative response are often referred to as

* *classification* problems

# 2.2 Assessing Model Accuracy

*There is no free lunch in statistics:* no one method dominates all others over al possible data sets. So for any given set of data, we need to decide which method produces the best results.

## 2.2.1 Measuring the Quality of Fit

*Mean squared error*(MSE) is used to measure how well the predicted response value actually matches the true observed response value:

$$MSE = \frac{1}{n}\sum\limits_{i=1}^n(y_i - \hat{f}(x_i))^2$$

where $\hat{f}(x_i)$ is the prediction that $\hat{f}$ gives for the *i*th observation.

* *training MSE*
* ***test*** *MSE*, which should be as small as possible, because *we are interested in the accuracy of the predictions that we obtain when we apply our method to previously unseen test data $(x_0, y_0)$ not used to train the statistical learning method.*
  * $$Ave(y_0 - \hat{f}(x_0))^2$$

Note: **there is no guarantee that the method with the lowest training MSE will also have the lowest test MSE.**

A quantity that summarizes the flexibility of a curve is called the *degrees of freedom*.

As the flexibility of the statistical learning method increases, we observe a monotone decrease in the training MSE and a *U-shape* in the test MSE, with the training MSE almost always to be smaller than the test MSE . 

A good estimated $\hat{f}$ has a minimum test MSE, but the lowest achievable test MSE among all possible methods can only be very close to the irreducible error $Var(\epsilon)$.  One important approach used to estimate this minimum test MSE is *cross-validation*, which uses the training data to estimate test MSE.

When a given method yields a small training MSE but a large test MSE, we are said to be *overfitting* the data.

## 2.2.2 The Bias-Variance Trade-Off

The expected test MSE, for a given value $x_0$, can always be decomposed into the sum of three fundamental quantities: the *variance* of $\hat{f}(x_0)$, the squared *bias* of $\hat{f}(x_0)$ and the variance of the error terms $\epsilon$:

$$E(y_0 - \hat{f}(x_0))^2 = Var(\hat{f}(x_0)) + [Bias(\hat{f}(x_0))]^2 + Var(\epsilon)$$

where $E(y_0 - \hat{f}(x_0))^2$ is the *expected test MSE*.

* **Variance** refers to the amount by which $\hat{f}$ would change if we estimated it using a different training data set. Ideally, the estimate for $f$ should not vary too much between training sets.
* **Bias** refers to the error that is introduced by approximating a real-life problem, which may be extremely complicated, by a much simpler model. For example, in the case where the true $f$ is substantially non-linear, using linear regression will always produce an inaccurate estimate, since the model we use doesn't represent the true model.

As a general rule, as we use more flexible methods, the variance will increase and the bias will decrease. The relationship between bias, variance, and the test set MSE is referred to as the *bias-variance trade-off*, which we should always keep in mind.

## 2.2.3 The Classification Setting

The most common approach for quantifying the accuracy of our estimate $f$ in the classification setting is the training *error rate*, the proportion of mistakes that are made if we apply our estimate $\hat{f}$ to the training observations:

$$\frac{1}{n}\sum_{i=1}^{n}I(y_i \neq \hat{y}_i)$$

where $\hat{y}_i$ is the predicted class label for the *i*th observation using $\hat{f}$, and $I(y_i \neq \hat{y}_i)$ is an *indicator variable* that equals 1 if $y_i \neq \hat{y}_i$ and zero if $y_i = \hat{y}_i$, which computes the fraction of incorrect classifications.

As in the regression setting, we are most interested in the *test error rate* associated with a set of test observations of the form $(x_0, y_0)$:

$$Ave(I(y_0 \neq \hat{y}_0))$$

where $\hat{y}_0$ is the predicted class label that results from applying the classifier to the test observation with predictor, $x_0$.

A *good* classifier is one for which the *test error* is smallest.

### The Bayes Classifier

The test error rate is minimized, on average, by a very simple classifier that *assigns each observation to the most likely class, given its predictor values*. That is, when the *conditional probability* $Pr(Y = j | X = x_0)$ is largest, we say that predictor vector $x_0$ is of  class *j*. This classifier is called the *Bayes classifier*. 

The points of which the conditional probability is exactly 50% form the *Bayes decision boundary*.

The Bayes classifier produces the lowest possible test error rate, called the *Bayes error rate*, which is given by

$$ 1 - E(\mathop{max}\limits_{j} Pr(Y = j | X = x_0))$$

where the expectation averages the probability over all possible values of *X*, and the Bayes error rate is analogous to the irreducible error.

### K-Nearest Neighbors

For real data, computing the Bayes classifier is impossible. To estimate the conditional distribution of $Y$ given $X$, one such method is the *K-nearest neighbors (KNN)* classifier, which is given as

$$Pr(Y=j|X=x_0) = \frac{1}{K}\sum\limits_{i\in{N_0}}I(y_i=j)$$

The choice of $K$ has a drastic effect on the KNN classifier obtained. In general, as $K$ grows, the method becomes less flexible, which corresponds to a low-variance but high-bias classifier. As in the regression setting, as the flexibility increases (that is, $K$ decreases), the training error rate consistently declines, but the test error exhibits a characteristic U-shape, declining at first before increasing again when the method becomes excessively flexible and overfits.



