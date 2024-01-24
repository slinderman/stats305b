# Generalized Linear Models
    
Logistic regression was a special case of a more general class of models called _generalized linear models_ (GLMs). In a GLM, the conditional distribution $p(Y \mid X)$ is modeled as an exponential family distribution whose mean parameter is a function of $X$. For example, if $Y \in \naturals$, we could model it with a Poisson GLM; if $Y \in \{1,\ldots,K\}$, we could model it as a categorical GLM. It turns out that many of the nice properties of logistic regression carry over to the more general case.

## Model
To construct a generalized linear model with exponential family observations, we set 
\begin{align*}
    \E[y_i \mid \mbx_i] &= f(\mbbeta^\top \mbx_i).
\end{align*}
    
From above, this implies,
\begin{align*}
    \nabla A(\eta_i) &= f(\mbbeta^\top \mbx_i) \\
    \Rightarrow \eta_i &= [\nabla A]^{-1} \big( f(\mbbeta^\top \mbx_i) \big),
\end{align*}
when $\nabla A(\cdot)$ is invertible. (In this case, the exponential family is said to be **minimal**).
    
The **canonical mean function** is $f(\cdot) = \nabla A(\cdot)$ so that $\eta_i = \mbbeta^\top \mbx_i$.
    
The (canonical) **link function** is the inverse of the (canonical) mean function.

    
### Logistic regression revisited
Consider the Bernoulli distribution once more. The gradient of the log normalizer is,
\begin{align*}
    \nabla A(\eta) &= \nabla \log (1 + e^\eta) 
    = \frac{e^\eta}{1+ e^\eta}
\end{align*}
This is the logistic function!

Thus, logistic regression is a Bernoulli GLM with the canonical mean function.


## Canonical case

Canonical mean functions lead to nice math. Consider the log joint probability,
\begin{align*}
    \cL(\mbbeta) 
    &= \sum_{i=1}^n \langle t(y_i), \eta_i \rangle - A(\eta_i)  + c \\
    &= \sum_{i=1}^n \langle t(y_i), \mbbeta^\top \mbx_i \rangle - A(\mbbeta^\top \mbx_i) + c,
\end{align*}
where we have assumed a canonical mean function so $\eta_i = \mbbeta^\top \mbx_i$.

The gradient is,
\begin{align*}
    \nabla \cL(\mbbeta) 
    &= \sum_{i=1}^n \langle t(y_i), \mbx_i \rangle - \langle \nabla A(\mbbeta^\top \mbx_i), \, \mbx_i \rangle\\
    &= \sum_{i=1}^n \langle t(y_i) - \E[t(Y_i)], \, \mbx_i \rangle
\end{align*}

In many cases, $t(y_i) = y_i \in \reals$ so
\begin{align*}
    \nabla \cL(\mbbeta) 
    &= \sum_{i=1}^n (y_i - \hat{y}_i) \mbx_i.
\end{align*}

And in that case the Hessian is
\begin{align*}
    \nabla^2_{\mbbeta} \cL(\mbbeta) 
    &= - \sum_{i=1}^n \nabla^2 A(\mbbeta^\top \mbx_i) \, \mbx_i \mbx_i^\top \\
    &= -\sum_{i=1}^n \Var[t(Y_i)] \, \mbx_i \mbx_i^\top
\end{align*}

Now recall the Newton's method updates, written here in terms of the change in weights,
\begin{align*}
    \Delta \mbbeta &= - [\nabla^2 \cL(\mbbeta)]^{-1} \nabla \cL(\mbbeta) \\
    &= \left[\sum_{i=1}^n \Var[t(y_i)] \, \mbx_i \mbx_i^\top \right]^{-1} \left[\sum_{i=1}^n (y_i - \hat{y}_i) \mbx_i \right]
\end{align*}

Letting $w_i = \Var[t(Y_i)]$,
\begin{align*}
    \Delta \mbbeta &=
    \left[\sum_{i=1}^n w_i \, \mbx_i \mbx_i^\top \right]^{-1} \left[ \sum_{i=1}^n (y_i - \hat{y}_i) \mbx_i \right] \\
    % &= (\mbX^\top \mbW \mbX)^{-1} [\mbX^\top \mbW \mbW^{-1} (\mby - \hat{\mby})] \\
    &= (\mbX^\top \mbW \mbX)^{-1} [\mbX^\top \mbW \hat{\mbz}]
\end{align*}
where $\mbW = \diag([w_1, \ldots, w_i])$ and $\hat{\mbz} = \mbW^{-1} (\mby - \hat{\mby})$. 

This is **iteratively reweighted least squares (IRLS)** with weights $w_i$ and working responses $\hat{z}_i = \frac{y_i - \hat{y}_i}{w_i}$, both of which are functions of the current weights $\mbbeta$. As in logistic regression, the working responses can be seen as linear approximations to the observed responses mapped back through the link function.

## Non-canonical case

When we choose an arbitrary covariance matrix, the expressions are a bit more complex. Let's focus on the case where $t(y_i) = y_i$ for scalar $y_i$, but allow for arbitrary mean function $f$. 
\begin{align*}
    \cL(\mbbeta) 
    &= \sum_{i=1}^n \langle y_i, \eta_i \rangle - A(\eta_i)  + c,
\end{align*}
but now $\eta_i = [\nabla A]^{-1} f(\mbbeta^\top \mbx_i)$. 

The gradient is,
\begin{align*}
    \nabla \cL(\mbbeta) 
    &= \sum_{i=1}^n \left(y_i - \hat{y}_i\right) \frac{\partial \eta_i}{\partial \mbbeta}.
\end{align*}
Applying the inverse function theorem, as above, yields,
\begin{align*}
\frac{\partial \eta_i}{\partial \mbbeta} 
&= \mathrm{Var}[Y]^{-1} \mbx_i = \mbx_i / w_i,
\end{align*}
and
\begin{align*}
    \nabla \cL(\mbbeta) 
    &= \sum_{i=1}^n \left(\frac{y_i - \hat{y}_i}{w_i} \right) f'(\mbbeta^\top \mbx_i) \mbx_i.
\end{align*}

## Deviance and Goodness of Fit

We can perform maximum likelihood estimation via Newton's method, but do the resulting parameters $\hat{\mbbeta}_{\mathsf{MLE}}$ provide a good fit to the data? One way to answer this question is by comparing the fitted model to two reference points: a **saturated** model and a **baseline** model.

For binomial GLMs (including logistic regression) and Poisson GLMs, the saturated model conditions on the mean equaling the observed response. For example, in a Poisson GLM the saturated model is,
\begin{align*}
\log p_{\mathsf{sat}}(\mby) 
&= \sum_{i=1}^n \log \mathrm{Po}(y_i; y_i) \\
&= \sum_{i=1}^n -\log y_i! + y_i \log y_i - y_i.
\end{align*}
The likelihood ratio statistic in this case is 
\begin{align*}
-2 \log \frac{p(\mby \mid \mbX; \hat{\mbbeta})}{p_{\mathsf{sat}}(\mby)} 
&= 2 \sum_{i=1}^n \log \mathrm{Po}(y_i; y_i) - \log \mathrm{Po}(y_i; \hat{\mu}_i)\\
&= 2 \sum_{i=1}^n y_i \log \frac{y_i}{\hat{\mu}_i} + \hat{\mu}_i - y_i \\
&= \sum_{i=1}^n r_{\mathsf{D}}(y_i, \hat{\mu}_i)^2 \\
&\triangleq D(\mby, \hat{\mbmu}),
\end{align*}
where $\hat{\mu}_i = f(\mbx_i^\top \hat{\mbbeta})$ is the predicted mean. 
We recognize the likelihood ratio statistic as the sum of squared deviance residuals! (See the [previous chapter](./04_expfam.md) ) 
In the GLM literature, the sum of deviance residuals is (somewhat confusingly) just called the **deviance**, $D(\mby, \hat{\mbmu})$. Larger deviance implies a poorer fit. 

The baseline model is typically a GLM with only an intercept term, in which case the MLE is $\mu_i \equiv \frac{1}{n} \sum_{i=1}^n y_i = \bar{y}$. For that baseline model, the deviance is,
\begin{align*}
D(\mby, \bar{y} \mbone)
&= 2 \sum_{i=1}^n y_i \log \frac{y_i}{\bar{y}} + \bar{y} - y_i \\
&= 2 \sum_{i=1}^n y_i \log \frac{y_i}{\bar{y}} \\
&= \sum_{i=1}^n r_{\mathsf{D}}(y_i, \bar{y})^2 \\
&= D(\mby, \bar{y} \mbone),
\end{align*}

:::{admonition} Note on intercepts
In models with an (unregularized) intercept term, the MLE should be such that $\sum \mu_i = \sum y_i$, which simplifies the deviance.
:::

### Fraction of Deviance Explained
As with linear models where we use the fraction of variance explaiend ($R^2$), in GLMs we can consider the fraction of _deviance_ explained. Note that the deviance is positive for any model that is not saturated. The **fraction of deviance explained** is $1 - \frac{D(\mby; \hat{\mbmu})}{D(\mby; \bar{y} \mbone)}$. Unless your model is worse than guessing the mean, the fraction of deviance explained is between 0 and 1. 

### Model Comparison 
The difference in deviance between two models with predicted means $\hat{\mbmu}_0$ and $\hat{\mbmu}_1$, the difference in deviance,
\begin{align*}
D(\mby; \hat{\mbmu}_0) - D(\mby; \hat{\mbmu}_1)
\end{align*}
has an approximately chi-squared null distribution. We can use this fact to sequentially add or subtract features from a model depending on whether the change in deviance is significant or not. 

### Model Checking
Just like in standard linear models, we should inspect the residuals in generalized linear models for evidence of model misspecification. For example, we can plot the residuals as a function of $\hat{\mu}_i$ and they should be approximately normal at all levels of $\hat{\mu}_i$.

## Cross-Validation

Finally, another common approach to model selection and comparison is to use cross-validation. The idea is to approximate out-of-sample predictive accuracy by randomly splitting the training data. 

**Leave-one-out cross validation (LOOCV)** withholds one datapoint out at a time, estimates parameters using the other $n-1$, and then evaluates predictive likelihood on the held out datapoint.

This approximates,
\begin{align}
    \E_{p^\star(x, y)}[\log p(y \mid x, \{x_i, y_i\}_{i=1}^n)] &\approx
    \frac{1}{n} \sum_{i=1}^n \log p(y_i \mid x_i, \{x_j, y_j\}_{j \neq i}).
\end{align}
where $p^\star(x, y)$ is the true data generating distribution

For small $n$ (or when using a small number of folds), a bias-correction can be used. 
<!-- (See pg. 175-176 of the book.) -->

Cross-validated predictive log likelihood estimates are similar to the \emph{jackknife} resampling method in the sense that it is estimating an expectation wrt the unknown data-generating distribution $p^\star$ by resampling the given data. 

