# Logistic Regression

[One, two, many...](https://www.science.org/doi/10.1126/science.1094492) We started with basic discrete distributions for single random variables, and then we modeled pairs of categorical variables with continency tables. Here, we build models for predicting categorical responses given several explanatory variables.

## Setup

Let $Y \in \{0,1\}$ denote a binary response and let $\mbX \in \reals^p$ denote associated covariates. For example, $Y$ could denote whether or not your favorite football team wins their match, and $X$ could represent features of the match like whether its a home or away game, who their opponent is, etc. We will model the conditional distribution of $Y$ given the covariates,
\begin{align*}
Y \mid \mbX = \mbx &\sim \mathrm{Bern}(\pi(\mbx))
\end{align*}
where
\begin{align*}
\pi(\mbx) &= \Pr(Y = 1 \mid \mbX=\mbx) = \E[Y \mid \mbX=\mbx].
\end{align*}

This is a standard regression setup. The modeling problem boils down to choosing the functional form of $\pi(\mbx)$. 

## Linear Regression
If you took STATS 305A, you know pretty much everything there is to know about linear regression with continuous response variables, $Y \in \reals$. Why don't we just apply that same model to binary responses? Specifically, let,
\begin{align*}
\pi(\mbx) &= \mbbeta^\top \mbx = \sum_{j=1}^p \beta_j x_j. 
\end{align*}

Then we'll just use ordinary least squares (OLS) to estimate $\hat{\mbbeta}$. What could go wrong? After all, $\{0,1\} \subset \reals$... 

There are a few issues:
1. The linear model produces probabilities $\pi(\mbx) \in \reals$ instead of just over the valid range $[0,1]$, so the model is necessarily misspecified.
2. Moreover, the variance of a Bernoulli random variable changes as a function of the probability,
    \begin{align*}
    \Var[Y \mid \mbX=\mbx] = \pi(\mbx) (1 - \pi(\mbx)),
    \end{align*}
    which violates the homoskedasticity assumption under which OLS is optimal. 

Nevertheless, it's not a totally crazy thing to do. When the estimated probabilities are in an intermediate range (say, 0.3-0.7), the outputs aren't that different from what we obtain with the alternative models below. But we can do better.

## Logistic Regression

The idea is simple: keep the linear part of linear regression, but apply a **mean (aka inverse link) function**, $f: \reals \mapsto [0,1]$, to ensure $\pi(\mbx)$ returns valid probabilities,
\begin{align*}
\pi(\mbx) &= f(\mbbeta^\top \mbx).
\end{align*}
There are infinitely many squashing nonlinearities that we could choose for $f$, but a particularly attractive choice is the **logistic (aka sigmoid) function**,
\begin{align*}
f(a) = \frac{e^a}{1 + e^a} = \frac{1}{1 + e^{-a}} \triangleq \sigma(a).
\end{align*}
One nice feature of the logistic function is that it is monotonically increasing, so increasing in $\mbbeta^\top \mbx$ yield larger probability estimates. That also means that we can invert the sigmoid function. In doing so, we find that the linear component of the model $\mbbeta^\top \mbx$ correspond to the **log odds** of the binary response since the inverse of the sigmoid function is the **logit function**,
\begin{align*}
\mbbeta^\top \mbx = \sigma^{-1}(\pi(\mbx)) &= \log \frac{\pi(\mbx)}{1 - \pi(\mbx)}.
\end{align*}
Finally, we'll see that the logistic function leads to some simpler mathematical calculations when it comes to parameter estimation and inference.

Another common mean function is the Gaussian CDF, and we'll consider that in a later chapter. 

## Relationship with Two-Way Contingency Tables

Suppose we have a single, binary covariate $X \in \{0,1\}$. In the last chapter, we constructed 2x2 contingency tables for such settings, and we used the log odds ratio to measure the association between $X$ and $Y$. We could do the same thing with logistic regression.

First, we need to extend the model with an intercept term,
\begin{align*}
\pi(x) &= \sigma \left(\beta_0 + \beta_1 x \right),
\end{align*}
for $x \in \{0,1\}$.

:::{admonition} Note about intercepts
:class: warning
We explicitly separated the intercept term above, but in general we can assume that the covariates include a constant term, $\mbx = (1, x_1, \ldots, x_p)^\top \in \reals^{p+1}$. Then the first coefficient in $\mbbeta = (\beta_0, \beta_1, \ldots, \beta_p)^\top \in \reals^{p+1}$ corresponds to the intercept.
:::

Under this model, $\beta$ specifies the log odds,
\begin{align*}
\beta = \sigma^{-1}(\pi(1)) - \sigma^{-1}(\pi(0)) 
= \log \frac{\pi(1) / (1 - \pi(1))}{\pi(0) / (1 - \pi(0))} 
= \log \theta.
\end{align*}
In other words, the coefficients of the logistic regression model correspond to the log odds in a contingency table.

## Maximum Likelihood Estimation

Unfortunately, unlike in standard linear regression (equivalently, when $f(a) = a$ is the identity function), there isn't a simple closed form estimator for $\hat{\mbbeta}$. However, we can use standard optimization techniques to do maximum likelihood estimation.

First, write the log likelihood of the parameters given a collection of covariates and responses, $\{\mbx_i, y_i\}_{i=1}^n$,
\begin{align*}
\cL(\mbbeta) 
&= \sum_{i=1}^n \log \mathrm{Bern}(y_i; \pi(\mbx_i)) \\
&= \sum_{i=1}^n y_i \log \pi(\mbx_i) + (1 - y_i) \log (1 - \pi(\mbx_i)) \\
&= \sum_{i=1}^n y_i \log \frac{\pi(\mbx_i)}{1 - \pi(\mbx_i)} + \log (1 - \pi(\mbx_i)).
\end{align*}
Now let's plug in the definition of $\pi(\mbx)$. The first term is just the log odds, which we already showed is equal to the linear component of the model. The second term simplifies too.
\begin{align*}
\cL(\mbbeta) 
&= \sum_{i=1}^n y_i \mbbeta^\top \mbx_i + \log \left(1 - \frac{e^{\mbbeta^\top \mbx_i}}{1 + e^{\mbbeta^\top \mbx_i}} \right) \\
&= \sum_{i=1}^n y_i \mbbeta^\top \mbx_i - \log \left(1 + e^{\mbbeta^\top \mbx_i} \right).
\end{align*}

### Computing the Gradient
We want to maximize the log likelihood, so let's take the gradient,
\begin{align*}
\nabla \cL(\mbbeta) 
&= \sum_{i=1}^n y_i \mbx_i - \frac{e^{\mbbeta^\top \mbx_i}}{1 + e^{\mbbeta^\top \mbx_i}} \mbx_i \\
&= \sum_{i=1}^n \left(y_i - \sigma(\mbbeta^\top \mbx_i) \right) \mbx_i \\
&= \sum_{i=1}^n \left(y_i - \pi(\mbx_i) \right) \mbx_i \\
&= \sum_{i=1}^n \left(y_i - \E(Y \mid \mbX = \mbx_i) \right) \mbx_i.
\end{align*}
The gradient is a weighted sum of the covariates, and the weights are the residuals $y_i - \pi(\mbx_i)$, i.e., the difference between the observed and expected response. 

This is pretty intuitive! Remember that the gradient points in the direction of steepest ascent. This tells us that to increase the log likelihood the most, we should move the coefficient in the direction of covariates where the residual is positive (we are underestimating the mean), and we should move opposite the direction of covariates where the residual is negative (where we are overestimating the mean). 

Now that we have a closed-form expression for the gradient, we can implement a simple gradient ascent algorithm to maximize the log likelihood,
\begin{align*}
\mbbeta^{(i+1)} &\leftarrow \mbbeta^{(i)} + \alpha_i \nabla \cL(\mbbeta^{(i)}),
\end{align*}
where $\alpha_i \in \reals_+$ is the step-size at iteration $i$ of the algorithm. If the step sizes are chosen appropriately, the alorithm is guaranteed to converge to at least a local optimum of the log likelihood. 

### Computing the Hessian

Can we say more about the optima found by gradient ascent? When the log likelihood is strictly concave, there is a unique global optimum, and gradient ascent (with appropriately chosen step sizes) must find it. To check the concavity of the log likelihood, we need to compute its Hessian,
\begin{align*}
\nabla^2 \cL(\mbbeta) 
&= -\sum_{i=1}^n \sigma'(\mbbeta^\top \mbx_i) \mbx_i \mbx_i^\top
\end{align*}
where $\sigma'(a)$ is the derivative of the logistic function. That is,
\begin{align*}
\sigma'(a) 
&= \frac{\dif}{\dif a}  \sigma(a)\\
&= \frac{e^a}{(1+e^a)^2} \\
&= \sigma(a) (1 - \sigma(a)).
\end{align*}
Plugging this in,
\begin{align*}
\nabla^2 \cL(\mbbeta) 
&= - \sum_{i=1}^n \sigma(\mbbeta^\top \mbx_i)(1 - \sigma(\mbbeta^\top \mbx_i)) \mbx_i \mbx_i^\top \\
&= - \sum_{i=1}^n \pi(\mbx_i)(1 - \pi(\mbx_i)) \mbx_i \mbx_i^\top \\
&= - \sum_{i=1}^n \Var[Y \mid \mbX = \mbx_i] \mbx_i \mbx_i^\top.
\end{align*}
In other words, the negative Hessian is a weighted sum of outer products of covariates where the weights are equal to the conditional variance.

### Putting it All Together
Since variances are non-negative, so are the weights. Altogether, this derivation shows that the Hessian is **negative definite**, which means:
- the log likelihood is strictly concave,
- the maximum likelihood estimate $\hat{\mbbeta}_{\mathsf{MLE}}$ is unique, and
- gradient ascent with appropriately chosen step sizes will find the MLE.

:::{admonition} Technicalities
:class: warning
- Technically, $\pi(\mbx)$ cannot be exactly zero unless $\mbbeta^\top \mbx = - \infty$, so the weights in this sum will be strictly positive if $\mbx_i$ and $\mbbeta$ are finite.
- In order for the Hessian to be negative definite, the covariates must also span $\reals^p$. 
- In order for gradient ascent to converge to the global maximum of a strictly concave function, the step sizes must either decay according to Robbins-Munro conditions, be smaller than the minimum eigenvalue of the negative Hessian, or be chosen according to a backtracking line search.
:::

