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

First, write the **negative** log likelihood of the parameters given a collection of covariates and responses, $\{\mbx_i, y_i\}_{i=1}^n$,
\begin{align*}
\cL(\mbbeta) 
&= - \sum_{i=1}^n \log \mathrm{Bern}(y_i; \pi(\mbx_i)) \\
&= - \sum_{i=1}^n y_i \log \pi(\mbx_i) + (1 - y_i) \log (1 - \pi(\mbx_i)) \\
&= - \sum_{i=1}^n y_i \log \frac{\pi(\mbx_i)}{1 - \pi(\mbx_i)} + \log (1 - \pi(\mbx_i)).
\end{align*}
Now let's plug in the definition of $\pi(\mbx)$. The first term is just the log odds, which we already showed is equal to the linear component of the model. The second term simplifies too.
\begin{align*}
\cL(\mbbeta) 
&= - \sum_{i=1}^n y_i \mbbeta^\top \mbx_i + \log \left(1 - \frac{e^{\mbbeta^\top \mbx_i}}{1 + e^{\mbbeta^\top \mbx_i}} \right) \\
&= - \sum_{i=1}^n y_i \mbbeta^\top \mbx_i - \log \left(1 + e^{\mbbeta^\top \mbx_i} \right).
\end{align*}

### Computing the Gradient
We want to maximize the log likelihood, or equivalently minimize the negative log likelihood, so let's take the gradient,
\begin{align*}
\nabla \cL(\mbbeta) 
&= - \sum_{i=1}^n y_i \mbx_i - \frac{e^{\mbbeta^\top \mbx_i}}{1 + e^{\mbbeta^\top \mbx_i}} \mbx_i \\
&= - \sum_{i=1}^n \left(y_i - \sigma(\mbbeta^\top \mbx_i) \right) \mbx_i \\
&= - \sum_{i=1}^n \left(y_i - \pi(\mbx_i) \right) \mbx_i \\
&= - \sum_{i=1}^n \left(y_i - \E(Y \mid \mbX = \mbx_i) \right) \mbx_i.
\end{align*}
The gradient is a weighted sum of the covariates, and the weights are the residuals $y_i - \pi(\mbx_i)$, i.e., the difference between the observed and expected response. 

This is pretty intuitive! Remember that the gradient points in the direction of steepest descent. This tells us that to increase the log likelihood the most (equivalently, decrease $\cL$ the most), we should move the coefficient in the direction of covariates where the residual is positive (we are underestimating the mean), and we should move opposite the direction of covariates where the residual is negative (where we are overestimating the mean). 

Now that we have a closed-form expression for the gradient, we can implement a simple gradient descent algorithm to minimize the negative log likelihood,
\begin{align*}
\mbbeta^{(i+1)} &\leftarrow \mbbeta^{(i)} - \alpha_i \nabla \cL(\mbbeta^{(i)}),
\end{align*}
where $\alpha_i \in \reals_+$ is the step-size at iteration $i$ of the algorithm. If the step sizes are chosen appropriately and the objective is well behaved, the alorithm converges to at least a local optimum of the log likelihood. 

### Convexity of the Log Likelhood

If the objective is convex, then all local optima are also global optima, and we can give stronger guarantees on gradient descent. To check the convexity of the log likelihood, we need to compute its Hessian,
\begin{align*}
\nabla^2 \cL(\mbbeta) 
&= \sum_{i=1}^n \sigma'(\mbbeta^\top \mbx_i) \mbx_i \mbx_i^\top
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
&= \sum_{i=1}^n \sigma(\mbbeta^\top \mbx_i)(1 - \sigma(\mbbeta^\top \mbx_i)) \mbx_i \mbx_i^\top \\
&= \sum_{i=1}^n \pi(\mbx_i)(1 - \pi(\mbx_i)) \mbx_i \mbx_i^\top \\
&= \sum_{i=1}^n \Var[Y \mid \mbX = \mbx_i] \mbx_i \mbx_i^\top.
\end{align*}
In other words, the Hessian is a weighted sum of outer products of covariates where the weights are equal to the conditional variance. 
Since variances are non-negative, so are the weights, which implies that the Hessian is **positive semi-definite**, which implies that the negative log likelihood is convex.


## Converge Rate of Gradient Descent
To determine when and at what rate gradient descent converges, we need to know more about the eigenvalues of the Hessian. 

If we can bound the maximum eigenvalue of the Hessian by $L$, then we can obtain a quadratic upper bound on the negative log likelihood,
\begin{align*}
\cL(\mbbeta') &\leq \cL(\mbbeta) + \nabla \cL(\mbbeta)^\top (\mbbeta' - \mbbeta) + \frac{L}{2} (\mbbeta' - \mbbeta)^\top \nabla^2 \cL(\mbbeta) (\mbbeta' - \mbbeta).
\end{align*}
That means the negative log likelihood is an $L$-smooth function.

For example, if the covariates have bounded norm, $\|\mbx_i\|_2 \leq B$, then we can bound the maximum eigenvalue of the Hessian by,
\begin{align*}
\lambda_{\mathsf{max}} 
&= \max_{\mbu \in \bbS_{p-1}} \mbu^\top \nabla^2 \cL(\mbbeta) \mbu \\
&= \max_{\mbu \in \bbS_{p-1}} \sum_{i=1}^n \Var[Y \mid \mbX = \mbx_i] \mbu^\top \mbx_i \mbx_i^\top \mbu \\
&\leq \frac{n B^2}{4}
\end{align*}
since the variance of a Bernoulli random variable is at most $\tfrac{1}{4}$ and since $\mbu^\top \mbx_i \leq B$ for all unit vectors $\mbu \in \bbS_{p-1}$ (the unit sphere embedded in $\reals^p$). This isn't meant to be a tight upper bound.

If we run gradient descent with a constant step size of $\alpha = 1/L$, then the algorithm converges at a rate of $1/t$, which means that after $t$ iterations
\begin{align*}
\cL(\mbbeta^\star) - \cL(\mbbeta^{(t)}) \leq \frac{L}{t} \|\mbbeta^\star - \mbbeta^{(0)}\|_2^2,
\end{align*}
where $\mbbeta^{(0)}$ is the initial setting of the parameters and $\mbbeta^\star$ is the global optimum. 

Put differently, if we want a gap of at most epsilon, we need to run $t = O(1/\epsilon)$ iterations of gradient descent. This is called a **sub-linear convergence** rate.

## Pathologies in the Linearly Separable Regime

Not

<!-- 
Assuming that the covariates span $\reals^p$, this derivation shows that the Hessian is **negative definite**, which means:
- the log likelihood is strictly concave,
- the maximum likelihood estimate $\hat{\mbbeta}_{\mathsf{MLE}}$ is unique, and
- gradient descent with appropriately chosen step sizes will find the MLE. -->


<!-- :::{admonition} Technicalities
:class: warning
- Technically, $\pi(\mbx)$ cannot be exactly zero unless $\mbbeta^\top \mbx = - \infty$, so the weights in this sum will be strictly positive if $\mbx_i$ and $\mbbeta$ are finite.
- In order for the Hessian to be negative definite, the covariates must also span $\reals^p$. 
- In order for gradient descent to converge to the global maximum of a strictly concave function, the step sizes must either decay according to Robbins-Munro conditions, be smaller than $1/L$ where $L$ is the maximum eigenvalue of the negative Hessian, or be chosen according to a backtracking line search.
::: -->

