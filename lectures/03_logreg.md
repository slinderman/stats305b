# Logistic Regression

[One, two, many...](https://www.science.org/doi/10.1126/science.1094492) We started with basic discrete distributions for single random variables, and then we modeled pairs of categorical variables with continency tables. Here, we build models for predicting categorical responses given several explanatory variables.

## Setup

Let $Y \in \{0,1\}$ denote a binary response and let $\mbX \in \reals^p$ denote associated covariates. For example, $Y$ could denote whether or not your favorite football team wins their match, and $X$ could represent features of the match like whether its a home or away game, who their opponent is, etc. We will model the conditional probability of success as a function of the covariates,
\begin{align*}
\Pr(Y = 1 \mid \mbX=\mbx) &= \E[Y \mid \mbX=\mbx] \triangleq \pi(\mbx).
\end{align*}

This is a standard regression setup. The modeling problem boils down to choosibng the functional form of $\pi(\mbx)$. 

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