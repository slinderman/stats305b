# Logistic Regression

[One, two, many...](https://www.science.org/doi/10.1126/science.1094492) We started with basic discrete distributions for single random variables, and then we modeled pairs of categorical variables with continency tables. Here, we build models for predicting categorical responses given several explanatory variables.

## Model

Let $Y \in \{0,1\}$ denote a binary response and let $\mbX \in \reals^p$ denote associated covariates. For example, $Y$ could denote whether or not your favorite football team wins their match, and $X$ could represent features of the match like whether its a home or away game, who their opponent is, etc. We will model the conditional probability of success as a function of the covariates,
\begin{align*}
\Pr(Y = 1 \mid \mbX=\mbx) &= \E[Y \mid \mbX=\mbx] \triangleq \pi(\mbx).
\end{align*}

This is a standard regression setup. The modeling problem boils down to choosibng the functional form of $\pi(\mbx)$. 

### Linear Probability Model
If you took STATS 305A, you know pretty much everything there is to know about linear models for continuous response variables, $Y \in \reals$. Why don't we just apply that same model to binary responses? Specifically, let,
\begin{align*}
\pi(\mbx) &= \mbbeta^\top \mbx = \sum_{j=1}^p \beta_j x_j. 
\end{align*}

<!-- :::{admonition} Note about intercepts
:class: warning
Typically we assume that $x_ -->

Then we'll just use ordinary least squares (OLS) to estimate $\hat{\mbbeta}$. What could go wrong? After all, $\{0,1\} \subset \reals$... 

There are a few issues:
1. The linear model produces probabilities $\pi(\mbx) \in \reals$ instead of just over the valid range $[0,1]$, so the model is necessarily misspecified.
2. Moreover, the variance of a Bernoulli random variable changes as a function of the probability,
  \begin{align*}
  \Var[Y \mid \mbX=\mbx] = \pi(\mbx) (1 - \pi(\mbx)),
  \end{align*}
  which violates the homoskedasticity assumption under which OLS is optimal. 