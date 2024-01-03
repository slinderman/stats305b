# Contingency Tables 

Last time we introduced basic distributions for discrete random variables &mdash; our first _models_! But a model of a single discrete random variable isn't all that interesting... What would be really cool is if we could model the joint distribution of _two_ categorical random variables. That's exactly what contingency tables do. 

## Definitions

Let $X \in \{1,\ldots, I\}$ and $Y \in \{1,\ldots, J\}$ be categorical random variables with $I$ and $J$ categories, respectively. We represent the **joint probability distribution** as an $I \times J$ table,
\begin{align*}
\mbPi = \begin{bmatrix}
\pi_{11} & \ldots & \pi_{1J} \\
\vdots & & \vdots \\
\pi_{K1} & \ldots & \pi_{KJ}
\end{bmatrix}
\end{align*}
where
\begin{align*}
\pi_{ij} = \Pr(X = i, Y = j).
\end{align*}
The probabilities must be normalized,
\begin{align*}
1 = \sum_{i=1}^I \sum_{j=1}^J \pi_{ij} \triangleq \pi_{\bullet \bullet}
\end{align*}
The **marginal probabilities** are given by,
\begin{align*}
\Pr(X = i) &= \sum_{j=1}^J \pi_{ij} \triangleq \pi_{i \bullet}, \\
\Pr(Y = j) &= \sum_{i=1}^I \pi_{ij} \triangleq \pi_{\bullet j}.
\end{align*}
Finally, the conditional probabilities are given by Bayes' rule,
\begin{align*}
\Pr(Y =j \mid X=i) &= \frac{\Pr(X=i, Y=j)}{\Pr(X=i)} = \frac{\pi_{ij}}{\pi_{i \bullet}} \triangleq \pi_{j | i}
\end{align*}

## Independence

One of the key questions in the analysis of contingency tables is whether $X$ and $Y$ are independent. In particular, they are independent if the joint distribution factors into a product of marginals,
\begin{align*}
X \perp Y \iff \pi_{ij} = \pi_{i \bullet} \pi_{\bullet j} \; \forall i,j.
\end{align*}

Equivalently, the variables are independent if the conditionals are _homogeneous_,
\begin{align*}
X \perp Y \iff \pi_{j|i} = \frac{\pi_{ij}}{\pi_{i \bullet}} = \frac{\pi_{i \bullet} \pi_{\bullet j}}{\pi_{i \bullet}} = \pi_{\bullet j} \; \forall i,j.
\end{align*}

## Sampling

Typically, we don't observe the probabilities $\mbPi$ directly, and we want to draw inferences about them given noisy observations. Let $\mbX \in \naturals^{I \times J}$ denote a matrix of counts $X_{ij}$ for each cell of the table. We need a model of how $\mbX$ is sampled. 

### Poisson Sampling

Under a Poisson sampling model,
\begin{align*}
X_{ij} &\sim \mathrm{Po}(\lambda_{ij})
\end{align*}
where $\lambda_{ij} / \lambda_{\bullet \bullet} = \pi_{ij}$. Here, the number of total counts is a random variable,
\begin{align*}
X_{\bullet \bullet} &\sim \mathrm{Po}(\lambda_{\bullet \bullet}).
\end{align*}
The sampling models below correspond to special cases of Poisson sampling when we condition on certain marginal counts.

### Multinomial Sampling

If we condition on the total number of counts, we obtain a multinomial sampling model,
\begin{align*}
\mathrm{vec}(\mbX) \mid X_{\bullet \bullet}= x_{\bullet \bullet} &\sim \mathrm{Mult}(x_{\bullet \bullet}, \mathrm{vec}(\mbPi)),
\end{align*}
where $\mathrm{vec}(\cdot)$ is a function that _vectorizes_ or _ravels_ a matrix into a vector. The corresponding pmf is,
\begin{align*}
\Pr(\mbX = \mbx \mid X_{\bullet \bullet} = x_{\bullet \bullet}) &= 
{x_{\bullet \bullet} \choose x_{11}; \cdots; x_{IJ}} \prod_{i=1}^I \prod_{j=1}^J \pi_{ij}^{x_{ij}}
\end{align*}

### Independent Multinomial Sampling

When the row variables are explanatory variables, we often model each row of counts as conditionally independent given the row-sums,
\begin{align*}
\mbX_{i} \mid X_{i \bullet} = x_{i \bullet} &\sim \mathrm{Mult}(x_{i \bullet}, \mbpi_{\cdot \mid i})
\end{align*}
with pmf
\begin{align*}
\Pr(\mbX=\mbx \mid X_{1 \bullet} = x_{1 \bullet}, \ldots X_{I \bullet} = x_{I \bullet})
&= \prod_{i=1}^I \mathrm{Mult}(\mbx_i \mid x_{i \bullet}, \mbpi_{\cdot \mid i}) \\
&= \prod_{i=1}^I \left[ {x_{i \bullet} \choose x_{i1}; \cdots; x_{iJ}} \prod_{j=1}^J \pi_{j \mid i}^{x_{ij}} \right]
\end{align*}

### Hypergeometric sampling
Sometimes we condition on _both_ the row and column sums. For 2x2 tables, under the null hypothesis that the rows are independent (i.e., assuming homogenous conditionals), the resulting sampling distribution is the hypergeometric,
\begin{align*}
X_{11} \mid X_{\bullet \bullet} = x_{\bullet \bullet}, X_{1 \bullet} = x_{1 \bullet}, X_{\bullet 1} = x_{\bullet 1}
&\sim \mathrm{HyperGeom}(x_{\bullet \bullet}, x_{1 \bullet}, x_{\bullet 1})
\end{align*}
with pmf
\begin{align*}
\mathrm{HyperGeom}(x_{11}; x_{\bullet \bullet}, x_{1 \bullet}, x_{\bullet 1})
&= 
\frac{{x_{1 \bullet} \choose x_{11}} {x_{\bullet \bullet} - x_{1 \bullet} \choose x_{\bullet 1} - x_{11}}}{{x_{\bullet \bullet} \choose x_{\bullet 1}}}
\end{align*}

We can arrive at this conditional distribution using Bayes' rule. The following is adapted from {cite:t}`blitzstein2019introduction` (Ch 3.9). We will abbreviate some of the probability notation so that it's not so cumbersome. Also, we'll index our rows and columns starting with 0, to be consistent with our notation below. Under the independent Poisson sampling model,
\begin{align*}
\Pr(x_{11} \mid x_{\bullet \bullet}, x_{1 \bullet}, x_{\bullet 1}) 
&=
\frac{\Pr(x_{11} \mid x_{\bullet \bullet}, x_{1 \bullet}) \Pr(x_{\bullet 1} \mid x_{11}, x_{\bullet \bullet}, x_{1 \bullet})}{\Pr(x_{\bullet 1} \mid x_{\bullet \bullet}, x_{1 \bullet})} \\
&=
\frac{\mathrm{Bin}(x_{11}; x_{1 \bullet}, \pi_{11}) \mathrm{Bin}(x_{01}; x_{0 \bullet}, \pi_{01})}{\Pr(x_{\bullet 1} \mid x_{\bullet \bullet}, x_{1 \bullet})},
\end{align*}
noting that $x_{01} = x_{\bullet 1} - x_{11}$ and $x_{0 \bullet} = x_{\bullet \bullet} - x_{1 \bullet}$.


Under the null hypothesis of independence, $\pi_{11} = \pi_{01} = p$, and we have 
\begin{align*}
X_{i1} &\ind\sim \mathrm{Bin}(x_{i \bullet}, p) & \text{for } i&\in0,1\\
\implies X_{\bullet 1} = X_{01} + X_{11} &\sim \mathrm{Bin}(x_{\bullet \bullet}, p ),
\end{align*} 
and $\Pr(x_{\bullet 1} \mid x_{\bullet \bullet}, x_{1 \bullet}) = \mathrm{Bin}(x_{\bullet 1}; x_{\bullet \bullet}, p)$.

Substituting in the binomial pmf yields,
\begin{align*}
\Pr(x_{11} \mid x_{\bullet \bullet}, x_{1 \bullet}, x_{\bullet 1}) 
&= 
\frac
{
    \left({x_{1 \bullet} \choose x_{11}} p^{x_{11}} (1-p)^{x_{1 \bullet} - x_{11}} \right)
    \left({x_{\bullet \bullet} - x_{1 \bullet} \choose x_{\bullet 1} - x_{11}} p^{x_{\bullet 1} - x_{11}} (1-p)^{x_{\bullet \bullet} - x_{1 \bullet} - x_{\bullet 1} + x_{11}} \right)
}
{
    {x_{\bullet \bullet} \choose x_{1 \bullet}} p^{x_{\bullet 1}} (1 - p)^{x_{\bullet \bullet} - x_{\bullet 1}}
} \\
&= 
\frac
{
    {x_{1 \bullet} \choose x_{11}}
    {x_{\bullet \bullet} - x_{1 \bullet} \choose x_{\bullet 1} - x_{11}} 
}
{
    {x_{\bullet \bullet} \choose x_{1 \bullet}}
} \\
&= \mathrm{HyperGeom}(x_{11}; x_{\bullet \bullet}, x_{1 \bullet}, x_{\bullet 1}).
\end{align*}
Interestingly, the probability $p$ cancels out in the hypergeometric pmf so that it only depends on the marginal counts. 

## Comparing Two Proportions

Contingency tables are often used to compare two groups $X \in \{0,1\}$ based on a binary response variables $Y \in \{0,1\}$. The resulting tables are 2x2. The association between $X$ and $Y$ can be summarized with a variety of statistics: the difference of proportions, the relative risk, and the odds ratio. We will focus on the latter.

For a Bernoulli random variable with probability $p$, the odds are defined as 
\begin{align*}
\Omega = \frac{p}{1 - p}. 
\end{align*}
Inversely, $p = \frac{\Omega}{\Omega + 1}$.

For a 2x2 table, each row defines a Bernoulli conditional,
\begin{align*}
Y \mid X=i &\sim \mathrm{Bern}(\pi_{1|i}) & \text{for } i &\in \{0,1\},
\end{align*}
where 
\begin{align*}
\pi_{1|i} =
 \frac{\pi_{i1}}{\pi_{i0} + \pi_{i1}}
 = \frac{\pi_{i1}}{\pi_{i \bullet}} 
\triangleq \pi_i.
\end{align*}
where we have introduced the shorthand notation $\pi_i$. The odds for row $i$ are,
\begin{align*}
\Omega_i = \frac{\pi_i}{1 - \pi_i} 
= \frac{\pi_{i1}}{1 - \pi_{i1}} 
= \frac{\pi_{i1}}{\pi_{i0}}.
\end{align*}

The _odds ratio_ $\theta$ is exactly what it sounds like, 
\begin{align*}
\theta 
&= \frac{\Omega_1}{\Omega_0} 
= \frac{\pi_{11} \pi_{00}}{\pi_{10} \pi_{01}}
\end{align*}

The odds ratio is non-negative, $\theta \in \reals_+$. When $X$ and $Y$ are independent, $\pi_{11} = \pi_{01}$ and $\pi_{00} = \pi_{10}$ so that $\theta = 1$. 

For inference it is often more convenient to work with the _log odds ratio_,
\begin{align*}
\log \theta &= \log \pi_{11} + \log \pi_{00} - \log \pi_{10} - \log \pi_{01}.
\end{align*}
Under independence, the log odds ratio is 0. The magnitude of the log odds ratio represents the strength of association. 

## Conditional Independence
We often need to control for confounding variables $Z$ when studying the relationship between $X$ and $Y$. Suppose $X$, $Y$, and $Z$ are all binary random variables. We can represent their joint distribution with a 2x2x2 contingency table,
\begin{align*}
\Pr(X=i, Y=j, Z=k) = \pi_{ijk}
\end{align*} 
(Maybe we should call this a contingency tensor?)

In this setting, controlling for $Z$ amounts to considering conditional probabilities, 
\begin{align*}
\Pr(X=i, Y=j \mid Z=k) = \frac{\Pr(X=i, Y=j, Z=k)}{Pr(Z=k)} 
= \frac{\pi_{ijk}}{\pi_{\bullet \bullet k}} 
\triangleq \pi_{ij|k},
\end{align*}
for each level $k$, instead of the marginal probabilities, $\pi_{ij \bullet}$.

We can define conditional odds ratios, etc., accordingly. For example, 
\begin{align*}
\log \theta_{XY|k} = \log \frac{\pi_{11|k} \pi_{00|k}}{\pi_{10|k} \pi_{01|k}}.
\end{align*}

We say that $X$ and $Y$ are _conditionally independent given $Z$_ (more concisely, $X \perp Y \mid Z$) if
\begin{align*}
\Pr(X=i, Y=j \mid Z=k) &= \Pr(X=i \mid Z=k) \Pr(Y=j \mid Z=k) \; \forall i,j,k.
\end{align*}

### Simpsons paradox

Conditional independence does not imply marginal independence. Indeed, measures of marginal association and conditional association can even differ in sign. This is called _Simpson's paradox_.

## Confidence Intervals for Log Odds Ratio 

Given a sample of counts $\mbX=\mbx$ from a contingency table, the MLE estimate of the probabilities is
\begin{align*}
\hat{\pi_{ij}} &= \frac{x_{ij}}{x_{\bullet \bullet}}
\end{align*}
For a 2x2 table, the sample estimate of log odds ratio is,
\begin{align*}
\log \hat{\theta} &= \log \frac{\hat{\pi}_{11} \hat{\pi}_{00}}{\hat{\pi}_{10} \hat{\pi}_{01}} 
= \log \frac{x_{11} x_{00}}{x_{10} x_{01}}.
\end{align*}

We can estimate 95% Wald confidence intervals usign the asymptotic normality of the estimator,
\begin{align*}
\log \hat{\theta} \pm 1.96 \, \hat{\sigma}(\log \hat{\theta})
\end{align*}
where
\begin{align*}
\hat{\sigma}(\log \hat{\theta})
&= \left(\frac{1}{x_{11}} + \frac{1}{x_{00}} + \frac{1}{x_{10}} + \frac{1}{x_{01}} \right)^{\frac{1}{2}}
\end{align*}
is an estimate of the standard error using the _delta method_.

### Delta method
The sample log odds ratio is a nonlinear function of our maximum likelihood estimates of $\hat{\pi}_{ij}$,
\begin{align*}
\hat{\pi}_{ij} &= \frac{x_{ij}}{n}.
\end{align*}
where $n = x_{\bullet \bullet} = \sum_{ij} x_{ij}$.

Let $\hat{\mbpi} = \mathrm{vec}(\hat{\mbPi}) = (\hat{\pi}_{11}, \hat{\pi}_{10}, \hat{\pi}_{01}, \hat{\pi}_{00})$ denote the vector of probability estimates. 

The MLE is asymptotically normal with variance given by the inverse Fisher information,
\begin{align*}
\sqrt{n}(\hat{\mbpi} - \mbpi) \to \mathrm{N}(0, \cI(\hat{\mbpi})^{-1}) 
\end{align*}
where
\begin{align*}
\cI(\hat{\mbpi})^{-1} 
&= 
\begin{bmatrix}
\pi_{11}(1-\pi_{11}) & -\pi_{11} \pi_{10} & - \pi_{11} \pi_{01} & -\pi_{11} \pi_{00} \\
-\pi_{10} \pi_{11} & \pi_{10} (1 - \pi_{10}) & - \pi_{10} \pi_{01} & -\pi_{10} \pi_{00} \\
-\pi_{01} \pi_{11} & -\pi_{01} \pi_{10} & \pi_{01} (1 - \pi_{01}) & -\pi_{01} \pi_{00} \\
-\pi_{00} \pi_{11} & -\pi_{00} \pi_{10} & -\pi_{00} \pi_{01} & \pi_{00} (1 - \pi_{00})
\end{bmatrix}
\end{align*}
The (multivariate) delta method is a way of estimating the variance of a scalar function of the estimator, $g(\hat{\mbpi})$. Using a first order Taylor approximation around the true probabilities,
\begin{align*}
g(\hat{\mbpi}) &\approx g(\mbpi) + \nabla g(\mbpi)^\top (\hat{\mbpi} - \mbpi)
\end{align*}
we can derive the approximate variance as,
\begin{align*}
\Var[g(\hat{\mbpi})] 
&\approx 
\Var[\nabla g(\mbpi)^\top \hat{\mbpi}] 
= \nabla g(\mbpi)^\top \Cov[\hat{\mbpi}] \nabla g(\mbpi).
\end{align*}
Then, the estimate of $g(\mbpi)$ is asymptotically normal as well, and its variance depends on the gradient of $g$,
\begin{align*}
\sqrt{n}(g(\hat{\mbpi}) - g(\mbpi)) \to \mathrm{N}(0, \nabla g(\mbpi)^\top \cI(\hat{\mbpi})^{-1} \nabla g(\mbpi)) 
\end{align*}


## Independence Testing in Two-Way Tables
- Likelihood ratio test

## Fisher's Exact Test for Two-Way Tables

## Bayesian Inference for Two-Way Tables