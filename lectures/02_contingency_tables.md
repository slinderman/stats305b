# Contingency Tables 

Last time we introduced basic distributions for discrete random variables &mdash; our first _models_! But a model of a single discrete random variable isn't all that interesting... Contingency tables allow us to model and reason about the joint distribution of _two_ categorical random variables. Two might not sound like a lot &mdash; we'll get to more complex models soon enough! &mdash; but it turns out plenty of important questions boil down to understanding the relationship between two variables.

## Motivating Example

We used the College Football National Championship to motivate our analyses in the last lecture, but I have to admit, I have a love-hate relationship with football. While it's fun to watch, it's increasingly clear that repetitive head injuries sustained in football can have devastating consequences, including an increased risk of chronic traumatic encephalopathy (CTE). A recent study from {cite:t}`mckee2023neuropathologic` in _JAMA Neurology_ showed that CTE can be found even in amateur high school and college athletes, and the New York Times highlighted their research in a very sad [article](https://www.nytimes.com/interactive/2023/11/16/us/cte-youth-football.html) last fall.

The only way to definitely diagnose CTE is via autopsy. {cite:t}`mckee2023neuropathologic` studied the brains of 152 people who had played contact sports and died under the age of 30 from various causes including injury, overdose, suicide, and others (but not from neurodegenerative disease). Of those 152 people, 92 had played football and the rest had played other sports like soccer, hockey, wrestling, rugby, etc. Of the 152 people, 63 were found to have CTE upon neuropathologic evaluation. Of the 92 football players, 48 had CTE.

We can summarize that result in a 2 $\times$ 2 table:

|                 | No CTE  | CTE | **Total** |
| --------------- | ------- | --- | --------- |
| **No Football** | 45      |  15 | 60        |
| **Football**    | 44      |  48 | 92        |
| **Total**       | 89      |  63 | 152       |

:::{admonition} Questions
With this data, can we say that playing football is associated with CTE? If so, how strong is the association? Can we say whether this association is causal? What are some caveats to consider when interpreting this data?
:::

## Contingency Tables

The table above is an example of a **contingency table**. It represents a sample from a **joint distribution** of two random variables, $X \in \{0,1\}$ indicating whether the person played football, and $Y \in \{0,1\}$ indicating whether they had CTE. 

More generally, let $X \in \{1,\ldots, I\}$ and $Y \in \{1,\ldots, J\}$ be categorical random variables. We represent the joint distribution as an $I \times J$ table,
\begin{align*}
\mbPi = \begin{bmatrix}
\pi_{11} & \ldots & \pi_{1J} \\
\vdots & & \vdots \\
\pi_{I1} & \ldots & \pi_{IJ}
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

We don't usually observe the probabilities $\mbPi$ directly. Instead, we have to draw inferences about them given noisy observations. Let $\mbX \in \naturals^{I \times J}$ denote a matrix of counts $X_{ij}$ for each cell of the table. We need a model of how $\mbX$ is sampled. 

### Poisson Sampling

Under a Poisson sampling model,
\begin{align*}
X_{ij} &\sim \mathrm{Po}(\lambda_{ij})
\end{align*}
where $\lambda_{ij} / \lambda_{\bullet \bullet} = \pi_{ij}$. The scale, $\lambda_{\bullet \bullet}$, is a free parameter. Here, the total count is a random variable,
\begin{align*}
X_{\bullet \bullet} &\sim \mathrm{Po}(\lambda_{\bullet \bullet}).
\end{align*}
The sampling models below correspond to special cases of Poisson sampling when we condition on certain marginal counts.

### Multinomial Sampling

If we condition on the total count, we obtain a multinomial sampling model,
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

:::{admonition} Deriving the hypergeometric distribution by Bayes' rule
:class: dropdown

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
:::

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
where, recall, 
\begin{align*}
\pi_{1|i} = \frac{\pi_{i1}}{\pi_{i0} + \pi_{i1}}.
\end{align*}
The odds for row $i$ are,
\begin{align*}
\Omega_i = \frac{\pi_i}{1 - \pi_i} 
= \frac{\pi_{i1}}{\pi_{i0}}.
\end{align*}

The _odds ratio_ $\theta$ is exactly what it sounds like, 
\begin{align*}
\theta 
&= \frac{\Omega_1}{\Omega_0} 
= \frac{\pi_{11} \pi_{00}}{\pi_{10} \pi_{01}}
\end{align*}

The odds ratio is non-negative, $\theta \in \reals_+$. When $X$ and $Y$ are independent, the homogeneity of conditionals implies that $\pi_{1|1} = \pi_{1|0}$ and $\pi_{0|1} = \pi_{0|0}$. In turn, $\Omega_1 = \Omega_0$ so that the odds ratio, $\theta$, is one.

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

We say that $X$ and $Y$ are _conditionally independent given $Z$_ (more concisely, $X \perp Y \mid Z$) if
\begin{align*}
\Pr(X=i, Y=j \mid Z=k) &= \Pr(X=i \mid Z=k) \Pr(Y=j \mid Z=k) \; \forall i,j,k.
\end{align*}

For 2x2xK tables, we define the conditional log odds ratios as,
\begin{align*}
\log \theta_{k} = \log \frac{\pi_{11|k} \pi_{00|k}}{\pi_{10|k} \pi_{01|k}}.
\end{align*}
Conditional independence corresponds to $\log \theta_k = 0 \; \forall k$.

### Simpsons paradox

Conditional independence does not imply marginal independence. Indeed, measures of marginal association and conditional association can even differ in sign. This is called _Simpson's paradox_.

## Confidence Intervals for Log Odds Ratio 

Given a sample of counts $\mbX=\mbx$ from a contingency table, the MLE estimate of the probabilities is
\begin{align*}
\hat{\pi}_{ij} &= \frac{x_{ij}}{x_{\bullet \bullet}}
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
The sample log odds ratio is a nonlinear function of the maximum likelihood estimates of $\hat{\pi}_{ij}$,
\begin{align*}
\hat{\pi}_{ij} &= \frac{x_{ij}}{n}.
\end{align*}
where $n = x_{\bullet \bullet} = \sum_{ij} x_{ij}$.

Let $\hat{\mbpi} = \mathrm{vec}(\hat{\mbPi}) = (\hat{\pi}_{11}, \hat{\pi}_{10}, \hat{\pi}_{01}, \hat{\pi}_{00})$ denote the vector of probability estimates. 

The MLE is asymptotically normal with variance given by the inverse Fisher information,
\begin{align*}
\sqrt{n}(\hat{\mbpi} - \mbpi) \to \mathrm{N}(0, \cI(\mbpi)^{-1}) 
\end{align*}
where
\begin{align*}
\cI(\mbpi)^{-1} 
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
\sqrt{n}(g(\hat{\mbpi}) - g(\mbpi)) \to \mathrm{N}(0, \nabla g(\mbpi)^\top \cI(\mbpi)^{-1} \nabla g(\mbpi)) 
\end{align*}

For the log odds ratio, $g(\mbpi) = \log \pi_{11} + \log \pi_{00} - \log \pi_{10} - \log \pi_{01}$ and 
\begin{align*}
\nabla g(\mbpi) &= \begin{pmatrix}
1 / \pi_{11} \\ -1 / \pi_{10} \\ -1 / \pi_{01} \\ 1 / \pi_{00} 
\end{pmatrix}.
\end{align*} 
Substituting this into the expression for the asymptotic variance yields,
\begin{align*}
\nabla g(\mbpi)^\top \cI(\mbpi)^{-1} \nabla g(\mbpi)
&= \sum_{ij} [\cI(\mbpi)^{-1}]_{ij} \cdot [\nabla g(\mbpi)]_i \cdot [\nabla g(\mbpi)]_j \\
&= \frac{1}{\pi_{11}} + \frac{1}{\pi_{00}} + \frac{1}{\pi_{10}} + \frac{1}{\pi_{01}}.
\end{align*}
Of course, we don't know $\mbpi$. Plugging in the estimates $\hat{\pi}_{ij} = x_{ij} / n$ yields the Wald standard error,
\begin{align*}
\hat{\sigma}(\log \hat{\theta}) 
&= \left(\frac{\nabla g(\hat{\mbpi})^\top \cI(\hat{\mbpi})^{-1} \nabla g(\hat{\mbpi})}{n} \right)^{\frac{1}{2}} \\
&= \left(\frac{1}{x_{11}} + \frac{1}{x_{00}} + \frac{1}{x_{10}} + \frac{1}{x_{01}} \right)^{\frac{1}{2}},
\end{align*}
as shown above.

## Independence Testing in Two-Way Tables
Last time, we derived Wald confidence intervals from the acceptance region of a Wald hypothesis test. We could do the reverse here to to test independence in $2 \times 2$ tables using the Wald confidence. Instead, we will derive an independence test that works more generally for $I \times J$ tables. Instead of a Wald test, we'll use a likelihood ratio test.

Let $\cH_0: \pi_{ij} = \pi_{i \bullet} \pi_{\bullet j}$ for all $i,j$ be our null hypothesis of independence. The null hypothesis imposes a constraint on the set of probabilities $\mbPi$. Rather than taking on any value $\mbPi \in \Delta_{IJ - 1}$, they are constrained to the $\Delta_{I-1} \times \Delta_{J-1}$ subset of probabilities that factor into an outer product of marginal probabilities. 

The likelihood ratio test compares the maximum likelihood under the constrained set to the maximum likelihood under the larger space of all probabilities,
\begin{align*}
\lambda &= 
-2 \log \frac
{
    \sup_{\mbpi_{i \bullet}, \mbpi_{\bullet j} \in \Delta_{I-1} \times \Delta_{J-1}} p(\mbx; \mbpi_{i \bullet} \mbpi_{\bullet j}^\top)
}
{
    \sup_{\mbPi \in \Delta_{IJ-1}} p(\mbx; \mbPi)
}
\end{align*}
The maximum likelihoods estimates of the constrained model are $\hat{\pi}_{i \bullet} = x_{i \bullet} / x_{\bullet \bullet}$ and $\hat{\pi}_{\bullet j} = x_{\bullet j} / x_{\bullet \bullet}$; under the unconstrained model they are $\hat{\pi}_{ij} = x_{ij} / x_{\bullet \bullet}$. Plugging these estimates in yields,
\begin{align*}
\lambda &= 
-2 \log \frac
{
    \prod_{ij} \left( \frac{x_{i \bullet} x_{\bullet j}}{x_{\bullet \bullet}^2} \right)^{x_{ij}}
}
{
    \prod_{ij} \left( \frac{x_{i j}}{x_{\bullet \bullet}} \right)^{x_{ij}}
} \\
&= -2 \sum_{ij} x_{ij} \log \frac{\hat{\mu}_{ij}}{x_{ij}}
\end{align*}
where $\hat{\mu}_{ij} = x_{\bullet \bullet} \hat{\pi}_{i \bullet} \hat{\pi}_{\bullet j} = x_{i \bullet} x_{\bullet j} / x_{\bullet \bullet}$ is the expected value of $X_{ij}$ under the null hypothesis of independence.

Under the null hypothesis, $\lambda$ is asymptotically distributed as chi-squared with $(IJ -1) - (I-1) - (J-1) = (I-1)(J-1)$ degrees of freedom,
\begin{align*}
\lambda \sim \chi^2_{(I-1)(J-1)},
\end{align*}
allowing us to construct p-values.

## Fisher's Exact Test for Two-Way Tables

The p-value for the likelihood ratio test is based on an asymptotic chi-squared distribution, which only holds as $n = x_{\bullet \bullet} \to \infty$. For two-way tables with small $n$, we can do exact inference using the hypergeometric sampling distribution for $x_{11}$ given the row- and column-marginals under the null hypothesis of independence. This is called _Fisher's exact test_.

Consider testing the null hypothesis $\cH_0: \log \theta = 0$ against the one-sided alternative,  $\cH_0 \log \theta > 0$ (i.e., a positive association between the two variables). The sample log odds ratio, $\log \hat{\theta} = \frac{x_{11} x_{00}}{x_{10} x_{01}}$, is monotonically increasing in $x_{11}$. That is, increasing $x_{11}$ leads to increasing $\log \hat{\theta}$. Fisher's exact test corresponds to the probability of seeing a value at least as large as $x_{11}$ under the null hypothesis,
\begin{align*}
\Pr(X_{11} \geq x_{11} \mid x_{1 \bullet}, x_{\bullet 1}, x_{\bullet \bullet}, \cH_0)
&= \sum_{k=x_{11}}^{\min\{x_{\bullet 1}, x_{1 \bullet} \}} \mathrm{HyperGeom}(k; x_{\bullet \bullet}, x_{1 \bullet}, x_{\bullet 1}).
\end{align*}

## Bayesian Inference for Two-Way Tables

Finally, let's conclude with some approaches for Bayesian inference. Again, this involves placing a prior on the parameters of interest. For example, in a two-way table, we could use independent, conjugate beta priors,
\begin{align*}
\pi_{1|i} &\iid\sim \mathrm{Beta}(\alpha, \beta) & \text{for } i &\in \{0, 1\}.
\end{align*}
When $\alpha = \beta = 1$, this reduces to $\pi_{1|i} \iid\sim \mathrm{Unif}([0,1])$. 

Under an independent beta prior, the posterior distribution on parameters factors into a product of betas as well,
\begin{align*}
p(\mbPi \mid \mbX=\mbx) 
&= \mathrm{Beta}(\pi_{1|1} \mid \alpha + x_{11}, \beta + x_{10}) \, 
\mathrm{Beta}(\pi_{1|0} \mid \alpha + x_{01}, \beta + x_{00})
\end{align*}

Under this prior, the rows are _almost surely dependent_, since $p(\pi_{1|1} = \pi_{1|0}) = 0$. Nevertheless, we can use this model to draw posterior inferences about association measures like the log odds ratio, $\log \theta$. For example, we can use Monte Carlo to estimate tail probabilities,
\begin{align*}
\Pr(\log \theta \geq t \mid \mbX = \mbx) 
&= \int \bbI\left[\log \frac{\pi_{1|1} / (1 - \pi_{1|1})}{\pi_{1|0} / (1 - \pi_{1|0})} \geq t \right] \; p(\mbPi \mid \mbX = \mbx) \dif \mbPi \\
&\approx \sum_{m=1}^M \bbI\left[\log \frac{\pi_{1|1}^{(m)} / (1 - \pi_{1|1}^{(m)})}{\pi_{1|0}^{(m)} / (1 - \pi_{1|0}^{(m)})} \geq t \right]
\end{align*}
where 
\begin{align*}
\pi_{1|1}^{(m)} &\iid\sim \mathrm{Beta}(\alpha + x_{11}, \beta + x_{10}), \\
\pi_{1|0}^{(m)} &\iid\sim \mathrm{Beta}(\alpha + x_{01}, \beta + x_{00}), \\
\end{align*}
Likewise, we can use the same approach to compute posterior credible intervals for $\log \theta$. 

:::{admonition} Question
:class: tip

How could you construct a more appropriate prior distribution for capturing correlations (or exact equality) between the conditional probabilities?

:::


## Conclusion

Contingency tables are fundamental tools for studying the relationship between two categorical random variables. We discussed models sampling contingency tables, conditioning on various marginals, as well as various measures of association between the random variables. Then we presented methods for inferring associations and testing hypotheses of independence. However, these methods were ultimately limited to just two (often binary) variables. Next, we'll consider models for capturing relationships between a response and several covariates.