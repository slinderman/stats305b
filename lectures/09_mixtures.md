# Mixture Models and EM

## Bayesian Mixture Models

Let,
- $N$ denote the number of data points
- $K$ denote the number of mixture components (i.e., clusters)
- $\mbx_n \in \reals^D$ denote the $n$-th data point
- $z_n \in \{1, \ldots, K\}$ be a latent variable denoting the cluster assignment of the $n$-th data point
- $\mbtheta_k$ be natural parameters of cluster $k$.
- $\mbpi \in \Delta_{K-1}$ be cluster proportions (probabilities).
<!-- - $\mbphi, \nu$ be hyperparameters of the prior on $\mbtheta$. -->
- $\mbalpha \in \reals_+^{K}$ be the concentration of the prior on $\mbpi$.

The generative model is as follows


> 1. Sample the proportions from a Dirichlet prior:
>     \begin{align*}
          \mbpi &\sim \mathrm{Dir}(\mbalpha)
      \end{align*}
> 
> 2. Sample the parameters for each component:
>     \begin{align*}
          \mbtheta_k &\iid{\sim} p(\mbtheta) \quad \text{for } k = 1, \ldots, K
      \end{align*}
> 
> 3. Sample the assignment of each data point:
>     \begin{align*}
          z_n &\iid{\sim} \mbpi \quad \text{for } n = 1, \ldots, N
      \end{align*}
> 
> 4. Sample data points given their assignments:
>     \begin{align*}
          \mbx_n &\sim p(\mbx \mid \mbtheta_{z_n}) \quad \text{for } n = 1, \ldots, N
      \end{align*}

### Joint distribution
The joint distribution is,
\begin{align*}
    p(\mbpi, \{\mbtheta_k\}_{k=1}^K, \{(z_n, \mbx_n)\}_{n=1}^N \mid \mbalpha) 
    &\propto 
    p(\mbpi \mid \mbalpha) \prod_{k=1}^K p(\mbtheta_k) \prod_{n=1}^N \prod_{k=1}^K \left[ \pi_k \, p(\mbx_n \mid \mbtheta_k) \right]^{\bbI[z_n = k]}
\end{align*}

### Exponential family mixture models

Assume an exponential family likelihood of the form,
\begin{align*}
    p(\mbx \mid \mbtheta_k) &= h(\mbx_n) \exp \left \{\langle t(\mbx_n), \mbtheta_k \rangle - A(\mbtheta_k) \right \}
\end{align*}
And a conjugate prior of the form,
\begin{align*}
    p(\mbtheta_k \mid \mbphi, \nu) &\propto \exp \left \{ \langle \mbphi, \mbtheta_k \rangle - \nu A(\mbtheta_k) \right \}
\end{align*}

:::{admonition} Example: Gaussian Mixture Model (GMM)

Assume the conditional distribution of $\mbx_n$ is a Gaussian with mean $\mbtheta_k \in \reals^D$ and identity covariance:
\begin{align*}
    p(\mbx_n \mid \mbtheta_k) &= \mathrm{N}(\mbx_n \mid \mbtheta_{k}, \mbI)
\end{align*}

The conjugate prior is a Gaussian prior on the mean:
\begin{align*}
    p(\mbtheta_k) &= \mathrm{N}(\mbmu_0, \sigma_0^{2} \mbI) \nonumber \\
    &\propto \exp \left\{-\tfrac{1}{2 \sigma_0^2} (\mbtheta_k - \mbmu_0)^\top (\mbtheta_k - \mbmu_0) \right\}
\end{align*}

In exponential family form, the prior precision is $\nu = 1/\sigma_0^2$ and the prior precision-weighted mean is $\mbphi = \mbmu_0 / \sigma_0^2$. 

:::

## Two Inference Algorithms

Let's stick with the Gaussian mixture model for now. Suppose we observe data points $\{\mbx_n\}_{n=1}^N$ and want to infer the assignments $\{z_n\}_{n=1}^N$ and means $\{\mbtheta_k\}_{k=1}^K$. Here are two intuitive algorithms.

### MAP Inference and K-Means

We could obtain point estimates $\hat{\mbtheta}_{k}$ and $\hat{z}_n$ that maximize the posterior probability. Recall that is called _maximum a posteriori_ (MAP) estimation. Here, we could find the posterior mode via coordinate ascent.

:::{prf:algorithm} MAP Estimation for a GMM
**Repeat** until convergence: 
1. For each $n=1,\ldots, N$, fix the means $\mbtheta$ and set,
    \begin{align*}
        z_n &= \arg \min_{k \in \{1,\ldots, K\}} \|\mbx_n - \mbtheta_k\|_2
    \end{align*}

2. For each $k=1,\ldots,K$, fix all assignments $\mbz$ and set,
    \begin{align*}
        \mbtheta_k &= \frac{1}{N_k} \sum_{n=1}^K \bbI[z_n=k] \mbx_n
    \end{align*}
:::

It turns out this algorithm goes by a more common name &mdash; you might recognize it as the **k-means algorithm**!

### Maximum Likelihood Estimation via EM

K-Means made **hard assignments** of data points to clusters in each iteration. That sounds a little extreme &mdash; do you really want to attribute a datapoint to a single class when it is right in the middle of two clusters? What if we used **soft assignments** instead?

:::{prf:algorithm} EM for a GMM
**Repeat** until convergence: 
1. For each data point $n$ and component $k$, compute the **responsibility**:
    \begin{align*}
        \omega_{nk} = \frac{\pi_k \mathrm{N}(\mbx_n \mid \mbtheta_k, \mbI)}{\sum_{j=1}^K \pi_j \mathrm{N}(\mbx_n \mid \mbtheta_j, \mbI)}
    \end{align*}

2. For each component $k$, update the mean:
    \begin{align*}
        \mbtheta_k^\star &= \frac{1}{N_k} \sum_{n=1}^K \omega_{nk} \mbx_n
    \end{align*}
:::

This is the **Expectation-Maximization (EM) algorithm**. As we will show, EM yields an estimate that maximizes the _marginal_ likelihood of the data.


## Theoretical Motivation
Rather than maximizing the **joint probability**, EM is maximizing the **marginal probability**,
\begin{align*}
    \log p(\{\mbx_n\}_{n=1}^N, \mbtheta) 
    &= \log p(\mbtheta) + \log \sum_{z_1=1}^K \cdots \sum_{z_N=1}^K p(\{\mbx_n, z_n\}_{n=1}^N \mid \mbtheta) \\
    &= \log p(\mbtheta) + \log \prod_{n=1}^N \sum_{z_n=1}^K p(\mbx_n, z_n \mid \mbtheta) \\
    &= \log p(\mbtheta) + \sum_{n=1}^N  \log \sum_{z_n=1}^K p(\mbx_n, z_n \mid \mbtheta) 
\end{align*}
For discrete mixtures (with small enough $K$) we can evaluate the log marginal probability (with what complexity?).  We can usually evaluate its gradient too, so we could just do gradient ascent to find $\mbtheta^*$. However, EM typically obtains faster convergence rates.

### Evidence Lower Bound (ELBO)
The key idea is to obtain a lower bound on the marginal probability,
\begin{align*}
    \log p(\{\mbx_n\}_{n=1}^N, \mbtheta) 
    &= \log p(\mbtheta) + \sum_{n=1}^N  \log \sum_{z_n} p(\mbx_n, z_n \mid \mbtheta) \\
    &= \log p(\mbtheta) + \sum_{n=1}^N  \log \sum_{z_n} q(z_n) \frac{p(\mbx_n, z_n \mid \mbtheta)}{q(z_n)} \\
    &= \log p(\mbtheta) + \sum_{n=1}^N  \log \E_{q(z_n)} \left[\frac{p(\mbx_n, z_n \mid \mbtheta)}{q(z_n)} \right]
\end{align*}
where $q(z_n)$ is any distribution on $z_n \in \{1,\ldots,K\}$ such that $q(z_n)$ is **absolutely continuous** w.r.t. $p(\mbx_n, z_n \mid \mbtheta)$. 

:::{admonition} Jensen's Inequality
:class: tip
Jensen's inequality states that,
\begin{align*}
    f(\E[Y]) \geq \E\left[ f(Y) \right]
\end{align*}
if $f$ is a **concave function**, with equality iff $f$ is linear.
:::
    
Applied to the log marginal probability, Jensen's inequality yields,
\begin{align*}
    \log p(\{\mbx_n\}_{n=1}^N, \mbtheta) 
    &= \log p(\mbtheta) + \sum_{n=1}^N  \log \E_{q_n} \left[\frac{p(\mbx_n, z_n \mid \mbtheta)}{q_n(z_n)} \right] \\
    &\geq \log p(\mbtheta) + \sum_{n=1}^N  \E_{q_n} \left[\log p(\mbx_n, z_n \mid \mbtheta) - \log q_n(z_n) \right] \\
    &\triangleq \cL[\mbtheta, \mbq]
\end{align*}
where $\mbq = (q_1, \ldots, q_N)$ is a tuple of densities.

This is called the **evidence lower bound**, or **ELBO** for short.  It is a function of $\mbtheta$ and a _functional_ of $\mbq$, since each $q_n$ is a probability density function. We can think of _EM as coordinate ascent on the ELBO_.


### M-step: Gaussian case

Suppose we fix $\mbq$. Since each $z_n$ is a discrete latent variable, $q_n$ must be a probability mass function. Let it be denoted by,
\begin{align*}
    q_n &= [\omega_{n1}, \ldots, \omega_{nK}]^\top.
\end{align*}
(These will be the **responsibilities** from before.)

Now, recall our basic model, $\mbx_n \sim \mathrm{N}(\mbtheta_{z_n}, \mbI)$, and assume a prior $\mbtheta_k \sim \mathrm{N}(\mbmu_0, \sigma_0^2 \mbI)$, Then,
\begin{align*}
    \cL[\mbtheta, \mbq] 
    &= \log p(\mbtheta) + 
    \sum_{n=1}^N \E_{q_n} [\log p(\mbx_n, z_n \mid \mbtheta)] + c \\
    &= \log p(\mbtheta) + 
    \sum_{n=1}^N \sum_{k=1}^K \omega_{nk} \log p(\mbx_n, z_n=k \mid \mbtheta) + c \\
    &= \sum_{k=1}^K \left[\frac{1}{\sigma_0^2} \mbmu_0^\top \mbtheta_k - \tfrac{1}{2 \sigma_0^2} \mbtheta_k^\top \mbtheta_k \right] +
    \sum_{n=1}^N \sum_{k=1}^K \omega_{nk} \left[ \mbx_n^\top \mbtheta_k - \tfrac{1}{2} \mbtheta_k^\top \mbtheta_k \right] + c
\end{align*}
    
Zooming in on just $\mbtheta_k$,
\begin{align*}
    \cL[\mbtheta, \mbq] 
    &= \widetilde{\mbphi}_{k}^\top \mbtheta_k - \tfrac{1}{2} \widetilde{\nu}_{k} \mbtheta_k^\top \mbtheta_k
\end{align*}
where
\begin{align*}
    \widetilde{\mbphi}_{k} &= \mbmu_0 / \sigma_0^2 + \sum_{n=1}^N \omega_{nk} \mbx_n
    \qquad
    \widetilde{\nu}_{k} = 1/\sigma_0^2 + \sum_{n=1}^N \omega_{nk}
\end{align*}
Taking derivatives and setting to zero yields, 
\begin{align*}
    \mbtheta_k^\star &=  \frac{\widetilde{\mbphi}_{k}}{\widetilde{\nu}_{k}} 
    = \frac{\mbmu_0/\sigma_0^2 + \sum_{n=1}^N \omega_{nk} \mbx_n}{1/\sigma_0^2 + \sum_{n=1}^N \omega_{nk}}.
\end{align*}
In the improper uniform prior limit where $\mbmu_0 = 0$ and $\sigma_0^2 \to \infty$, we recover the EM updates shown above.

### E-step: Gaussian case
As a function of $q_n$, for discrete Gaussian mixtures with identity covariance,
\begin{align*}
    \cL[\mbtheta, \mbq] 
    &= \E_{q_n}\left[ \log p(\mbx_n, z_n \mid \mbtheta) - \log q_n(z_n)\right] + c \\
    &= \sum_{k=1}^K \omega_{nk} \left[ \log \mathrm{N}(\mbx_n \mid \mbtheta_k, \mbI) + \log \pi_k - \log \omega_{nk} \right] + c
\end{align*}
where $\mbpi = [\pi_1, \ldots, \pi_K]^\top$ is the vector of prior cluster probabilities.

We also have two constraints: $\omega_{nk} \geq 0$ and $\sum_k \omega_{nk} = 1$. Let's ignore the non-negative constraint for now (it will automatically be satisfied anyway) and write the Lagrangian with the simplex constraint,
\begin{align*}
    \cJ(\mbomega_n, \lambda) &= \sum_{k=1}^K \omega_{nk} \left[ \log \mathrm{N}(\mbx_n \mid \theta_k, \mbI) + \log \pi_k - \log \omega_{nk} \right] - \lambda \left( 1 - \sum_{k=1}^K \omega_{nk} \right)
\end{align*}

Taking the partial derivative wrt $\omega_{nk}$ and setting to zero yields,
\begin{align*}
    \frac{\partial}{\partial \omega_{nk}} \cJ(\mbomega_n, \lambda) 
    &= \log \mathrm{N}(\mbx_n \mid \mbtheta_k, \mbI) + \log \pi_k - \log \omega_{nk} - 1 + \lambda = 0 \\
    \Rightarrow \log \omega_{nk}^\star &= \log \mathrm{N}(\mbx_n \mid \mbtheta_k, \mbI) + \log \pi_k + \lambda - 1 \\
    \Rightarrow \omega_{nk}^\star &\propto \pi_k \mathrm{N}(\mbx_n \mid \mbtheta_k, \mbI) 
\end{align*}
Enforcing the simplex constraint yields,
\begin{align*}
    \omega_{nk}^\star &= \frac{\pi_k \mathrm{N}(\mbx_n \mid \mbtheta_k, \mbI)}{\sum_{j=1}^K \pi_j \mathrm{N}(\mbx_n \mid \mbtheta_j, \mbI)},
\end{align*}
just like above.

Note that 
\begin{align*}
    \omega_{nk}^\star &\propto p(z_n=k) \, p(\mbx_n \mid z_n=k, \mbtheta) = p(z_n = k \mid \mbx_n, \mbtheta) .
\end{align*}
That is, the responsibilities equal the posterior probabilities!

### The ELBO is tight after the E-step
Equivalently, $q_n$ equals the posterior, $p(z_n \mid \mbx_n, \mbtheta)$.
At that point, the ELBO simplifies to,
\begin{align*}
    \cL[\mbtheta, \mbq] 
    &= \log p(\mbtheta) + \sum_{n=1}^N  \E_{q_n} \left[\log p(\mbx_n, z_n \mid \mbtheta) - \log q_n(z_n) \right] \\
    &= \log p(\mbtheta) + \sum_{n=1}^N  \E_{p(z_n \mid \mbx_n, \mbtheta)} \left[\log p(\mbx_n, z_n \mid \mbtheta) - \log p(z_n \mid \mbx_n, \mbtheta) \right] \\
    &= \log p(\mbtheta) + \sum_{n=1}^N  \E_{p(z_n \mid \mbx_n, \mbtheta)} \left[\log p(\mbx_n \mid \mbtheta) \right] \\
    &= \log p(\mbtheta) + \sum_{n=1}^N \log p(\mbx_n \mid \mbtheta) \\
    &= \log p(\{\mbx_n\}_{n=1}^N, \mbtheta)
\end{align*}

:::{admonition} EM as a minorize-maximize (MM) algorithm
:class: tip
Note that the <b>The ELBO is tight after the E-step!</b>.

We can view the EM algorihtm as a **minorize-maximize (MM)** algorithm where we iteratively lower bound the ELBO and and then maximize the lower bound.
:::

<!--
### EM as a minorize-maximize (MM) algorithm
 \begin{figure}
    \centering
    \includegraphics[width=3.5in]{figures/lecture7/em.png}
    \caption{Bishop, Figure 9.14: EM alternates between constructing a lower bound (minorizing) and finding new parameters that maximize it.}
    \label{fig:em_mm}
\end{figure} -->


### M-step: General Case
Now let's consider the general Bayesian mixture with exponential family likelihoods and conjugate priors. As a function of $\mbtheta$,
\begin{align*}
    \cL[\mbtheta, \mbq] 
    &= \log p(\mbtheta) + 
    \sum_{n=1}^N \E_{q_n} [\log p(\mbx_n, z_n \mid \mbtheta)] + \mathrm{c} \\
    &= \log p(\mbtheta) + 
    \sum_{n=1}^N \sum_{k=1}^K \omega_{nk} \log p(\mbx_n, z_n=k \mid \mbtheta) + \mathrm{c} \\
    &= \sum_{k=1}^K \left[\mbphi^\top \mbtheta_k - \nu A(\mbtheta_k)\right] +
    \sum_{n=1}^N \sum_{k=1}^K \omega_{nk} \left[ t(\mbx_n)^\top \mbtheta_k - A(\mbtheta_k) \right] + \mathrm{c} 
\end{align*}

Zooming in on just $\mbtheta_k$,
\begin{align*}
    \cL[\mbtheta, \mbq] 
    &= \widetilde{\mbphi}_{k}^\top \mbtheta_k - \widetilde{\nu}_{k} A(\mbtheta_k)
\end{align*}
where
\begin{align*}
    \widetilde{\mbphi}_{k} &= \mbphi + \sum_{n=1}^N \omega_{nk} t(\mbx_n)
    \qquad
    \widetilde{\nu}_{k} = \nu + \sum_{n=1}^N \omega_{nk}
\end{align*}
Taking derivatives and setting to zero yields, 
\begin{align*}
    \label{eq:gen_mstep}
    \mbtheta_k^* &= \left[\nabla A \right]^{-1} \left(\frac{\widetilde{\mbphi}_{k}}{\widetilde{\nu}_{k}}\right)
\end{align*}

Recall that $\nabla A^{-1}: \cM \mapsto \Omega$ is a mapping from mean parameters to natural parameters (and the inverse exists for minimal exponential families). Thus, the generic M-step above amounts to finding the natural parameters $\mbtheta_k^*$ that yield the expected sufficient statistics $\widetilde{\mbphi}_{k} / \widetilde{\nu}_{k}$ by inverting the gradient mapping.

### E-step: General Case
In our first pass, we assumed $q_n$ was a finite pmf. More generally, $q_n$ will be a probability density function, and optimizing over functions usually requires the _calculus of variations_. (Ugh!)

However, note that we can write the ELBO in a slightly different form,
\begin{align*}
    \cL[\mbtheta, \mbq] 
    &= \log p(\mbtheta) + \sum_{n=1}^N  \E_{q_n} \left[\log p(\mbx_n, z_n \mid \mbtheta) - \log q_n(z_n) \right] \\
    &= \log p(\mbtheta) + \sum_{n=1}^N  \E_{q_n} \left[\log p(z_n \mid \mbx_n, \mbtheta) + \log p(\mbx_n \mid \mbtheta) - \log q_n(z_n) \right] \\
    &= \log p(\mbtheta) + \sum_{n=1}^N  \left[\log p(\mbx_n \mid \mbtheta) -\KL{q_n(z_n)}{p(z_n \mid \mbx_n, \mbtheta)} \right] \\
    &= \log p(\{\mbx_n\}_{n=1}^N, \mbtheta) - \sum_{n=1}^N  \KL{q_n(z_n)}{p(z_n \mid \mbx_n, \mbtheta)}
\end{align*}
where $\KL{\cdot}{\cdot}$ denote the **Kullback-Leibler divergence**.

Recall, the KL divergence is defined as,
\begin{align*}
    \KL{q(z)}{p(z)} &= \int q(z) \log \frac{q(z)}{p(z)} \dif z.
\end{align*}
    
It gives a notion of how similar two distributions are, but it is _not a metric!_ (It is not symmetric.) Still, it has some intuitive properties:
1. It is non-negative, $\KL{q(z)}{p(z)} \geq 0$.
2. It equals zero iff the distributions are the same, $\KL{q(z)}{p(z)} = 0 \iff q(z) = p(z)$ almost everywhere.

Maximizing the ELBO wrt $q_n$ amounts to minimizing the KL divergence to the posterior $p(z_n \mid \mbx_n, \mbtheta)$,
\begin{align*}
    \cL[\mbtheta, \mbq] 
    &= \log p(\mbtheta) + \sum_{n=1}^N  \left[\log p(\mbx_n \mid \mbtheta) -\KL{q_n(z_n)}{p(z_n \mid \mbx_n, \mbtheta)} \right] \\
    &= -\KL{q_n(z_n)}{p(z_n \mid \mbx_n, \mbtheta)} + c
\end{align*}
As we said, the KL is minimized when $q_n(z_n) = p(z_n \mid \mbx_n, \mbtheta)$, so the optimal update is, 
\begin{align*}
    q_n^\star(z_n) &= p(z_n \mid \mbx_n, \mbtheta),
\end{align*}
just like we found above.

## Conclusion

Mixture models are basic building blocks of statistics, and our first encounter with **discrete latent variable models (LVMs)**.  (Where have we seen continuous LVMs so far?) Mixture models have widespread uses in both density estimation (e.g., kernel density estimators) and data science (e.g., clustering). Next, we'll talk about how to extend mixture models to cases where the cluster assignments are correlated in time.