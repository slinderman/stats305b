# Hidden Markov Models

**Hidden Markov Models (HMMs)** are latent variable models for sequential data. Like the mixture models from the previous chapter, HMMs have discrete latent states. Unlike mixture models, the discrete latent states of an HMM are not independent: the state at time $t$ depends on the state at time $t-1$. These dependencies allow the HMM to capture how states transition from one to another over time. Thanks to the **Markovian** dependency structure, posterior inference and parameter estimation in HMMs remain tractable.

## Gaussian Mixture Models

Recall the basic Gaussian mixture model,

$$
\begin{aligned}
    z_t &\stackrel{\text{iid}}{\sim} \mathrm{Cat}(\mbpi) \\
    x_t  \mid z_t &\sim \mathcal{N}(\mbmu_{z_t}, \mbSigma_{z_t})
\end{aligned}
$$

where

-   $z_t \in \{1,\ldots,K\}$ is a **latent mixture assignment**
-   $x_t \in \reals^D$ is an **observed data point**
-   $\mbpi\in \Delta_{K-1}$,
    $\mbmu_k \in \reals^D$, and
    $\mbSigma_k \in \reals_{\succeq 0}^{D \times D}$ are
    parameters

(Here we've switched to indexing data points by $t$ rather than $n$.)

Let $\mbTheta$ denote the set of parameters. We can be
Bayesian and put a prior on $\mbTheta$ and run Gibbs or VI,
or we can point estimate $\mbTheta$ with EM, etc.


::: {admonition} Exercise
:class: tip
Draw the graphical model for a GMM.
:::

## The EM algorithm

Recall the EM algorithm for mixture models,

-   **E step:** Compute the posterior distribution

    $$
    \begin{aligned}
                q(\mbz_{1:T})
                &= p(\mbz_{1:T}  \mid \mbx_{1:T}; \mbTheta) \\
                &= \prod_{t=1}^T p(z_t  \mid \mbx_t; \mbTheta) \\
                &= \prod_{t=1}^T q_t(z_t)
    \end{aligned}
    $$

-   **M step:** Maximize the ELBO wrt $\mbTheta$,

    $$
    \begin{aligned}
                \mathcal{L}(\mbTheta)
                &= \mathbb{E}_{q(\mbz_{1:T})}\left[\log p(\mbx_{1:T}, \mbz_{1:T}; \mbTheta) - \log q(\mbz_{1:T}) \right] \\
                &= \mathbb{E}_{q(\mbz_{1:T})}\left[\log p(\mbx_{1:T}, \mbz_{1:T}; \mbTheta) \right]  + c.
            \end{aligned}
    $$

For exponential family mixture models, the M-step only requires expected
sufficient statistics.


## Hidden Markov Models

Hidden Markov Models (HMMs) are like mixture models
with temporal dependencies between the mixture assignments.

<!-- ::: {.center}
![image](figures/lap7/hmm.png){width="70%"}
::: -->

This graphical model says that the joint distribution factors as,

$$
\begin{aligned}
    p(z_{1:T}, \mbx_{1:T}) &= p(z_1) \prod_{t=2}^T p(z_t  \mid z_{t-1}) \prod_{t=1}^T p(\mbx_t  \mid z_t).\end{aligned}
$$

We call this an HMM because the *hidden* states follow a Markov chain,
$p(z_1) \prod_{t=2}^T p(z_t  \mid z_{t-1})$.


An HMM consists of three components:

1.  **Initial distribution:**
    $z_1 \sim \mathrm{Cat}(\mbpi_0)$

2.  **Transition matrix:**
    $z_t \sim \mathrm{Cat}(\mbP_{z_{t-1}})$ where
    $\mbP\in [0,1]^{K \times K}$ is a *row-stochastic*
    transition matrix with rows $\mbP_k$.

3.  **Emission distribution:**
    $\mbx_t \sim p(\cdot  \mid \boldsymbol{\theta}_{z_t})$


## Inference and Learning in HMM

We are interested in questions like:

-   What are the *predictive distributions* of
    $p(z_{t+1}  \mid \mbx_{1:t})$?

-   What is the *posterior marginal* distribution
    $p(z_t  \mid \mbx_{1:T})$?

-   What is the *posterior pairwise marginal* distribution
    $p(z_t, z_{t+1}  \mid \mbx_{1:T})$?

-   What is the *posterior mode*
    $z_{1:T}^\star = \mathop{\mathrm{arg\,max}}p(z_{1:T}  \mid \mbx_{1:T})$?

-   How can we *sample the posterior*
    $p(\mbz_{1:T}  \mid \mbx_{1:T})$ of an HMM?

-   What is the *marginal likelihood* $p(\mbx_{1:T})$?

-   How can we *learn the parameters* of an HMM?

:::{admonition} Exercise
:class: tip
On the surface, what makes these inference problems harder than in the simple mixture model case?
:::

## Computing the predictive distributions

The predictive distributions give the probability of the latent state $z_{t+1}$ given observations *up to but not including* time $t+1$. Let,

$$
\begin{aligned}
        \alpha_{t+1}(z_{t+1}) &\triangleq p(z_{t+1}, \mbx_{1:t}) \\
        &= \sum_{z_1=1}^K \cdots \sum_{z_{t}=1}^K p(z_1) \prod_{s=1}^{t} p(\mbx_s  \mid z_s) \, p(z_{s+1}  \mid z_{s})\\
        &= \sum_{z_{t}=1}^K \left[ \left( \sum_{z_1=1}^K \cdots \sum_{z_{t-1}=1}^K p(z_1) \prod_{s=1}^{t-1} p(\mbx_s  \mid z_s) \, p(z_{s+1}  \mid z_{s}) \right) p(\mbx_t  \mid z_t) \, p(z_{t+1}  \mid z_{t}) \right]  \\
        &= \sum_{z_{t}=1}^K \alpha_t(z_t) \, p(\mbx_t  \mid z_t) \, p(z_{t+1}  \mid z_{t}).
\end{aligned}
$$

We call $\alpha_t(z_t)$ the *forward messages*. We
can compute them recursively! The base case is
$p(z_1  \mid \varnothing) \triangleq p(z_1)$.


We can also write these
recursions in a vectorized form. Let

$$
\begin{aligned}
        \mbalpha_t &=
        \begin{bmatrix}
        \alpha_t(z_t = 1) \\
        \vdots \\
        \alpha_t(z_t=K)
        \end{bmatrix}
        =
        \begin{bmatrix}
        p(z_t = 1, \mbx_{1:t-1}) \\
        \vdots \\
        p(z_t = K, \mbx_{1:t-1})
        \end{bmatrix}
        \qquad \text{and} \qquad
        \mbl_t =
        \begin{bmatrix}
        p(\mbx_t  \mid z_t = 1) \\
        \vdots \\
        p(\mbx_t  \mid z_t = K)
        \end{bmatrix}
    \end{aligned}
$$

both be vectors in $\reals_+^K$. Then,

$$
\begin{aligned}
        \mbalpha_{t+1} &= \mbP^\top (\mbalpha_t \odot \mbl_t)
    \end{aligned}
$$

where $\odot$ denotes the Hadamard (elementwise)
product and $\mbP$ is the transition matrix.


Finally, to get the
predictive distributions we just have to normalize,

$$
\begin{aligned}
        p(z_{t+1}  \mid \mbx_{1:t}) &\propto p(z_{t+1}, \mbx_{1:t}) = \alpha_{t+1}(z_{t+1}).
    \end{aligned}
$$

:::{admonition} Question
:class: tip
What does the normalizing constant
tell us?
:::

## Computing the posterior marginal distributions

The posterior marginal
distributions give the probability of the latent state $z_{t}$ given
*all the observations* up to time $T$.

$$
\begin{aligned}
        p(z_{t}  \mid \mbx_{1:T})
        &\propto \sum_{z_1=1}^K \cdots \sum_{z_{t-1}=1}^K \sum_{z_{t+1}=1}^K \cdots \sum_{z_T=1}^K p(\mbz_{1:T}, \mbx_{1:T}) \\
        &= \nonumber
        \bigg[ \sum_{z_{1}=1}^K \cdots \sum_{z_{t-1}=1}^K p(z_1) \prod_{s=1}^{t-1} p(\mbx_s  \mid z_s) \, p(z_{s+1}  \mid z_{s}) \bigg]
        \times  p(\mbx_t  \mid z_t)   \\
        &\qquad \times \bigg[ \sum_{z_{t+1}=1}^K \cdots \sum_{z_T=1}^K \prod_{u=t+1}^T p(z_{u}  \mid z_{u-1}) \, p(\mbx_u  \mid z_u) \bigg] \\
        &= \alpha_t(z_t) \times p(\mbx_t  \mid z_t) \times \beta_t(z_t)
    \end{aligned}
$$

where we have introduced the *backward messages*
$\beta_t(z_t)$.

## Computing the backward messages

The backward messages can be computed
recursively too,

$$
\begin{aligned}
        \beta_t(z_t)
        &\triangleq \sum_{z_{t+1}=1}^K \cdots \sum_{z_T=1}^K \prod_{u=t+1}^T p(z_{u}  \mid z_{u-1}) \, p(\mbx_u  \mid z_u) \\
        &= \sum_{z_{t+1}=1}^K p(z_{t+1}  \mid z_t) \, p(\mbx_{t_1}  \mid z_{t+1}) \left(\sum_{z_{t+2}=1}^K \cdots \sum_{z_T=1}^K \prod_{u=t+2}^T p(z_{u}  \mid z_{u-1}) \, p(\mbx_u  \mid z_u) \right) \\
        &= \sum_{z_{t+1}=1}^K p(z_{t+1}  \mid z_t) \, p(\mbx_{t_1}  \mid z_{t+1}) \, \beta_{t+1}(z_{t+1}).
    \end{aligned}
$$

For the base case, let $\beta_T(z_T) = 1$.


Let

$$
\begin{aligned}
        \boldsymbol{\beta}_t &=
        \begin{bmatrix}
        \beta_t(z_t = 1) \\
        \vdots \\
        \beta_t(z_t=K)
        \end{bmatrix}
    \end{aligned}
$$

be a vector in $\reals_+^K$. Then,

$$
\begin{aligned}
        \boldsymbol{\beta}_{t} &= \mbP(\boldsymbol{\beta}_{t+1} \odot \mbl_{t+1}).
    \end{aligned}
$$

Let $\boldsymbol{\beta}_T = \boldsymbol{1}_K$.

Now we have everything we need to compute the posterior marginal,

$$
\begin{aligned}
        p(z_t = k  \mid \mbx_{1:T}) &= \frac{\alpha_{t,k} \, l_{t,k} \, \beta_{t,k}}{\sum_{j=1}^K \alpha_{t,j} l_{t,j} \beta_{t,j}}.
    \end{aligned}
$$

We just derived the **forward-backward algorithm** for HMMs!

### What do the backward messages represent?

:::{admonition} Exercise
:class: tip
If the forward
messages represent the predictive probabilities
$\alpha_{t+1}(z_{t+1}) = p(z_{t+1}, \mbx_{1:t})$, what do the
backward messages represent?
:::

## Computing the posterior pairwise marginals

:::{admonition} Exercise
:class: tip
Use the forward
and backward messages to compute the posterior pairwise marginals
$p(z_t, z_{t+1}  \mid \mbx_{1:T})$.
:::

## Normalizing the messages for numerical stability

If you're working with
long time series, especially if you're working with 32-bit floating
point, you need to be careful.

The messages involve products of probabilities, which can quickly
overflow.

There's a simple fix though: after each step, re-normalize the messages
so that they sum to one. I.e replace

$$
\begin{aligned}
        \mbalpha_{t+1} &= \mbP^\top (\mbalpha_t \odot \mbl_t)
\end{aligned}
$$

with

$$
\begin{aligned}
        \overline{\mbalpha}_{t+1} &= \frac{1}{A_t} \mbP^\top (\overline{\mbalpha}_t \odot \mbl_t) \\
        A_t &= \sum_{k=1}^K \sum_{j=1}^K P_{jk} \overline{\alpha}_{t,j} l_{t,j}
            \equiv \sum_{j=1}^K \overline{\alpha}_{t,j} l_{t,j} \quad \text{(since $\mbP$ is row-stochastic)}.
\end{aligned}
$$

This leads to a nice interpretation: The normalized
messages are predictive likelihoods
$\overline{\alpha}_{t+1,k} = p(z_{t+1}=k  \mid \mbx_{1:t})$,
and the normalizing constants are
$A_t = p(\mbx_t  \mid \mbx_{1:t-1})$.

## EM for Hidden Markov Models

Now we can put it all together. To perform EM in an HMM,

-   **E step:** Compute the posterior distribution
    \begin{align*}
                q(\mbz_{1:T})
                &= p(\mbz_{1:T}  \mid \mbx_{1:T}; \mbTheta).
    \end{align*}

    (Really, run the **forward-backward algorithm** to get posterior marginals and pairwise marginals.)

-   **M step:** Maximize the ELBO wrt $\mbTheta$,

    \begin{align*}
                \mathcal{L}(\mbTheta)
                &= \mathbb{E}_{q(\mbz_{1:T})}\left[\log p(\mbx_{1:T}, \mbz_{1:T}; \mbTheta) \right]  + c \\
                &= \nonumber \mathbb{E}_{q(\mbz_{1:T})}\left[\sum_{k=1}^K \mathbb{I}[z_1=k]\log \pi_{0,k} \right] +
                \mathbb{E}_{q(\mbz_{1:T})}\left[\sum_{t=1}^{T-1} \sum_{i=1}^K \sum_{j=1}^K \mathbb{I}[z_t=i, z_{t+1}=j]\log P_{i,j} \right] \\
                &\qquad + \mathbb{E}_{q(\mbz_{1:T})}\left[\sum_{t=1}^T \sum_{k=1}^K \mathbb{I}[z_t=k]\log p(\mbx_t; \theta_k) \right]
    \end{align*}

For exponential family observations, the M-step only requires expected
sufficient statistics.

:::{admonition} Questions
:class: tip

-   How can we sample the posterior?
-   How can we find the posterior mode?
-   How can we choose the number of states?
-   What if my transition matrix is sparse?
:::

## Conclusion

HMMs add temporal dependencies to the latent states of mixture models. They're a simple yet powerful model for sequential data.

The emission distribution can be extended in many ways. For example, we could include temporal dependencies in the emissions via an autoregressive HMM, or condition on external covariates as in an input-output HMM

Like mixture models, we can derive efficient **stochastic EM** algorithms for HMMs, which keep rolling averages of sufficient statistics across mini-batches (e.g., individual trials from a collection of sequences).

It's always good to implement models and algorithms yourself at least once, like you will in Homework 3. Going forward, you should check out our implementations in [Dynamax](https://probml.github.io/dynamax/) and [SSM](https://lindermanlab.github.io/ssm-docs/)!
