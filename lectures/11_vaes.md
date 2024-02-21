# Variational Autoencoders

## Generative Model
Variational Autoencodres (VAEs) are "deep" but conceptually simple generative models,
\begin{align*}
    \mbz_n &\sim \mathrm{N}(\mbzero, \mbI) \\
    \mbx_n &\sim \mathrm{N}(g(\mbz_n; \mbtheta), \mbI)
\end{align*}
where $g: \reals^H \to \reals^D$ is a nonlinear mapping from $\mbz_n \in \reals^H$ to $\E[\mbx_n] \in \reals^D$, parameterized by $\mbtheta$.

We will assume $g$ is a simple **feedforward neural network** of the form,
\begin{align*}
    g(\mbz; \mbtheta) &= g_L(g_{L-1}(\cdots g_1(\mbz) \cdots))
\end{align*}
where each **layer** is a cascade of a linear mapping followed by an element-wise nonlinearity (except for the last layer, perhaps). For example,
\begin{align*}
    g_\ell(\mbu_{\ell}) = \mathrm{relu}(\mbW_\ell \mbu_{\ell} + \mbb_\ell); \qquad \mathrm{relu}(a) = \max(0, a).
\end{align*}
The generative parameters consist of the weights and biases, $\mbtheta = \{\mbW_\ell, \mbb_\ell\}_{\ell=1}^L$.


## Learning and Inference

We have two goals. The **learning goal** is to find the parameters that **maximize the marginal likelihood of the data**,
\begin{align*}
    \mbtheta^\star 
    &= \arg \max_{\mbtheta} p(\mbX; \mbtheta)\\
    &= \arg \max_{\mbtheta} \prod_{n=1}^N \int p(\mbx_n \mid \mbz_n; \mbtheta) \, p(\mbz_n; \mbtheta) \dif \mbz_n
\end{align*}
    
The **inference goal** is to find the **posterior distribution of latent variables**,
\begin{align*}
    p(\mbz_n \mid \mbx_n; \mbtheta) 
    &= \frac{p(\mbx_n \mid \mbz_n; \mbtheta) \, p(\mbz_n; \mbtheta)}{\int p(\mbx_n \mid \mbz_n'; \mbtheta) \, p(\mbz_n'; \mbtheta)\dif \mbz_n'}
\end{align*}

Both goals require an integral over $\mbz_n$, but that is intractable for deep generative models.


## The Evidence Lower Bound (ELBO)

**Idea:** Use the ELBO to get a bound on the marginal probability and maximize that instead.
\begin{align*}
\log p(\mbX ; \mbtheta) 
&= \sum_{n=1}^N \log p(\mbx_n; \mbtheta) \\
&\geq \sum_{n=1}^N \log p(\mbx_n; \mbtheta) - \KL{q_n(\mbz_n; \mblambda_n)}{p(\mbz_n \mid \mbx_n; \mbtheta)} \\
&= \sum_{n=1}^N \underbrace{\E_{q_n(\mbz_n)}\left[ \log p(\mbx_n, \mbz_n; \mbtheta) - \log q_n(\mbz_n; \mblambda_n) \right]}_{\text{"local ELBO"}} \\
&\triangleq \sum_{n=1}^N \cL_n(\mblambda_n, \mbtheta) \\
&= \cL(\mblambda, \mbtheta)
\end{align*}
where $\mblambda = \{\mblambda_n\}_{n=1}^N$. 

Here, I've written the ELBO as a sum of _local ELBOs_ $\cL_n$.

## Variational Inference
The ELBO is still maximized (and the bound is tight) when each $q_n$ is equal to the true posterior,
\begin{align*}
    q_n(\mbz_n; \mblambda_n) &= p(\mbz_n \mid \mbx_n, \mbtheta).
\end{align*}
Unfortunately, the posterior no longer has a simple, closed form.

:::{admonition} Question
The deep generative model above has a Gaussian prior on $\mbz_n$ and a Gaussian likelihood for $\mbx_n$ given $\mbz_n$. Why isn't the posterior Gaussian?
:::

Nevertheless, we can still constrain $q_n$ to belong to a simple family. For example, we could constrain it to be Gaussian and seek the best Gaussian approximation to the posterior. This is sometimes called **fixed-form variational inference**. Let,
\begin{align*}
    \cQ = \left\{q: q(\mbz; \mblambda) = \mathrm{N}\big(\mbz \mid \mbmu, \diag(\mbsigma^2)\big) \text{ for } \mblambda = (\mbmu, \log \mbsigma^2) \in \reals^{2H} \right\}
\end{align*}

Then, for fixed parameters $\mbtheta$, the best $q_n$ in this **variational family** is,
\begin{align*}
    q_n^\star 
    &= \arg \min_{q_n \in \cQ} \KL{q_n(\mbz_n; \mblambda_n)}{p(\mbz_n \mid \mbx_n; \mbtheta)} \\
    &= \arg \max_{\mblambda_n \in \reals^{2H}} \cL_n(\mblambda_n, \mbtheta).
\end{align*}

We can maximize the ELBO with **stochastic gradient ascent** using unbiased estimates of the gradient, $\widehat{\nabla}_{\mblambda_n} \cL(\mblambda_n, \mbtheta)$, e.g., using the **score-function** or the **pathwise gradient estimators**.


## Variational Expectation-Maximization (vEM)
Now we can introduce a new algorithm.

:::{prf:algorithm} Variational EM (vEM)
Repeat until either the ELBO or the parameters converges:
1. **M-step:** Set $\mbtheta \leftarrow \arg \max_{\mbtheta} \cL(\mblambda, \mbtheta)$
2. **E-step:** For $n=1,\ldots,N$ :
    
    a. Set $\mblambda_n \leftarrow \arg \max_{\mblambda_n \in \mbLambda} \cL_n(\mblambda_n, \mbtheta)$

3. Compute (an estimate of) the ELBO $\cL(\mblambda, \mbtheta)$.
:::

Unfortunately, none of these steps will have closed form solutions, so we'll have to use approximations. 


### Generic M-step with Stochastic Gradient Ascent

For exponential family mixture models, the M-step had a closed form solution. For deep generative models, we need a more general approach.
    
If the parameters are unconstrained and the ELBO is differentiable wrt $\mbtheta$, we can use stochastic gradient ascent. 
\begin{align*}
    \mbtheta &\leftarrow \mbtheta + \alpha \nabla_{\mbtheta} \cL(q, \mbtheta) \\
    &= \mbtheta + \alpha \sum_{n=1}^N \mathbb{E}_{q(\mbz_n; \mblambda_n)} \left[ \nabla_{\mbtheta} \log p(\mbx_n, \mbz_n; \mbtheta) \right]
\end{align*}
    
Note that the expected gradient wrt $\mbtheta$ can be computed using ordinary Monte Carlo --- nothing fancy needed!
    
### The Variational E-step
    
Assume $\cQ$ is the family of Gaussian distributions with diagonal covariance:
\begin{align*}
    \cQ = \left\{q: q(\mbz; \mblambda) = \mathrm{N}\big(\mbz \mid \mbmu, \diag(\mbsigma^2)\big) \text{ for } \mblambda = (\mbmu, \log \mbsigma^2) \in \reals^{2H} \right\}
\end{align*}
This family is indexed by **variational parameters** $\mblambda_n = (\mbmu_n, \log \mbsigma_n^2) \in \reals^{2H}$.
    
To perform SGD, we need an unbiased estimate of the gradient of the local ELBO, but
\begin{align*}
\nabla_{\mblambda_n} \cL_n(\mblambda_n, \mbtheta) 
&= \nabla_{\mblambda_n} \E_{q(\mbz_n; \mblambda_n)} \left[ \log p(\mbx_n, \mbz_n; \mbtheta) - \log q(\mbz_n; \mblambda_n) \right] \\
&\textcolor{red}{\neq} \;  \E_{q(\mbz_n; \mblambda_n)} \left[ \nabla_{\mblambda_n} \left(\log p(\mbx_n, \mbz_n; \mbtheta) - \log q(\mbz_n; \mblambda_n)\right) \right].
\end{align*}

### Reparameterization Trick
One way around this problem is to use the **reparameterization trick**, aka the **pathwise gradient estimator**. Note that,
\begin{align*}
    \mbz_n \sim q(\mbz_n; \mblambda_n) \quad \iff \quad
    \mbz_n &= r(\mblambda_n, \mbepsilon), \quad \mbepsilon \sim \cN(\mbzero, \mbI) 
\end{align*}
where $r(\mblambda_n, \mbepsilon) = \mbmu_n + \mbsigma_n \mbepsilon$ is a reparameterization of $\mbz_n$ in terms of parameters $\mblambda_n$ and noise $\mbepsilon$.

We can use the **law of the unconscious statistician** to rewrite the expectations as,
\begin{align*}
    \E_{q(\mbz_n; \mblambda_n)} \left[h(\mbx_n, \mbz_n, \mbtheta, \mblambda_n) \right]
    &= \E_{\mbepsilon \sim \cN(\mbzero, \mbI)} \left[h(\mbx_n, r(\mblambda_n, \mbepsilon), \mbtheta, \mblambda_n) \right]
\end{align*}
where 
\begin{align*}
h(\mbx_n, \mbz_n, \mbtheta, \mblambda_n) = \log p(\mbx_n, \mbz_n; \mbtheta) - \log q(\mbz_n; \mblambda_n).
\end{align*} 
The distribution that the expectation is taken under no longer depends on the parameters $\mblambda_n$, so we can simply take the gradient inside the expectation,
\begin{align*}
    \nabla_{\mblambda} \E_{q(\mbz_n; \mblambda_n)} \left[h(\mbx_n, \mbz_n, \mbtheta, \mblambda_n) \right]
    &=  \E_{\mbepsilon \sim \cN(\mbzero, \mbI)} \left[\nabla_{\mblambda_n} h(\mbx_n, r(\mblambda_n, \mbepsilon), \mbtheta, \mblambda_n) \right]
\end{align*}
Now we can use Monte Carlo to obtain an unbiased estimate of the final expectation!
<!-- \begin{align*}
    \widehat{\nabla}_{\mblambda} \E_{q(\mbtheta; \mblambda)} \left[h(\mbtheta, \mblambda) \right]
    &= \frac{1}{M} \sum_{m=1}^M \nabla_{\mblambda} h(r(\mblambda, \mbepsilon_m), \mblambda); & \epsilon_m \iid{\sim} \cN(\mbzero, \mbI)
\end{align*} -->
    
    
<!-- \item Last lecture we introduced the score-function and pathwise gradient estimators to tackle this problem. For example,
\begin{align*}
\nabla_{\mblambda_n} \cL_n(\mblambda_n, \mbtheta) 
&= \nabla_{\mblambda_n} \E_{q(\mbz_n; \mblambda_n)} \left[ \log p(\mbx_n, \mbz_n; \mbtheta) - \log q(\mbz_n; \mblambda_n) \right] \\
&= \E_{\mbepsilon_n \sim \mathrm{N}(\mbzero, \mbI)} \left[ \nabla_{\mblambda_n} \left(\log p(\mbx_n, r(\mblambda_n, \mbepsilon_n); \mbtheta) - \log q(r(\mblambda_n, \mbepsilon_n); \mblambda_n)\right) \right]
\end{align*}
where $r(\mblambda_n, \mbepsilon_n) = \mbmu_n + \mbsigma_n \mbepsilon_n$. -->

### Working with mini-batches of data
We can view the ELBO as an expectation over data indices,
\begin{align*}
    \cL(\mblambda, \mbtheta) 
    &= \sum_{n=1}^N \cL_n(\mblambda_n, \mbtheta) \\
    &= N \, \E_{n \sim \mathrm{Unif}([N])}[\cL_n(\mblambda_n, \mbtheta)].
\end{align*}
We can use Monte Carlo to approximate the expectation (and its gradient) by drawing **mini-batches** of data points at random.

In practice, we often cycle through mini-batches of data points deterministically. Each pass over the whole dataset is called an **epoch.**


### Algorithm
Now we can add some detail to our variational expectation maximization algorithm.

:::{prf:algorithm} Variational EM (with the reparameterization trick)

For epoch $i=1,\ldots,\infty$:

For $n=1,\ldots,N$:
    
1. Sample $\epsilon_n^{(m)} \iid{\sim} \cN(\mbzero, \mbI)$ for $m=1,\ldots,M$.

2. **M-Step**:

    a. Estimate
    \begin{align*}
    \hat{\nabla}_{\mbtheta} \cL_n(\mblambda_n, \mbtheta)
    &= 
    \frac{1}{M} \sum_{m=1}^M \left[ \nabla_{\mbtheta} \log p(\mbx_n, r(\mblambda_n, \mbepsilon_n^{(m)}); \mbtheta) \right]
    \end{align*}
    
    b. Set $\mbtheta \leftarrow \mbtheta + \alpha_i N \hat{\nabla}_{\mbtheta} \cL_n(\mblambda_n, \mbtheta)$

3. **E-step:** 

    a. Estimate 
    \begin{align*}
    \hat{\nabla}_{\mblambda} \cL_n(\mblambda_n, \mbtheta) 
    &= \frac{1}{M} \sum_{m=1}^M \nabla_{\mblambda} \left[\log p(\mbx_n, r(\mblambda_n, \mbepsilon_n^{(m)}); \mbtheta) - \log q(r(\mblambda_n, \mbepsilon_n^{(m)}), \mblambda_n) \right]
    \end{align*} 
    
    b. Set $\mblambda_n \leftarrow \mblambda_n + \alpha_i \hat{\nabla}_{\mblambda} \cL_n(\mblambda_n, \mbtheta)$.

4. Estimate the ELBO 
    \begin{align*}
    \hat{\cL}(\mblambda, \mbtheta) 
    &= \frac{N}{M} \sum_{m=1}^M \log p(\mbx_n, r(\mblambda_n, \mbepsilon_n^{(m)}); \mbtheta) - \log q(r(\mblambda_n, \mbepsilon_n^{(m)}); \mblambda_n)
    \end{align*}

5. Decay step size $\alpha_i$ according to schedule.
:::


## Amortized Inference

Note that vEM involves optimizing separate variational parameters $\mblambda_n$ for each data point. For large datasets where we are optimizing using mini-batches of data points, this leads to a strange asymmetry: we update the generative model parameters $\mbtheta$ every mini-batch, but we only update the variational parameters for the $n$-th data point once per epoch. Is there any way to share information across data points?

Note that the optimal variational parameters are just a function of the data point and the model parameters,
\begin{align*}
    \mblambda_n^\star &= \arg \min_{\mblambda_n} \KL{q(\mbz_n; \mblambda_n)}{p(\mbz_n \mid \mbx_n, \mbtheta)} 
    \triangleq f^\star(\mbx_n, \mbtheta).
\end{align*}
for some implicit and generally nonlinear function $f^\star$.

VAEs learn an approximation to $f^\star(\mbx_n, \mbtheta)$ with an **inference network**, a.k.a. **recognition network** or **encoder**.
    
The inference network is (yet another) neural network that takes in a data point $\mbx_n$ and outputs variational parameters $\mbz_n$,
\begin{align*}
    \mblambda_n & \approx f(\mbx_n, \mbphi),
\end{align*}
where $\mbphi$ are the weights of the network.
    
The advantage is that the inference network shares information across data points &mdash; it _amortizes_ the cost of inference, hence the name. The disadvantage is the output will not minimize the KL divergence. However, in practice we might tolerate a worse variational posterior and a weaker lower bound if it leads to faster optimization of the ELBO overall.


### Putting it all together
Logically, I find it helpful to distinguish between the E and M steps, but with recognition networks and stochastic gradient ascent, the line is blurred.

The final algorithm looks like this. 

:::{prf:algorithm} Variational EM (with amortized inference)

Repeat until either the ELBO or the parameters converges:

1. Sample data point $n \sim \mathrm{Unif}(1, \ldots, N)$. [Or a minibatch of data points.]

2. Estimate the local ELBO $\cL_n(\mbphi, \mbtheta)$ with Monte Carlo. [Note: it is a function of $\mbphi$ instead of $\mblambda_n$.]

3. Compute unbiased Monte Carlo estimates of the gradients $\widehat{\nabla}_{\mbtheta} \cL_n(\mbphi, \mbtheta)$ and $\widehat{\nabla}_{\mbphi} \cL_n(\mbphi, \mbtheta)$. 
[The latter requires the reparameterization trick.]

3. Set
\begin{align*}
    \mbtheta &\leftarrow \mbtheta + \alpha_i \widehat{\nabla}_{\mbtheta} \cL_n(\mbphi, \mbtheta) \\
    \mbphi &\leftarrow \mbphi + \alpha_i \widehat{\nabla}_{\mbphi} \cL_n(\mbphi, \mbtheta)
\end{align*}
with step size $\alpha_i$ decreasing over iterations $i$ according to a valid schedule.

:::