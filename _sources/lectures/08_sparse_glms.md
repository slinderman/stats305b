# Sparse GLMs

One reason we like linear and generalized linear models is that the parameters are readily interpretable. The parameter $\beta_j$ relates changes in covariate $x_j$ to changes in the natural parameter of the response distribution. One common application of such models is for _variable selection_, finding a subset of covariates that are most predictive of the response. To that end, we would like our estimates, $\hat{\mbbeta}$, to be _sparse_. When we have a vast number covariates &mdash; as in genome-wide association studies (GWAS) where we aim to predict a trait given thousands of single nucleotide polymorphisms (SNPs) in the genome &mdash; sparse solutions help focus our attention on the most relevant covariates. 

## Lasso Regression

The Lasso yields sparse solutions for linear models by solving an $\ell_1$ regularized linear regression problem. The goal is to minimize, 
\begin{align*}
\cL(\mbbeta) 
&= \frac{1}{2} \|\mby - \mbX \mbbeta\|_2^2 + \lambda \|\mbbeta\|_1 \\
&= \frac{1}{2} \sum_{i=1}^n (y_i - \mbx_i^\top \mbbeta)^2 + \lambda \sum_{j=1}^p |\beta_j|.
\end{align*}
This is a convex objective function! 

It's tempting to just use vanilla gradient descent to find the minimizer,
\begin{align*}
\mbbeta^{(t+1)} &\leftarrow \mbbeta^{(t)} - \alpha_t \nabla \cL(\mbbeta^{(t)}),
\end{align*}
where $\alpha_t \in \reals_+$ is the step size at iteration $t$. 

One way to think about this update is as the solution to a quadratic minimization problem,
\begin{align*}
\mbbeta^{(t+1)} &\leftarrow 
\arg \min_{\mbz} \cL(\mbbeta^{(t)}) + \nabla \cL(\mbbeta^{(t)})^\top (\mbz - \mbbeta^{(t)}) + \frac{1}{2 \alpha_t} \| \mbz - \mbbeta^{(t)}\|_2^2
\end{align*}
We can think of the surrogate problem as a second order approximation of the objective in which the Hessian is replaced with $\frac{1}{\alpha_t} \mbI$. 

The $\cO(1/\epsilon)$ convergence rate we discussed for gradient descent in the context of [logistic regression](./03_logreg.md) applied to $L$-smooth functions; i.e., functions that are continuously differentiable with $L$-Lipschitz gradients. (For twice differentiable objective functions, the Lipschitz condition is equivalent to an upper bound on the eigenvalues of the Hessian.) 

Unfortunately, the Lasso objective it is not continuously differentiable: the **gradient at $\beta_j=0$ is discontinuous** due to the absolute value in the $\ell_1$ norm. We could instead use _subgradient_ descent, where we take a step in the direction of any subgradient of $\cL$, but that approach tends to be much slower, with convergence rates of only $\cO(1/\epsilon^2)$. Thankfully, we can do much better than that for the Lasso using proximal gradient descent!

## Proximal Gradient Descent

Proximal gradient descent is an optimization algorithm for convex objectives that decompose into a differentiable part and a non-differentiable part,
\begin{align*}
\cL(\mbbeta) &= \cL_{\mathsf{d}}(\mbbeta) + \cL_{\mathsf{nd}}(\mbbeta)
\end{align*}
where $\cL_{\mathsf{d}}$ is convex and _differentiable_, whereas $\cL_{\mathsf{nd}}$ is convex but _not differentiable_. 

The idea is to stick as close to vanilla gradient descent as possible, while correcting for the non-differentiable part of the objective. To that end, let's apply the quadratic approximation logic to just the differentiable part so that our update becomes,
\begin{align*}
\mbbeta^{(t+1)} &\leftarrow 
\arg \min_{\mbz} \cL_{\mathsf{d}}(\mbbeta^{(t)}) + \nabla \cL_{\mathsf{d}}(\mbbeta^{(t)})^\top (\mbz - \mbbeta^{(t)}) + \frac{1}{2 \alpha_t} \| \mbz - \mbbeta^{(t)}\|_2^2 + \cL_{\mathsf{nd}}(\mbbeta^{(t)}) \\
&= \arg \min_{\mbz} \frac{1}{2 \alpha_t} \| \mbz - (\mbbeta^{(t)} - \alpha_t \nabla \cL_{\mathsf{d}}(\mbbeta^{(t)})) \|_2^2 + \cL_{\mathsf{nd}}(\mbz) \\
\end{align*}
The resulting update balances two parts:
1. Stay close to the vanilla gradient descent update, $\mbbeta^{(t)} - \alpha_t \nabla \cL_{\mathsf{d}}(\mbbeta^{(t)})$.
2. Also minimize the non-differentiable part of the objective, $\cL_{\mathsf{nd}}(\mbbeta^{(t)})$.

As a sanity check, note that we recover vanilla gradient descent with $\cL_{\mathsf{nd}}(\mbbeta^{(t)}) = 0$.

### Proximal Mapping
We call the function,
\begin{align}
\mathrm{prox}(\mbu; \alpha_t) 
&= \arg \min_{\mbz} \frac{1}{2 \alpha_t} \| \mbz - \mbu \|_2^2 + \cL_{\mathsf{nd}}(\mbz)
\end{align}
the **proximal mapping**. 

:::{admonition} Notes
:class: tip
- The proximal mapping depends on the form of the non-differentiable part of the objective, even though we have suppressed that in the notation.
- However, it does _not_ depend on the form of the continuous part of the objective.
:::

### Algorithm
With this definition, the proximal gradient descent algorithm is,

```{prf:algorithm} Proximal Gradient Descent
:label: prox_grad

**Input:** Initial parameters $\mbbeta^{(0)}$, proximal mapping $\mathrm{prox}(\cdot; \cdot)$.

- **For** $t=1,\ldots, T$

    - Set $\mbbeta^{(t)} \leftarrow \mathrm{prox}(\mbbeta^{(t-1)} - \alpha_t \nabla \cL_{\mathsf{d}}(\mbbeta^{(t-1)}); \alpha_t)$.

**Return** $\mbbeta^{(T)}$.
```

So far, it's not obvious that this framing is helpful. We still have a potentially challenging optimization problem to solve in computing the proximal mapping. However, for many problems of interest, the proximal mapping has simpled closed solutions.

### Proximal Gradient Descent for Lasso Regression
Consider the Lasso problem. The objective decomposes into convex differentiable and non-differentiable parts,
\begin{align*}
\cL_{\mathsf{d}}(\mbbeta) &= \frac{1}{2} \|\mby - \mbX \mbbeta\|_2^2 \\
\cL_{\mathsf{nd}}(\mbbeta) &= \lambda \|\mbbeta\|_1.
\end{align*}

### Proximal Mapping
The proximal mapping is,
\begin{align*}
\mathrm{prox}(\mbu; \alpha_t) 
&= \arg \min_{\mbz} \frac{1}{2 \alpha_t} \| \mbz - \mbu \|_2^2 + \lambda \|\mbz\|_1 \\
&= \arg \min_{\mbz} \sum_{j=1}^p  \frac{1}{2 \alpha_t} (z_j - u_j)^2 + \lambda |z_j| 
\end{align*}
It separates into optimization problems for each coordinate! 

Isolate one coordinate and consider the case where $z_j > 0$. Completing the square yields,
\begin{align*}
f(z_j) 
&= \frac{1}{2 \alpha_t} (z_j - u_j)^2 + \lambda z_j  \\
&= \frac{1}{2 \alpha_t} z_j^2 - (\frac{u_j}{\alpha_t} - \lambda) z_j + \mathrm{const}\\
&= \frac{1}{2 \alpha_t} (z_j - (u_j - \alpha_t \lambda))^2 + \mathrm{const}
\end{align*}
which is minimized at $z_j = \max\{0, u_j - \alpha_t \lambda \}$. The same logic applied to the case where $z_j \leq 0$ yields, $z_j = \min\{0, u_j + \alpha_t \lambda \}$. Combined, we have that,
\begin{align*}
[\mathrm{prox}(\mbu; \alpha_t)]_j 
&= \begin{cases}
u_j - \alpha_t \lambda & \text{if } u_j > \alpha_t \lambda \\
0 &\text{if } |u_j| < \alpha_t \lambda \\
u_j + \alpha_t \lambda &\text{if } u_j < -\alpha_t \lambda
\end{cases}
\triangleq S_{\alpha_t \lambda}(u_j)
\end{align*}
This is called the **soft-thresholding operator**. (Plot it!)

### Iterative Soft-Thresholding Algorithm
Now let's plug in the gradient of the differentiable part,
\begin{align*}
\nabla \cL_{\mathsf{d}}(\mbbeta) &= \mbX^\top (\mby - \mbX \mbbeta).
\end{align*}
Substituting this into the proximal gradient descent algorithm yields what is sometimes called the **iterative soft-thresholding algorithm (ISTA)**,

```{prf:algorithm} Iterative Soft-Thresholding
:label: ista

**Input:** Initial parameters $\mbbeta^{(0)}$, covariates $\mbX \in \reals^{n \times p}$, responses $\mby \in \reals^n$

- **For** $t=1,\ldots, T$

    - Set $\mbbeta^{(t)} \leftarrow S_{\alpha_t \lambda}(\mbbeta^{(t-1)} - \alpha_t \mbX^\top (\mby - \mbX \mbbeta^{(t-1)}))$.

**Return** $\mbbeta^{(T)}$.
```

### Convergence 
If $\nabla \cL_{\mathsf{d}}$ is $L$-smooth then proximal gradient descent with fixed step size $\alpha_t = 1/L$ then,
\begin{align*}
f(\mbbeta^{(t)}) - f(\mbbeta^\star) \leq \frac{L}{2 t} \|\mbbeta^{(0)} - \mbbeta^\star \|_2^2,
\end{align*}
so it matches the gradient descent convergence rate of $\cO(1/\epsilon)$. (With Nesterov's accelerated gradient techniques, you can speed this up to $\cO(1/\sqrt{\epsilon})$.

<!-- 
:::{admonition} Note about intercepts
:class: warning
Note that we have explicitly separated out the intercept in the objective since we typically don't include that term in the regularizer.
::: -->

## Proximal Newton Method

One great thing about proximal gradient descent is its generality. We could easily apply it to $\ell_1$ regularized GLMs, substituting the gradient of the negative log likelihood, which also has a simple closed form expression. The proximal operator remains the same, and we obtain the same converge rates as gradient descent on standard GLMs.

However, we saw that Newton's method yielded significantly faster convergence rates of $\cO(\log \log \frac{1}{\epsilon})$. Can we obtain similar performance for $\ell_1$ regularized GLMs? 

To obtain a **proximal Newton method**, we proceed in the same fashion as above, but rather than approximating the second order term with $\alpha_t^{-1} \mbI$, we will use the Hessian of $\cL_{\mathsf{d}}$.  That leads to a proximal mapping of the form,
\begin{align*}
\mathrm{prox}(\mbu; \mbH_t) 
&= \arg \min_{\mbz} \frac{1}{2} \| \mbz - \mbu \|_{\mbH_t}^2 + \cL_{\mathsf{nd}}(\mbz)
\end{align*}
where $\|\mbx \|_{\mbH_t}^2 = \mbx^\top \mbH_t \mbx$ is a squared norm induced by the positive definite matrix $\mbH_t$. 

:::{admonition} Note
:class: tip
Note that proximal mapping for proximal gradient descent corresponds to the special case in which $\mbH_t = \frac{1}{\alpha_t} \mbI$. 
:::


Let $\mbg_t = \nabla \cL_{\mathsf{d}}(\mbbeta^{(t)})$ and $\mbH_t = \nabla^2 \cL_{\mathsf{d}}(\mbbeta^{(t)})$ denote the gradient and Hessian, respectively. The _undamped_ proximal Newton update is,
\begin{align*}
\hat{\mbbeta}^{(t+1)}
&\leftarrow \arg \min_{\mbz} \cL_{\mathsf{d}}(\mbbeta^{(t)}) + (\mbz - \mbbeta^{(t)})^\top \mbg_t + \frac{1}{2} (\mbz - \mbbeta^{(t)})^\top \mbH_t (\mbz - \mbbeta^{(t)}) + \cL_{\mathsf{nd}}(z) \\
&= \arg \min_{\mbz} \frac{1}{2} \| \mbz - (\mbbeta^{(t)} -  \mbH_t^{-1} \mbg_t)\|_{\mbH_t}^2 + \cL_{\mathsf{nd}}(\mbz)  \\
&= \mathrm{prox}(\mbbeta^{(t)} -  \mbH_t^{-1} \mbg_t; \mbH_t)
\end{align*}

As with Newton's method, however, we often need to use damped updates,
\begin{align*}
\mbbeta^{(t+1)} &= \mbbeta^{(t)} + \alpha_t (\hat{\mbbeta}^{(t+1)} - \mbbeta^{(t)}),
\end{align*}
with step size $\alpha_t \in [0, 1]$.

The challenge, as we will see below, is that solving the proximal Newton mapping can be more challenging. 

## Proximal Newton with $\ell_1$ Regularization

Let's consider the proximal Newton mapping for $\ell_1$ regularized GLMs, like logistic regression. Here, the non-differentiable part of the objective is $\cL_{\mathsf{nd}}(\mbbeta) = \lambda \|\mbbeta\|$. Unfortunately, the proximal Newton update no longer has a closed form solution because when we introduce the Hessian, the problem no longer separates across coordinates:
\begin{align*}
\mathrm{prox}(\mbu; \mbH_t) 
&= \arg \min_{\mbz} \frac{1}{2} \| \mbz - \mbu \|_{\mbH_t}^2 + \lambda \|\mbz\|_1 \\
&= \arg \min_{\mbz} \frac{1}{2} (\mbz - \mbu)^\top \mbH_t (\mbz - \mbu) + \lambda \|\mbz\|_1 .
\end{align*}

However, if we fix all coordinates but one, the problem is tractable. First, expand the quadratic form in the proximal mapping,
\begin{align*}
\mathrm{prox}(\mbu; \mbH_t) 
&= \arg \min_{\mbz} \sum_{j=1}^p \left[ \frac{1}{2}  H_{t,j,j} (z_j - u_j)^2 + \sum_{k=j+1}^p H_{t,j,k} (z_j - u_j) (z_k - u_k) + \lambda |z_j| \right].
\end{align*}
As a function of a single coordinate $z_j$, the objective is,
\begin{align*}
f(z_j) 
&= 
\frac{1}{2} H_{t,j,j} z_j^2 + \left(-H_{t,j,j} u_j + \sum_{k\neq j}^p H_{t,j,k} (z_k - u_k)\right) z_j + \lambda |z_j| \\
&= 
\frac{1}{2 H_{t,j,j}^{-1}} \left(z_j - \mu_j \right)^2 + \lambda |z_j| + \mathrm{const}
\end{align*}
where 
\begin{align*}
\mu_j \triangleq H_{t,j,j}^{-1} \left(H_{t,j,j} u_j - \sum_{k\neq j}^p H_{t,j,k} (z_k - u_k)\right)
\end{align*}
This is the same problem we solved above for Lasso regression! If we freeze the values of $\mbz_{\neg j}$, the minimizer as a function of $z_j$ is,
\begin{align*}
z_j &\leftarrow S_{\lambda / H_{t,j,j}}(\mu_j).
\end{align*}

To implement the proximal Newton step, we perform coordinate descent, iteratively minimizing with respect to one coordinate at a time until convergence.

```{prf:algorithm} Proximal Newton Mapping for $\ell_1$ Regularized Logistic Regression
:label: prox-newton

**Input:** initial guess $\mbz^{(0)}$, target $\mbu \in \reals^p$, Hessian $\mbH_t \in \reals^{p \times p}_{\succeq 0}$, num iterations $M$

- **For** $m=1,\ldots, M$
    - **For** $j=1, \ldots, p$
        - Set 
        \begin{align*}
        \mu_j^{(m)} = H_{t,j,j}^{-1} \left(H_{t,j,j} u_j - \sum_{k < j} H_{t,j,k} (z_k^{(m)} - u_k) - \sum_{k > j} H_{t,j,k} (z_k^{(m-1)} - u_k) \right)
        \end{align*}
        - Set $z_j^{(m)} \leftarrow S_{\lambda/H_{t,j,j}}(\mu_j^{(m)})$.

**Return** $\mbz^{(M)}$.
```

As with regular Newton's method, proximal Newton exhibits local quadratic convergence to obtain error $\epsilon$ in $\cO(\log \log 1/\epsilon)$ iterations. Though here, each iteration requires an inner coordinate descent loop to solve the proximal mapping. 

:::{admonition} Note
:class: warning
In practice, you may need to also implement a backtracking line search to choose the step size $\alpha_t$, since you may not start in the local quadratic regime. Logistic regression with decent initialization is reasonably well behaved, but Poisson regression with log link functions can be sensitive. 
:::

### Simplifications for GLMs

Another way to arrive at the same answer is to leverage the relationship between Newton's method and iteratively reweighted least squares. We cast each Newton update as the solution to a weighted least squares problem. In the $\ell_1$ regularized case, each _proximal_ Newton update involves solving a weighted least squares problem with an extra $\ell_1$ norm penalty. That problem can be solved by coordinate descent, and each update amounts to solving a scalar quadratic minimization problem as above. The difference is that the abstract terms in the proximal Newton mapping can be identified as weights and residuals.

There are also lots of speed-ups to be had by exploiting problem-specific structure. See {cite:t}`friedman2010regularization` for details.

## Conclusion

The proximal methods disussed today are what run behind the scenes of modern packages for sparse linear and logistic regression. In particular, [`sklearn.linear_model.Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) uses a fast coordinate descent algorithm like discussed above, and [GLMNet](https://glmnet.stanford.edu/articles/glmnet.html) {cite:p}`friedman2010regularization` uses a proximal Newton algorithm with coordinate descent for the proximal step. 
