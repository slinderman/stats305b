## Generalized Linear Models
    
Logistic regression was a special case of a more general class of models called _generalized linear models_ (GLMs). In a GLM, the conditional distribution $p(Y \mid X)$ is modeled as an exponential family distribution whose mean parameter is a function of $X$. For example, if $Y \in \naturals$, we could model it with a Poisson GLM; if $Y \in \{1,\ldots,K\}$, we could model it as a categorical GLM. It turns out that many of the nice properties of logistic regression carry over to the more general case.

## Model
To construct a generalized linear model with exponential family observations, we set 
\begin{align*}
    \E[y_i \mid \mbx_i] &= f(\mbbeta^\top \mbx_i).
\end{align*}
    
From above, this implies,
\begin{align*}
    \nabla A(\eta_i) &= f(\mbbeta^\top \mbx_i) \\
    \Rightarrow \eta_i &= [\nabla A]^{-1} \big( f(\mbbeta^\top \mbx_i) \big),
\end{align*}
when $\nabla A(\cdot)$ is invertible. (In this case, the exponential family is said to be **minimal**).
    
The **canonical mean function** is $f(\cdot) = \nabla A(\cdot)$ so that $\eta_i = \mbbeta^\top \mbx_i$.
    
The (canonical) **link function** is the inverse of the (canonical) mean function.

    
### Logistic regression revisited
Consider the Bernoulli distribution once more. The gradient of the log normalizer is,
\begin{align*}
    \nabla A(\eta) &= \nabla \log (1 + e^\eta) 
    = \frac{e^\eta}{1+ e^\eta}
\end{align*}
This is the logistic function!

Thus, logistic regression is a Bernoulli GLM with the canonical mean function.


## Canonical case

Canonical mean functions lead to nice math. Consider the log joint probability,
\begin{align*}
    \cL(\mbbeta) 
    &= \sum_{i=1}^n \langle t(y_i), \eta_i \rangle - A(\eta_i)  + c \\
    &= \sum_{i=1}^n \langle t(y_i), \mbbeta^\top \mbx_i \rangle - A(\mbbeta^\top \mbx_i) + c,
\end{align*}
where we have assumed a canonical mean function so $\eta_i = \mbbeta^\top \mbx_i$.

The gradient is,
\begin{align*}
    \nabla \cL(\mbbeta) 
    &= \sum_{i=1}^n \langle t(y_i), \mbx_i \rangle - \langle \nabla A(\mbbeta^\top \mbx_i), \, \mbx_i \rangle\\
    &= \sum_{i=1}^n \langle t(y_i) - \E[t(y_i)], \, \mbx_i \rangle
\end{align*}

In many cases, $t(y_i) = y_i \in \reals$ so
\begin{align*}
    \nabla \cL(\mbbeta) 
    &= \sum_{i=1}^n (y_i - \hat{y}_i) \mbx_i.
\end{align*}

And in that case the Hessian is
\begin{align*}
    \nabla^2_{\mbbeta} \cL(\mbbeta) 
    &= - \sum_{i=1}^n \nabla^2 A(\mbbeta^\top \mbx_i) \, \mbx_i \mbx_i^\top \\
    &= -\sum_{i=1}^n \Var[y_i \mid \mbx_i] \, \mbx_i \mbx_i^\top
\end{align*}

Now recall the Newton's method updates, written here in terms of the change in weights,
\begin{align*}
    \Delta \mbbeta &= - [\nabla^2 \cL(\mbbeta)]^{-1} \nabla \cL(\mbbeta) \\
    &= \left[\sum_{i=1}^n \Var[y_i \mid \mbx_i] \, \mbx_i \mbx_i^\top \right]^{-1} \left[\sum_{i=1}^n (y_i - \hat{y}_i) \mbx_i \right]
\end{align*}

Letting $w_i = \Var[y_i \mid \mbx_i]$,
\begin{align*}
    \Delta \mbbeta &=
    \left[\sum_{i=1}^n w_i \, \mbx_i \mbx_i^\top \right]^{-1} \left[ \sum_{i=1}^n (y_i - \hat{y}_i) \mbx_i \right] \\
    % &= (\mbX^\top \mbW \mbX)^{-1} [\mbX^\top \mbW \mbW^{-1} (\mby - \hat{\mby})] \\
    &= (\mbX^\top \mbW \mbX)^{-1} [\mbX^\top \mbW \hat{\mbz}]
\end{align*}
where $\mbW = \diag([w_1, \ldots, w_i])$ and $\hat{\mbz} = \mbW^{-1} (\mby - \hat{\mby})$. 

This is **iteratively reweighted least squares (IRLS)** with weights $w_i$ and targets $\hat{z}_i = \frac{y_i - \hat{y}_i}{w_i}$, both of which are functions of the current weights $\mbbeta$.

## Non-canonical case

When we choose an arbitrary covariance matrix, the expressions are a bit more complex. Let's focus on the case where $t(y_i) = y_i$ for scalar $y_i$, but allow for arbitrary mean function $f$. 
\begin{align*}
    \cL(\mbbeta) 
    &= \sum_{i=1}^n \langle y_i, \eta_i \rangle - A(\eta_i)  + c,
\end{align*}
but now $\eta_i = [\nabla A]^{-1} f(\mbbeta^\top \mbx_i)$. 

The gradient is,
\begin{align*}
    \nabla \cL(\mbbeta) 
    &= \sum_{i=1}^n \left(y_i - \hat{y}_i\right) \frac{\partial \eta_i}{\partial \mbbeta}.
\end{align*}
Applying the inverse function theorem, as above, yields,
\begin{align*}
\frac{\partial \eta_i}{\partial \mbbeta} 
&= \mathrm{Var}[Y; \eta_i]^{-1} \mbx_i = \mbx_i / w_i,
\end{align*}
and
\begin{align*}
    \nabla \cL(\mbbeta) 
    &= \sum_{i=1}^n \left(\frac{y_i - \hat{y}_i}{w_i} \right) f'(\mbbeta^\top \mbx_i) \mbx_i.
\end{align*}
