# Random Graphs Models

Graphs, a.k.a. **networks**, are useful ways to represent relational data. Perhaps the most intuitive example is a social network. The **nodes** of the network represent users and the **edges** between pairs of nodes represent which users follow one another. Networks come up in many other domains as well. For example, in neuroscience we use networks to model connectivity between neurons. In chemistry, networks are used to represent how atoms bond together to form molecules. There is a rich line of work in statistics, social science, and machine learning on modeling random networks and making predictions from network-valued data.

## Definitions

Let $\mbX \in \{0,1\}^{n \times n}$ be a binary **adjacency matrix** on $n$ nodes where $X_{ij} = 1$ indicates there is an edge from node $i$ to node $j$. 
- An **undirected network** is one in which $X_{ij} = 1 \implies X_{ji} = 1$. That is, $\mbX$ is symmetric. 
- A **directed network** may have an asymmetric adjacency matrix.
- A network has no **self-loops** if $X_{ii} = 0 \forall i=1,\ldots,n$.
- A network is **simple** if it is undirected and has no self loops.
- A network is **connected** if for for all pairs of nodes $(i,j)$ there exists a path from node $i$ to node $j$.
- The **degree** of a node is its number of neighbors. In a directed graph, we may distinguish between the in-degree and out-degree.
- A graph is called **sparse** if the number of edges $E$ is $\cO(n)$. Otherwise it is called **dense.**

Sometimes we want to model graphs where the edges come with some metadata, like a weight. We can represent a weighted graph as a matrix $\mbX \in \reals^{n \times n}$ where $X_{ij}$ denotes the weight on an edge from node $i$ to node $j$, with $X_{ij} = 0$ meaning there is no edge.

These definitions are just a brief overview of terms from [**graph theory**](https://en.wikipedia.org/wiki/Graph_theory). There is an enormous literature in mathematics, computer science, probability, etc. dealing with these objects.

## Prediction and inference problems

There are many questions we might like to answer with graphs. We will focus on the following three:
1. **Edge prediction**: given observations of a subset of the adjacency matrix, predict the missing values of $\mbX$.
2. **Community discovery**: given an adjacency matrix, assign nodes to clusters based on their connectivity.
3. **Feature prediction** (or supervised learning, or simply regression): given an adjacency matrix representing a graph (of possibly varying number of nodes), predict some property of that graph. For example, given a graph representing a molecule, predict whether it will be fluorescent.

One way to answer these questions is by building a generative model for networks. It is not strictly necessary to have a generative model, as we'll talk about at the end of this lecture, but having one will allow us to answer these kinds of questions and more.

## Random networks

Let's start by considering a distribution on binary adjacency matrices with $n$ nodes. If we consider directed graphs with self loops, there are $2^{n^2}$ possible values that $\mbX$ can take on. 

:::{admonition} How many undirected graphs without self loops are there?
:class: dropdown
There are ${n \choose 2} = n(n-1)/2$ pairs of nodes, and each pair is a possible edge. Thus, there are $2^{{n \choose 2}}$ possible adjacency matrices. We can represent them by a binary adjacency matrix $\mbX$ that is symmetric and has zeros along the diagonal.
:::


Random network models are distributions over this space. One of the design considerations for such models is balancing 
- **expressivity**: the need for a model that captures a range of realistic connectivity patterns, with 
- **interpretability**: the want for a model to have relatively few parameters that govern its generative process.

## Erdős-Rényi Model

The Erdős-Rényi (ER) model for random graphs is perhaps the simplest non-trivial model. Under this model, each edge is an iid Bernoulli random variable,
\begin{align*}
X_{ij} &\iid\sim \mathrm{Bern}(\rho)
\end{align*}
with a single parameter $\rho \in [0,1]$ determining the **sparsity** of the graph. We can consider either directed or undirected versions of the model. Much of the theory of ER graphs concerns the undirected case without self loops.

The expected number of edges in an undirected model without self loops is ${n \choose 2} \rho$. The degree of node $i$ is binomial distributed,
\begin{align*}
\mathrm{deg}(i) &\sim \mathrm{Bin}(n-1, \rho)
\end{align*}
which is well approximated as $\mathrm{Po}(n\rho)$ for large graphs.

For example, Erdős and Rényi showed that $\rho=\frac{\ln n}{n}$ is a sharp threshold for connectness. Above this threshold the graph will almost surely be connected; below this threshold, there will almost surely be isolated nodes.

## Stochastic Block Model

While mathematically attractive, the ER model is a rather simple model of networks. One way to enrich our models is by introducing auxiliary variables. 

In a stochastic block model (SBM), each node $i=1,\ldots,n$ has a community assignment, $z_i \in \{1,\ldots,K\}$. Given the community assignments, the edges are conditionally independent,
\begin{align*}
X_{ij} \mid z_i=k, z_j=k' &\sim \mathrm{Bern}(\rho_{k,k'}),
\end{align*}
where $\rho_{k,k'} \in [0,1]$ for all $k=1,\ldots,K$ and $k'=1,\ldots,K$. Let $\mbR \in [0,1]^{K \times K}$ denote the matrix of community connection probabilities with entries $\rho_{k,k'}$.  Note that we have written this as a model for **directed** graphs.

:::{admonition} SBM is a generalization of the Erdős-Rényi model
:class: tip
Note that when $K=1$, the SBM reduces to the standard ER model. Thus, we can think of the SBM as a natural generalization from the one community ER model to a heterogeneous network with many communities.
:::

### Posterior inference in the SBM

Stochastic block models are like the graph version of a discrete mixture model. 
They are often used for **community discovery** &mdash; clustering nodes into groups based on their connectivity patterns. 
In this application, the community assignments are treated as latent variables drawn from a categorical prior,
\begin{align*}
z_i &\sim \mathrm{Cat}(\mbpi).
\end{align*}
We will give the model parameters conjugate priors as well,
\begin{align*}
\rho_{k,k'} &\iid{\sim} \mathrm{Beta}(\alpha, \beta) &\text{for } k,k'&= 1,\ldots,K \\
\mbpi &\sim \mathrm{Dir}(\gamma \mbone_K)
\end{align*}
The goal is to infer the posterior distribution of community assignments and parameters given the observed adjacency matrix.

One simple way to approch this problem is Gibbs sampling. In this model, the latent variables and parameters are conditionally conjugate. 
The complete conditional distributions of the community assignment variables are,
\begin{align*}
\Pr(z_i = k \mid \mbX, \mbpi, \mbR, \mbz_{\neg i}) 
&\propto p(z_i = k \mid \mbpi) \prod_{j \neq i} \mathrm{Bern}(X_{ij} \mid \rho_{k,z_j}) \\
&\propto \pi_k \prod_{j \neq i} \mathrm{Bern}(X_{ij} \mid \rho_{k,z_j}).
\end{align*}
This is a simple categorical distribution.

Given the community assignments, the connection probabilities $\mbR$ and community probabilities $\mbpi$ are conditionally conjugate too,
\begin{align*}
p(\rho_{k,k'} \mid \mbX, \mbz)
&\propto \mathrm{Beta}(\rho_{k,k'} \mid \alpha, \beta) \prod_{i=1}^n \prod_{j=1}^n \mathrm{Bern}(X_{ij} \mid \rho_{z_i,z_j}) \\
&= \mathrm{Beta}(\rho_{k,k'} \mid \alpha', \beta'),
\end{align*}
and,
\begin{align*}
p(\mbpi \mid \mbz) 
&\propto \mathrm{Dir}(\mbpi \mid \gamma \mbone_K) \prod_{i=1}^n \mathrm{Cat}(z_i \mid \mbpi) \\
&= \mathrm{Dir}(\mbpi \mid \mbgamma').
\end{align*}

:::{admonition} Exercise
Derive expressions for $\alpha'$, $\beta'$ and $\mbgamma'$.
:::

## Latent Space Models

Just as the SBM is analogous to a discrete mixture model, a latent space model (LSM) is analogous to a continuous factor model. 
In an LSM, each node is associated with a continuous latent variable $\mbz_i \in \reals^K$, and connection probabilities are determined by the inner product of those variables,
\begin{align*}
X_{ij} \mid \mbZ &\sim \mathrm{Bern}(\sigma(\mbz_i^\top \mbLambda \mbz_j + b)),
\end{align*}
where $b \in \reals$ is a bias parameter and $\mbLambda$ is a positive definite mixing matrix.

Such models are widely used in the analysis of social network data {cite:p}`hoff2002latent`. The latent variables $\mbz_i$ offer a low-dimensional summary of nodes and their relationships.

:::{admonition} Connection to Mixed Effects Models
:class: tip
Think back to [HW2: Bayesian GLMs](../assignments/hw2/hw2.ipynb), where you developed Bayesian generalized linear mixed effects models. We have the same sort of model here, but for graphs!
:::

### Posterior inference in the LSM

To complete the generative model, assume a Gaussian prior on the latent variables,
\begin{align*}
\mbz_i &\iid\sim \mathrm{N}(\mbzero, \mbI).
\end{align*}
Unlike the SBM, the conditional distributions are not conjugate in the LSM. Instead,
\begin{align*}
p(\mbz_i \mid \mbX) &\propto \mathrm{N}(\mbz_i \mid \mbzero, \mbI) \prod_{j \neq i} \mathrm{Bern}(\sigma(\mbz_i^\top \mbLambda \mbz_j + b)).
\end{align*}
One way to approach the posterior inference problem is MCMC. For example, we could use a Metropolis-Hastings algorithm to update the latent variables one at a time. This is what {cite:t}`hoff2002latent` suggest.

Alternatively, we could use  Pólya-gamma augmentation, like we did in HW2, to render the model conditionally conjugate. In that case, each latent variable would have a Gaussian conditional, holding the others fixed. 

## Latent Position Models

Finally, a latent position model (sometimes called a latent distance model) follows a similar form to the LSM above. Let $\mbz_i \in \reals^K$ denote the position of node $i$ in some latent space. Connection probabilities depend on distances in this space,
\begin{align*}
X_{ij} \mid \mbZ &\sim \mathrm{Bern}(\sigma(-\|\mbz_i - \mbz_j\|_2 + b)),
\end{align*}
{cite:t}`hoff2007modeling` showed that the LSM weakly generalizes the latent position model, in the sense that an LSM with $K+1$ dimensional factors can represent the same likelihood as a latent position model with $K$ dimensional positions. The latent position model is still an attractive model in its own right, since the learned embeddings admit a simple interpretation. It seems easier to reason about distances in a visualization than to think about inner products.

## Exchangeabile Random Graphs and Aldous-Hoover 

The models above all assume that the edges $X_{ij}$ are conditionally independent random variables given the latent variables $\mbz_i$ and $\mbz_j$ associated with the corresponding nodes.
Conditional independence assumptions like these are natural when information is limited. 

Consider modeling a collection of variables $(x_1, \ldots, x_n)$. If no information is available to order or group the variables, we must assume they are **exchangeable**:
\begin{align}
    p(x_1, \ldots, x_n) = p(x_{\pi(1)}, \ldots, x_{\pi(n)}) 
\end{align}
for any permutation $\pi$. 
The simplest exchangeable distributions assume independent and identically distributed r.v.'s,
\begin{align}
    p(x_1, \ldots, x_n) &= \prod_{i=1}^n p(x_n).
\end{align}
More generally, we may assume the variables are conditionally independent given a parameter $z$, which has been marginalized over,
\begin{align}
    p(x_1, \ldots, x_n) &= \int \left[ \prod_{i=1}^n p(x_i \mid z) \right] p(z) \dif z.
\end{align}
Marginally, $x_1, \ldots, x_n$ are **not** independent, but they are exchangeable.

### de Finetti and Aldous-Hoover
de Finetti's theorem states that as $n \to \infty$, any suitably well-behaved exchangeable distribution on $(x_1, \ldots, x_n)$ can be expressed as a mixture of independent and identical distributions, as above.
Though the theorem does not hold in the finite case, it is often cited as a motivation for conditional independence assumptions in Bayesian models.

Extensions of de Finetti's theorem have been proven for partially exchangeable arrays, like graphs {cite:p}`aldous1981representations,hoover1979relations`. A random graph is exchangeable if its distribution is invariant to permutations of the node labels. That is, the distribution of the matrix $\mbX = (X_{ij})$ is equal to the distribution of $\mbX$ with its rows and columns simultaneously permuted: $(X_{ij}) \stackrel{d}{=} (X_{\sigma(i)\sigma(j)})$ for any permutation $\sigma$ of $[n]$.

As discussed by {cite:t}`orbanz2013bayesian`, a simple graph is exchangeable if and only if there is a random symmetric function $W: [0,1]^2 \mapsto [0,1]$ such that,
\begin{align*}
(X_{ij}) \stackrel{d}{=} (\mathrm{Bern}(W(U_i, U_j)))
\end{align*} 
where $U_i$ are iid uniform random variates, independent of $W$. To see that the latent variable models above fall into this framework, we simply have to map the per-node latent variables $z_i$ to the uniform variables $U_i$ via the inverse CDF.

The random function $W$ is called a **graphon**. This object plays an important role in the theory of graph limits, which are central objects in probability theory for random networks.

## Sparse Random Graph Models

While exchangeability and invariance to node relabeling seem intuitive properties of many graphs, they have some unrealistic consequences. For example, _exchangeable random graphs are either dense or empty_ (Fact VII.2 in {cite:p}`orbanz2013bayesian`). 

To see why, consider a simple random graph on $n$ nodes. There are ${n \choose 2} = n(n-1)/2$ possible edges. The expected number of edges is,
\begin{align*}
\bbE\left[\sum_{i=1}^n \sum_{j=1}^{i-1} X_{ij} \right]  
&= \sum_{i=1}^n \sum_{j=1}^{i-1} \bbE[X_{ij}] \\
&= \sum_{i=1}^n \sum_{j=1}^{i-1} \bbE[W(U_i, U_j)] \\
&= \sum_{i=1}^n \sum_{j=1}^{i-1} \bbE[\bbE[W(U_i, U_j) \mid U_i, U_j]] \\
&= {n \choose 2} \epsilon \\
\text{where  } \epsilon &= \frac{1}{2} \int_{[0,1]^2} W(u,v) \dif u \dif v.
\end{align*}
The $\frac{1}{2}$ factor arises because $W$ is symmetric in its arguments.

There are two possibilities:
1. $\epsilon = 0$ in which case the graph is empty, or
2. $\epsilon > 0$ in which case the expected number of edges is ${n \choose 2} \epsilon = \Theta(n^2)$, which implies that the graph is almost surely dense.

### Random Walk Graph Models
So how can we generate sparse random graphs? One way is to imagine nodes arriving in a sequence. {cite:t}`bloem2018random` propose the following **random walk graph model**,

1. Initialize the graph $G_1$ with a single edge connecting two vertices.
2. For $t=2,\ldots,T$ generate $G_t$ from $G_{t-1}$ as follows:
    a. Select a vertex $V$ from $G_{t-1}$ at random
    b. With probability $\alpha$, attach a new vertex to $V$
    c. Else, run a simple random walk, starting at $V$, for a Poisson distributed number of time steps. Connect the terminal vertex $V'$ to $V$ if they are not already connected; otherwise add a new vertex to $V$.

By construction, each step of the algorithm adds one edge and at most one node, so the graph is sparse &mdash; total number of edges is $\Theta(n)$. 

Unfortunately, inference in this model is considerably more challenging. Typically, we would observe the final graph, but not the order in which the nodes arrived. Thus, we need to perform inference over the underlying permutation of nodes. Methods like sequential Monte Carlo could be used for this purpose, but it is a nontrivial inference problem {cite:p}`bloem2018random`. 

## Exponential Random Graph Models

The models discussed thus far involve relatively simple generative processes that produce rich distributions over random graphs. The marginal distributions $p(\mbX)$, integrating over the latent variables (i.e., the community assignments, latent positions, or node orderings), can be extremely difficult to write in closed form, but still we can use our toolkit of approximate Bayesian inference techniques.

Another alternative is to directly parameterize a marginal distribution over graphs. Exponential random graph models (ERGMs) assume the marginal distribution belongs to an exponential family,
\begin{align*}
p(\mbX) &= \exp \left\{ \sum_{k=1}^K \eta_k t_k(\mbX) - A(\mbeta) \right\}.
\end{align*}
The model is defined by sufficient statistics $t_k(\mbX)$ and parameterized by the natural parameters $\eta_k$. 

For example, the sufficient statistic $t_1(\mbX)$ could be the number of edges, and the sufficient statistic $t_2(\mbX)$ could count the number of triangles in the graph. (_Triadic closure_ is an important concept in social network analysis, and it captures the tendency for my friends' friends to be my friends as well.) 

In the machine learning literature, models like these are called **energy based models** because they prescribe the form of the log probability up to an unknown normalizing constant. The challenge is that for ERGMs, the normalizing constant is typically intractable,
\begin{align*}
A(\mbeta) &= \log \sum_{\mbX'} \exp \left\{ \sum_{k=1}^K \eta_k t_k(\mbX') \right\}.
\end{align*}
since the sum ranges over $2^{n^2}$ possible values of $\mbX'$.

Even generating a sample of an ERGM can be difficult. Typically, we need to resort to approximate methods like MCMC.

:::{admonition} Exercise
Come up with a Metropolis-Hastings algorithm to draw a sample from an ERGM with $n$ nodes and parameters $\mbeta$.
:::

### Parameter estimation for ERGMs

Learning the parameters of an ERGM is even more challenging. There have been many proposed algorithms. Here, we discuss a method called **persistent contrastive divergence** {cite:p}`tieleman2008training`, which was originally proposed for training another class of energy based models called restricted Boltzman machines.

The idea is to maximize the log likelihood using an approximate gradient,
\begin{align*}
\frac{\partial}{\partial \eta_k} \cL(\mbeta) 
&= \frac{\partial}{\partial \eta_k} \left[\sum_{k=1}^K \eta_k t_k(\mbX) - A(\mbeta) \right] \\
&= t_k(\mbX) - \frac{\partial}{\partial \eta_k} A(\mbeta) \\
&= t_k(\mbX) - \bbE[t_k(\mbX')],
\end{align*}
where we used the fact that gradients of the log normalizer yield expected sufficient statistics.

Persistent contrastive divergence is stochastic gradient ascent on the marginal likelihood using a simple approximation to the expectation. 
1. Initialize an MCMC chain at $\mbX^{(0)} = \mbX$, the observed graph. Initialize parameters $\mbeta^{(0)}$.
2. For iterations $i=1,2\ldots$
    
    a. Update $\mbX^{(i)}$ by starting at $\mbX^{(i-1)}$ and applying a Markov transition operator with stationary distribution $p(\mbX \mid \mbeta^{(i-1)})$. 
    
    b. Use the sample $\mbX^{(i)}$ to obtain a one-sample Monte Carlo estimate of the expected sufficient statistics, $\hat{t}_k^{(i)} = t_k(\mbX^{(i)})$.
    
    c. Update the parameters $\eta_k^{(i)} \leftarrow \eta_k^{(i-1)} + \alpha_i (t_k(\mbX) - \hat{t}_k^{(i)})$

## Graph Neural Networks

