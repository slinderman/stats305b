# Overview

Welcome to **STATS 305B**! Officially, this course is called _Applied Statistics II_. Unofficially, I'm calling it **_Models and Algorithms for Discrete Data_**, because that's what it's really about. We will cover models ranging from generalized linear models to sequential latent variable models, autoregressive models, and transformers. On the algorithm side, we will cover a few techniques for convex optimization, as well as approximate Bayesian inference algorithms like MCMC and variational inference. I think the best way to learn these concepts is to implement them from scratch, so coding will be a big focus of this course. By the end of the course, you'll have a strong grasp of classical techniques as well as modern methods for modeling discrete data.

## Logistics
Instructor: Scott Linderman <br>
TAs: Xavier Gonzalez and Leda Liang<br>
Term: Winter 2023-24 <br>
Time: Monday and Wednesday, 1:30-2:50pm <br>
Location: Room [380-380D](https://campus-map.stanford.edu/?srch=380-380D), Stanford University

**Office Hours**
- Scott: Wednesday 9-10am in the 2nd floor lounge of the Wu Tsai Neurosciences Institute
- Leda: Thursday 5-7pm in  Sequoia Hall, Room 207 (Bowker)
- Xavier: Friday 3-5pm in Building 360, Room 361A


## Prerequisites
Students should be comfortable with undergraduate probability and statistics as well as multivariate calculus and linear algebra. This course will emphasize implementing models and algorithms, so coding proficiency with Python is required. (HW0: Python Primer will help you get up to speed.)


## Books
This course will draw from a few textbooks:
- Agresti, Alan. Categorical Data Analysis, 2nd edition. John Wiley & Sons, 2002. [link](https://onlinelibrary.wiley.com/doi/book/10.1002/0471249688)
- Gelman, Andrew, et al. Bayesian Data Analysis, 3rd edition. Chapman and Hall/CRC, 2013. [link](http://www.stat.columbia.edu/~gelman/book/)
- Bishop, Christopher. Pattern Recognition and Machine Learning. Springer, 2006. [link](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)

We will also cover material from research papers.

## Schedule

_Please note that this is a **tentative** schedule. It may change slightly depending on our pace._

| Date         | Topic                                  | Reading |
| ------------ | -------------------------------------- | ------- |
| Jan  8, 2024 | [Discrete Distributions and the Basics of Statistical Inference](lectures/01_distributions.ipynb) | {cite:p}`agresti2002categorical` Ch. 1 |
| Jan 10, 2024 | [Contingency Tables](lectures/02_contingency_tables.md) | {cite:p}`agresti2002categorical` Ch. 2-3 |
| Jan 15, 2024 | _MLK Day. No class_                    |  | 
| Jan 17, 2024 | [Logistic Regression](lectures/03_logreg.md) | {cite:p}`agresti2002categorical` Ch. 4-5 | 
| Jan 22, 2024 | [Exponential Families](lectures/04_expfam.md) | {cite:p}`agresti2002categorical` Ch. 4-5 |
| Jan 24, 2024 | [Generalized Linear Models](lectures/05_glms.md) | {cite:p}`agresti2002categorical` Ch. 6 | 
| Jan 29, 2024 | [Bayesian Inference](lectures/06_bayes.md) | {cite:p}`gelman1995bayesian` Ch. 1 |
| Jan 31, 2024 | [Bayesian GLMs](lectures/07_bayes_glms_soln.ipynb) | {cite:p}`albert1993bayesian` |
| Feb 5, 2024 | [L1-regularized GLMs](lectures/08_sparse_glms.md) | {cite:p}`friedman2010regularization` and {cite:p}`lee2014proximal`|
| Feb  7, 2024 | **Midterm (in class)**                 |         |
| Feb 12, 2024 | [Mixture Models and EM](lectures/09_mixtures.md) | {cite:p}`bishop2006pattern` Ch. 9 |
| Feb 14, 2024 | [Hidden Markov Models](lectures/10_hmms.md) | {cite:p}`bishop2006pattern` Ch. 13 | 
| Feb 19, 2024 | _Presidents' Day. No class_            |         |
| Feb 21, 2024 | [Variational Autoencoders](lectures/11_vaes.md) ([Demo](lectures/11_vaes_demo.ipynb)) | {cite:p}`kingma2019introduction` Ch.1-2 |
| Feb 26, 2024 | [Recurrent Neural Networks](lectures/12_rnns.md) | {cite:p}`goodfellow2016deep` Ch. 10 |
| Feb 28, 2024 | [Tranformers](lectures/13_transformers.md) | {cite:p}`turner2023introduction` |
| Mar  4, 2024 | [State Space Layers (S4, S5, Mamba)](lectures/14_ssm.pdf) <br> _Guest lecture by [Jimmy Smith](https://jimmysmith1919.github.io/)_     | {cite:p}`smith2023simplified` and {cite:p}`gu2023mamba` |
| Mar  6, 2024 | [Random Graph Models](lectures/15_graphs.md) |         |
| Mar 11, 2024 | _Cancelled_ | |
| Mar 13, 2024 |[Denoising Diffusion Models](lectures/16_diffusion.md) | {cite:p}`turner2024denoising` | 

## Assignments
There will be 5 assignments due roughly every other Friday. They will not be equally weighted. The first one is just a primer to get you up to speed; the last one will be a bit more substantial than the rest.
- [**Homework 0: Python Primer**](assignments/hw0/hw0.ipynb)
  - Released Mon, Jan 8, 2024
  - Due Fri, Jan 12, 2024 at 11:59pm

- [**Homework 1: Logistic Regression**](assignments/hw1/hw1.ipynb)
  - Released Wed, Jan 17, 2024
  - Due Fri, Jan 26, 2024 at 11:59pm

- [**Homework 2: Bayesian GLMs**](assignments/hw2/hw2.ipynb)
  - Released Wed, Jan 31, 2024
  - Due Wed, Feb 14, 2024 at 11:59pm

- [**Homework 3: Hidden Markov Models**](assignments/hw3/hw3.ipynb)
  - Released Fri, Feb 16, 2024
  - Due Mon, Feb 26, 2024 at 11:59pm

- [**Homework 4: Large Language Models**](assignments/hw4/hw4.ipynb)
  - Released Wed, Feb 28, 2024
  - Due Fri, Mar 15, 2024 at 11:59pm


## Exams
- **Midterm Exam**: In class on Wed, Feb 7, 2024
  - You may bring a cheat sheet covering _one side_ of an 8.5x11" piece of paper

- **Final Exam**: Wed, March 20, 2024 from 3:30-6:30pm in Room 530-127
  - In addition to reviewing the midterm and the lecture notes, you may want to try these [practice problems](final/practice.pdf) (solutions are [here](final/practice_solutions.pdf)).
  - You may bring a cheat sheet covering _both sides_ of an 8.5x11" piece of paper


## Grading

Tentatively:
| Assignment    | Percentage |
| ------------- | ---------- | 
| HW 0          | 5%         |
| HW 1-3        | 15% each   |
| HW 4          | 20%        |
| Midterm       | 10%        |
| Final         | 15%        |
| Participation | 5%         |