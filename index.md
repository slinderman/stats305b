# Overview

Welcome to **STATS 305B**! Officially, this course is called _Applied Statistics II_. Unofficially, I'm calling it **_Models and Algorithms for Discrete Data_**, because that's what it's really about. We will cover models ranging from generalized linear models to sequential latent variable models, autoregressive models, and transformers. On the algorithm side, we will cover a few techniques for convex optimization, as well as approximate Bayesian inference algorithms like MCMC and variational inference. I think the best way to learn these concepts is to implement them from scratch, so coding will be a big focus of this course. By the end of the course, you'll have a strong grasp of classical techniques as well as modern methods for modeling discrete data.

## Logistics
Instructor: Scott Linderman <br>
TAs: Xavier Gonzalez and Leda Liang<br>
Term: Winter 2023-24 <br>
Time: Monday and Wednesday, 1:30-2:50pm <br>
Location: Room [380-380D](https://campus-map.stanford.edu/?srch=380-380D), Stanford University


## Prerequisites
Students should be comfortable with probability and statistics as well as multivariate calculus and linear algebra. This course will emphasize implementing models and algorithms, so coding proficiency with Python is required. (HW0: Python Primer will help you get up to speed.)


## Books
We will draw from a couple of sources that are freely available online:
- Agresti, Alan. Categorical Data Analysis, 2nd edition. John Wiley & Sons, 2002. [link](https://onlinelibrary.wiley.com/doi/book/10.1002/0471249688)

(We will add to this list as the course progresses.)

## Tentative Schedule

| Date         | Topic                                  | Reading |
| ------------ | -------------------------------------- | ------- |
| Jan  8, 2024 | Discrete Distributions and the Basics of Statistical Inference | {cite:p}`agresti2002categorical` Ch. 1        |
| Jan 10, 2024 | Contingency Tables                     |  {cite:p}`agresti2002categorical` Ch. 2-3       |
| Jan 15, 2024 | _MLK Day. No class_                    |         | 
| Jan 17, 2024 | [Logistic Regression](lectures/03_logreg.md) | {cite:p}`agresti2002categorical` Ch. 4-5 | 
| Jan 22, 2024 | Exponential Family GLMs                | {cite:p}`agresti2002categorical` Ch. 4-5 |
| Jan 24, 2024 | Model Selection and Diagnostics        |         | 
| Jan 29, 2024 | L1-regularized GLMs                    |         |
| Jan 31, 2024 | Bayesian Probit Regression             |         |
| Feb  5, 2024 | Generalized Linear Mixed Models        |         |
| Feb  7, 2024 | **Midterm (in class)**                 |         |
| Feb 12, 2024 | Discrete Latent Variable Models        |         |
| Feb 14, 2024 | Hidden Markov Models                   |         | 
| Feb 19, 2024 | _Presidents' Day. No class_            |         |
| Feb 21, 2024 | Mean-Field Variational Inference       |         |
| Feb 26, 2024 | Recurrent Neural Networks              |         |
| Feb 28, 2024 | Attention and Tranformers              |         |
| Mar  4, 2024 | State Space Layers (S4, S5, Mamba)     |         |
| Mar  6, 2024 | Models for Graph-Structured Data       |         |
| Mar 11, 2024 | (Discrete) Denoising Diffusion Models  |         | 
| Mar 13, 2024 | Everything Else                        |         |

## Assignments
- **Homework 0: Python Primer**
  - Released Mon, Jan 8, 2024
  - Due Fri, Jan 12, 2024 at 11:59pm

## Exams
- **Midterm Exam**: In class on Wed, Feb 7, 2024
  - You may bring a cheat sheet covering _one side_ of an 8.5x11" piece of paper

- **Final Exam**: Wed, March 20, 2024 from 3:30-6:30pm (location TBD)
  - You may bring a cheat sheet covering _both sides_ of an 8.5x11" piece of paper