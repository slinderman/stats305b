# Overview

Welcome to **STATS 305B**! Officially, this course is called _Applied Statistics II_. Unofficially, I call it **_Models and Algorithms for Discrete Data_**. We will cover models ranging from generalized linear models to sequential latent variable models, autoregressive models, and transformers. On the algorithm side, we will cover a few techniques for convex optimization, as well as approximate Bayesian inference algorithms like MCMC and variational inference. I think the best way to learn these concepts is to implement them from scratch, so coding will be a big focus of this course. By the end of the course, you'll have a strong grasp of classical techniques as well as modern methods for modeling discrete data.

## Logistics
Instructor: Scott Linderman <br>
TAs: Amber Hu and Michael Salerno<br>
Term: Winter 2024-25 <br>
Time: Monday and Wednesday, 1:30-2:50pm <br>
Location: [Sequoia Hall, Room 200](https://campus-map.stanford.edu/?srch=Sequoia+Hall+200), Stanford University

**Office Hours**
* Scott: Wed 10-11am, Wu Tsai Neurosciences Institute, 2nd Floor in the Theory Center
* Michael: Thu, 5-7pm, Sequoia library (Rm 105)
* Amber: Fri 1:30-3:30pm, Sequoia library (Rm 105) [except Feb 7 and 14]
    * [Feb 3 and 10 only] Mon 10am-12pm, Wu Tsai Neurosciences Institute, 2nd Floor in the Theory Center

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

| Date         | Topic                                  | Slides | Additional Reading |
| ------------ | -------------------------------------- | ------ | ------------------ |
| Mon, Jan  6, 2025 | [Basics of Probability and Statistics](lectures/01_distributions.ipynb) and [Contingency Tables](lectures/02_contingency_tables_v2.ipynb) <br> **HW0 Released** | [download](slides/01-basics.pdf)| {cite:p}`agresti2002categorical` Ch. 1-3 |
| Wed, Jan 8, 2025 | [Logistic Regression](lectures/03_logreg.ipynb) | [download](slides/03_logreg.pdf) | {cite:p}`agresti2002categorical` Ch. 4-5 | 
| Fri, Jan 10, 2025 |<span style="color:red">**HW0 Due**</span> | |
| Mon, Jan 13, 2025 | [Exponential Families](lectures/04_expfam.ipynb)  <br> **HW1 Released** | [download](slides/04_expfam.pdf) | {cite:p}`agresti2002categorical` Ch. 4-5 |
| Wed, Jan 25, 2025 | [Generalized Linear Models](lectures/05_glms_solns.ipynb) | [download](slides/05_glms.pdf) | {cite:p}`agresti2002categorical` Ch. 6 | 
| Mon, Jan 20, 2025 | _MLK Day. No class_                    |  |  |
| Wed, Feb  22, 2025 | [Sparse GLMs](lectures/06_sparse_glms_solns.ipynb) | [download](slides/06_sparse_glms.pdf) | {cite:p}`friedman2010regularization` and {cite:p}`lee2014proximal`|
| Fri, Jan 24, 2025 | <span style="color:red">**HW1 Due**</span> | | 
| Mon, Jan 27, 2025 | [Bayesian Inference](lectures/07_bayes.ipynb) <br> **HW2 Released** | [download](slides/07_bayes.pdf) | {cite:p}`gelman1995bayesian` Ch. 1 |
| Wed, Jan 29, 2025 | [Markov Chain Monte Carlo](lectures/08_mcmc.ipynb) and [Bayesian GLM Demo](lectures/08_bayes_glms_soln.ipynb) | [download](slides/08_mcmc.pdf) | | 
| Mon, Feb  3, 2025 | [Variational Inference](lectures/09_vi.ipynb) | [download](slides/09_vi.pdf) | {cite:p}`blei2017variational` | 
| Wed, Feb  5, 2025 | <span style="color:red">**Midterm Exam from 1:30-2:50pm in MCCULL 115.**</span> | [download](midterm/midterm_2024.pdf) <br> [download](midterm/midterm_2024_solns.pdf) | |
| Mon, Feb 10, 2025 | [Mixture Models and EM](lectures/10_mixtures.ipynb) | [download](slides/10_mixtures.pdf) | {cite:p}`bishop2006pattern` Ch. 9 |
| Wed, Feb 12, 2025 | [Hidden Markov Models](lectures/11_hmms.md) <br> <span style="color:red">**HW2 Due**</span>; **HW3 Released** | [download](slides/11_hmms.pdf) | {cite:p}`bishop2006pattern` Ch. 13 | 
| Mon, Feb 17, 2025 | _Presidents' Day. No class_            |     |    |
| Wed, Feb 19, 2025 | [Linear Gaussian Latent Variable Models](lectures/12_lglvms.ipynb) | [download](slides/12_lglvms.pdf) | |
| Mon, Feb 24, 2025 | [Variational Autoencoders](lectures/13_vaes.ipynb) <br> <span style="color:red">**HW3 Due**</span>; **HW4 Released**  | [download](slides/13_vaes.pdf) | {cite:p}`kingma2019introduction` Ch.1-2 |
| Wed, Feb 26, 2025 | [Transformers](lectures/14_transformers.md) | [download](slides/14_transformers.pdf) | {cite:p}`turner2023introduction` |
| Mon, Mar  3, 2025 | [Recurrent Neural Networks](lectures/15_rnns.md)  | [download](slides/15_rnns.pdf) | {cite:p}`goodfellow2016deep` Ch 9 <br>{cite:p}`smith2023simplified` and {cite:p}`gu2023mamba` |
| Wed, Mar  5, 2025 | [Denoising Diffusion Models](lectures/16_diffusion.md) | [download](slides/16_diffusion.pdf) | {cite:p}`turner2024denoising` | 
| Mon, Mar 10, 2025 | Point Processes | | |
| Wed, Mar 12, 2025 | Wrap Up | | |
| Fri, Mar 14, 2025 | <span style="color:red">**HW4 Due**</span> | |  |

## Assignments
There will be 5 assignments due roughly every other Friday. They will not be equally weighted. The first one is just a primer to get you up to speed; the last one will be a bit more substantial than the rest.
- [**Homework 0: Python Primer**](assignments/hw0/hw0.ipynb)
  - Released Mon, Jan 6, 2025
  - Due Fri, Jan 10, 2025 at 11:59pm

- [**Homework 1: Logistic Regression**](assignments/hw1/hw1.ipynb)
  - Released Mon, Jan 13, 2025
  - Due Fri, Jan 24, 2025 at 11:59pm

- [**Homework 2: Bayesian GLMs**](assignments/hw2/hw2.ipynb)
  - Released Wed, Jan 29, 2025
  - Due Wed, Feb 12, 2025 at 11:59pm

- [**Homework 3: Hidden Markov Models**](assignments/hw3/hw3.ipynb)
  - Released Wed, Feb 12, 2025
  - Due Mon, Feb 24, 2025 at 11:59pm

- [**Homework 4: Large Language Models**](assignments/hw4/hw4.ipynb)
  - Released Mon, Feb 24, 2025
  - Due Fri, Mar 14, 2025 at 11:59pm

### Late Policy
We will allow 5 late days to be used as needed throughout the quarter. 

## Exams
- **Midterm Exam**: Wed, Feb. 5 from 1:30-2:50pm in MCCULL 115
  - You may bring a cheat sheet covering _one side_ of an 8.5x11" piece of paper
  - Practice Exam: [download](midterm/midterm_2024.pdf)
  - Practice Exam Solutions: [download](midterm/midterm_2024_solns.pdf)
  - We will provide a reference of common distributions: [download](midterm/distributions.pdf)

- **Final Exam**: Wed, Mar 19 from 3:30-6:30pm in Room TBD
  <!-- - In addition to reviewing the midterm and the lecture notes, you may want to try these [practice problems](final/practice.pdf) (solutions are [here](final/practice_solutions.pdf)). -->
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
