# Statistical Learning Theory HS 2023 @ ETH Zurich
## Markov Chain Monte Carlo Sampling
<img align="right" height="130" src="https://github.com/jiaqingxie/Statistical-Learning-Theory-ETHZ/blob/main/Images/1.png"></img>
This project explores several [MCMC sampling procedures](https://www.cs.princeton.edu/courses/archive/spr06/cos598C/papers/AndrieuFreitasDoucetJordan2003.pdf) and applies them to (1) the image denoising problem by assuming an underlying Ising model, and (2) to a combinatorial optimization problem, namely TSP.
<br/><br/>
## Deterministic Annealing
<img align="right" height="130" src="https://github.com/jiaqingxie/Statistical-Learning-Theory-ETHZ/blob/main/Images/2_1.png"></img>
<img align="right" height="130" src="https://github.com/jiaqingxie/Statistical-Learning-Theory-ETHZ/blob/main/Images/2_2.png"></img>
Implementation of the DA algorithm according to [Deterministic annealing for clustering, compression, classification, regression, and related optimization problems](https://ieeexplore.ieee.org/document/726788) and empirical analysis of its phase transition behaviour.
<br/><br/>

## Histogram Clustering
<img align="right" height="130" src="https://github.com/jiaqingxie/Statistical-Learning-Theory-ETHZ/blob/main/Images/3.png"></img>
In this project, we segmented an image with histogram clustering (HC). We implemented two different methods: maximum a posterior probability (MAP) and deterministic annealing (DA) for predicting the cluster membership for each pixel.

## Constant Shift Embedding
<img align="right" height="130" src="https://github.com/jiaqingxie/Statistical-Learning-Theory-ETHZ/blob/main/Images/4.png"></img>
This project explores a tecnhique called Constant Shift Embedding which embeds pairwise clustering problems in vector spaces while preserving the cluster structure, as explained in [Optimal cluster preserving embedding of nonmetric proximity data](https://ieeexplore.ieee.org/document/1251147).
<br/><br/>

## Mean Field Approximation
<img align="right" height="130" src="https://github.com/jiaqingxie/Statistical-Learning-Theory-ETHZ/blob/main/Images/5.png"></img>
This code applies MFA, as introduced in [An Introduction to Variational Methods for Graphical Models](https://people.eecs.berkeley.edu/~jordan/papers/variational-intro.pdf), to two problem settings: (1) the 2D Ising model for image denoising, and (2) to solve Smooth-K-means, a slighly different version of K-means, in which smoothness constraints on the solution space make the problem combinatorially harder.
