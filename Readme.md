# Statistical Learning Theory HS 2023 @ ETH Zurich
## Markov Chain Monte Carlo Sampling
<img align="right" height="130" src="https://github.com/jiaqingxie/Statistical-Learning-Theory-ETHZ/blob/main/Images/1.png"></img>
In this project, we explored [MCMC sampling procedures](https://www.cs.princeton.edu/courses/archive/spr06/cos598C/papers/AndrieuFreitasDoucetJordan2003.pdf) . We first tackled
 the image reconstruction problem (using Ising Model) followed by the traveling salesman problem.
<br/><br/>

## Deterministic Annealing
<img align="right" height="130" src="https://github.com/jiaqingxie/Statistical-Learning-Theory-ETHZ/blob/main/Images/2_1.png"></img>
<img align="right" height="130" src="https://github.com/jiaqingxie/Statistical-Learning-Theory-ETHZ/blob/main/Images/2_2.png"></img>
Implementation of the DA algorithm according to [Deterministic annealing for clustering, compression, classification, regression, and related optimization problems](https://ieeexplore.ieee.org/document/726788). We also plot bifuration plots and embedding information.
<br/><br/>

## Histogram Clustering
<img align="right" height="130" src="https://github.com/jiaqingxie/Statistical-Learning-Theory-ETHZ/blob/main/Images/3.png"></img>
In this project, we segmented an image with histogram clustering (HC). We implemented two different methods: maximum a posterior probability (MAP) and deterministic annealing (DA) for predicting the cluster membership for each pixel.

## Constant Shift Embedding
<img align="right" height="130" src="https://github.com/jiaqingxie/Statistical-Learning-Theory-ETHZ/blob/main/Images/4.png"></img>
In this project, we explored the tecnhique called _Constant Shift Embedding_ for restating pairwise clustering problems in vector spaces while preserving the cluster structure. We applied the algorithm  to cluster the groups of research community members based on the email correspondence matrix.

## Mean Field Approximation
<img align="right" height="130" src="https://github.com/jiaqingxie/Statistical-Learning-Theory-ETHZ/blob/main/Images/5.png"></img>
This code applies Mean Field Assumption, as mentioned in [An Introduction to Variational Methods for Graphical Models](https://people.eecs.berkeley.edu/~jordan/papers/variational-intro.pdf). We implemented a Meand Field Approximation approach on KMeans and evaluate/compare its performance (against Vanilla KMneas) on two problems:  
    <br> 1) The 2D Ising model (aka image reconstruction model) and  </li>
<br> 2) A Wine Dataset <b>different from the one in the second Project </b>. 
