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
Many real-world phenomena are described by pairwise proximity data, modeling interactions between the entities of the system. This in contrast to the more common situation where each data sample is given as a feature vector. Even though the clustering of the proximity data may be performed directly on the data matrix, there are some advantatages of  embedding the data into a vector space. For example, it enables the use of some standard preprocessing techniques such as denoising or dimensionality reduction. In this coding exercise, we will explore the tecnhique called _Constant Shift Embedding_ for restating pairwise clustering problems in vector spaces [1] while preserving the cluster structure. We will apply the algorithm described in [1] to cluster the groups of research community members based on the email correspondence matrix. The data and its description is given in [2].

### References 

[1] [Optimal cluster preserving embedding of nonmetric proximity data](https://ieeexplore.ieee.org/document/1251147)

[2] [email-Eu-core](https://snap.stanford.edu/data/email-Eu-core.html)
<br/><br/>

## Mean Field Approximation
<img align="right" height="130" src="https://github.com/jiaqingxie/Statistical-Learning-Theory-ETHZ/blob/main/Images/5.png"></img>
This code applies Mean Field Assumption, as mentioned in [An Introduction to Variational Methods for Graphical Models](https://people.eecs.berkeley.edu/~jordan/papers/variational-intro.pdf). We implemented a Meand Field Approximation approach on KMeans and evaluate/compare its performance (against Vanilla KMneas) on two problems:  
    <br> 1) The 2D Ising model (aka image reconstruction model) and  </li>
<br> 2) A Wine Dataset <b>different from the one in the second Project </b>. 
