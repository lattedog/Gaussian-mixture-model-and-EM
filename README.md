# Gaussian mixture model and the EM method

# Project description
This code is written to employ expectation-maximization method to find the maximum likelihood solution for the Gaussian mixture model (GMM), which is a specfic example of estimating parameters for models with latent/unobserved variables.


## Table of contents
* [Data generation](#data-generation)
* [EM method](#EM-method)


# Data generation
The data for the GMM is created from the mixture of 3 multivariate Gaussian distribution in 2-D space.

Here we randomly choose the parameters for the 3 Gaussians to be:

```python
mean[1] = np.array([-2,-4])
cov[1] = np.array([[3, 2], [2, 2]])

mean[2] = np.array([3,5])
cov[2] = np.array([[1,0], [0, 3]])

mean[3] = np.array([-5, 8])
cov[3] = np.array([[3,-2], [-2, 4]])
```

First, we plot the 3 Gaussian distributions with different colors, shown as: 

![alt text](https://github.com/lattedog/Gaussian-mixture-model-and-EM/blob/master/raw%20data.png)

If we choose the mixing coefficients to be p1 = 0.3, p2 = 0.5, p3 = 0.2, and we sample N = 1000 data points, 

![alt text](https://github.com/lattedog/Gaussian-mixture-model-and-EM/blob/master/Gaussian%20mixtue%20data.png)


# EM method

There are many good materials online to explain the details of the EM method. I learnt this from the book *"Pattern Recognition and Machine Learning"* by *Christopher M. Bishop*. 

The steps EM method takes are:
1. Initialization of the mean vectors **mu**, covariance matrix **sigma**, and the mixing coeffcients **mix**.
2. "E step": evaluate the responsibilities or posterior probabilities of the latent variables.
3. "M step": optimize the log likelihhod of the joint distribution over model parameters, assuming the posterior probabilities of the latent variables are static. 
4. Calculate the log likelihood of the full data; check whether the process has converged. 


One thing that can make a big difference in terms of the running time to convergence is the initialization. Here we explore a few examples, initializing the mean vectors from different places.

## Trial 1:

3 randomly chosen points were chosen for the mean vectors of the 3 Gaussian distributions. 

```python
center_sk = [np.array([1,-5]),
             np.array([-8,0]),
             np.array([0,-3])]
```
             
The gif below depicts the converging process.

![](https://github.com/lattedog/Gaussian-mixture-model-and-EM/blob/master/trial1/trial_1.gif)


## Trial 2:

Another 3 randomly chosen points were chosen for the mean vectors of the 3 Gaussian distributions. 


```python
center_sk = [np.array([-6,0]),
             np.array([2,-7]),
             np.array([-4,-8])]
```

The gif below depicts the converging process.

![](https://github.com/lattedog/Gaussian-mixture-model-and-EM/blob/master/trial2/trial_2.gif)


## Trial 3:

K-means is employed first on the data to obtain the estimate of the 3 mean vectors, and these are used to initialize model parameters. 

```python
center_sk = [np.array([2.927, 4.978]),
             np.array([-5.065, 7.803]),
             np.array([-1.973, -3.978])]
```
As seen below, the convergence process is much faster.  

![](https://github.com/lattedog/Gaussian-mixture-model-and-EM/blob/master/trial3/trial_3.gif)
