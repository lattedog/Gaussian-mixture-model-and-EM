#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:34:51 2019

THis code generates a Gaussian mixtrue and use EM method to get the 
maximum likelihood solution.

@author: yuxing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math

from numpy.random import rand, randn, multivariate_normal

from sklearn.cluster import KMeans

# =============================================================================
#  create 3 Gaussian distributions, and generate the Gaussian mixture
# =============================================================================

k = 3

mean = dict.fromkeys([range(1,k+1)])
cov = dict.fromkeys([range(1,k+1)])

mean[1] = np.array([-2,-4])
cov[1] = np.array([[3, 2], [2, 2]])

mean[2] = np.array([3,5])
cov[2] = np.array([[1,0], [0, 3]])

mean[3] = np.array([-5, 8])
cov[3] = np.array([[3,-2], [-2, 4]])


N = 1000


# plot the 3 Gaussian distributions


fig, ax = plt.subplots( nrows=1, ncols=1 )
x, y = multivariate_normal(mean[1], cov[1], N).T
ax.plot(x,y,'bo')

x, y = multivariate_normal(mean[2], cov[2], N).T
ax.plot(x,y,'ro')

x, y = multivariate_normal(mean[3], cov[3], N).T
ax.plot(x,y,'go')

fig.savefig('raw data.png')



# sample from the Gaussian mixture with the following mixing coefficients

# mixture coefficients

p1 = 0.2
p2 = 0.5
p3 = 1 - p1 - p2

df = pd.DataFrame(index = range(N), columns = ["x","y",'k'])


for n in range(N):
    
    # sample according to the probabilities
    k_sample = np.random.choice(range(1, k+1), p=[p1, p2, p3])
    
    x, y = multivariate_normal(mean[k_sample], cov[k_sample], 1).T
    
   # print(k_sample, x,y)
    
    df.loc[n] = x[0], y[0], k_sample
    

df.plot(x = 'x', y = 'y', style = 'bo')
plt.savefig('Gaussian mixtue data.png')



# =============================================================================
# Code for doing the expectation-maximization method.
# =============================================================================


def gaussian_density(mu, sigma, x_i):
    '''
    This function computes the density of a Gaussian distribution at x_i
    in multidimensional space, determined by the mean vector mu and the
    covaraince matrix sigma. 
    
    mu is D*1 vector
    sigma is a D*D matrix
    
    '''
    
    assert mu.shape == x_i.shape, "The shape of the input doens't match the mean vector."
    
    D = mu.shape[0]
    
    coeff = (2 * math.pi)**(-D/2.0) * (np.linalg.det(sigma))**(-0.5)
    
    exponent = -0.5* np.dot(np.dot((x_i - mu).T, np.linalg.inv(sigma)), x_i - mu)
    
    #print(exponent.shape)
    
    density = coeff * math.exp(exponent)
    
    return density
    
    
d = gaussian_density(mean[1], cov[1], np.array([2,3]))


#2. E-step, evaluate the posterior probability of the latent variables z.
    
def E_step(df_input, N, k, D, mu, sigma, mix):
    
    r_z = np.zeros((N,k))
    
    for i in range(N):
        
        x_i = np.array(df_input.iloc[i])
        
        for j in range(k):
            r_z[i][j] = mix[j+1] * gaussian_density(mu[j+1], sigma[j+1], x_i)
        
    r_z = r_z / np.sum(r_z, axis = 1).reshape(r_z.shape[0],1)
    
    return r_z


# 3. M-step, maximize the likelihood and update the model parameters

def M_step(df_input, N, k, D, mu, sigma, mix, r_z):

    num_k = np.sum(r_z, axis = 0)
    
    mu_new = np.dot(r_z.T, df_input) / num_k.reshape(k,1)
    
    
    for i in range(1, k+1):
        Nk[i] = num_k[i-1]
        mix[i] = Nk[i] / N
        mu[i] = (mu_new[i-1]).astype(float)
        
        vec = np.sqrt(r_z[:,i-1]).reshape(N,1) * (df_input - mu[i])
        
        sigma[i] = (1 / Nk[i] * np.dot(vec.T, vec)).astype(float)
        
    return mu, sigma, mix


# 4. Evaluate the log likelihood
def eval_likeli(df_input, mu, sigma, mix):
    
    likeli = 0
    
    for n in range(N):
        
        temp = 0
        x_n = np.array(df_input.iloc[n])
        
        for i in range(1, k+1):
            temp = temp + mix[i] * gaussian_density(mu[i], sigma[i], x_n)
        
        likeli += np.log(temp)
        
    return likeli


from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, k, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, k+1):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))



# initialize the model parameters

mu = dict.fromkeys([range(1,k+1)])
sigma = dict.fromkeys([range(1,k+1)])
mix = dict.fromkeys([range(1,k+1)])

Nk = dict.fromkeys([range(1,k+1)])


# remove the last column and send the copy as the input.
df_input = df.iloc[:,:-1].copy()


# N is the number of data points and D is the dimension of the vector.
[N, D] = df_input.shape

# 1. initialize the mean vectors mu_k, sigma_k, mixing p_k. We can use k-means
# first, to give us good initial values for the mean vectors. We initialize
# the covariance matrix as identity matrix, and the mixing coefficients 
# as 1/k.

kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(df_input))
center_sk = list(kmeans.cluster_centers_)  

# trail 1
#center_sk = [np.array([1,-5]),
#             np.array([-8,0]),
#             np.array([0,-3])]

#trail 2
#center_sk = [np.array([-6,0]),
#             np.array([2,-7]),
#             np.array([-4,-8])]


for i in range(1, k+1):
    mu[i] = center_sk[i-1]
    sigma[i] = np.eye(D)
    mix[i] = 1/k
    
    
# loop to iterate until convergence    
    
old_likeli, new_likeli = 0, 10

eps = 10**-6

# number of clusters
k = 3


# count the round number until convergence
counter = 1



while( abs(old_likeli - new_likeli) >= eps):
    
    # make a plot to show the current density
    df.plot(x = 'x', y = 'y', style = 'bo')

    for pos, covar, w in zip(list(mu.values())[1:], list(sigma.values())[1:], list(mix.values())[1:]):
        draw_ellipse(pos, covar, alpha=w, k = k)
        
    
    plt.savefig("converging_{}.png".format(counter))
    
    print('Round {}, likelihood = {}'.format(counter, new_likeli))
    

    
    old_likeli = new_likeli
     
    # E-step to calculate the posterior probability for the latent variable
    r_z = E_step(df_input, N, k, D, mu, sigma, mix)
    
    # M-step to calculate the updated parameters
    mu, sigma, mix = M_step(df_input, N, k, D, mu, sigma, mix, r_z)
    
    # Evaluate the log likelihood function to see if it converges
    new_likeli = eval_likeli(df_input, mu, sigma, mix)
    
    
    
        
    counter += 1
    
    
    
# make gif from the saved figures
import glob
import moviepy.editor as mpy

gif_name = 'trail_3'
fps = 3
file_list = glob.glob('*.png') # Get all the pngs in the current directory
list.sort(file_list, key=lambda x: int(x.split('_')[1].split('.png')[0])) # Sort the images by #, this may need to be tweaked for your use case
clip = mpy.ImageSequenceClip(file_list, fps=fps)
clip.write_gif('{}.gif'.format(gif_name), fps=fps)
