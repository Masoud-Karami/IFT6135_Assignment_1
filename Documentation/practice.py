# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:03:26 2019

@author: karm2204
"""

import numpy as np
from utils import relu
def forward_propagation_with_dropout_and_batch_norm(X, parameters, keep_prob = 0.5):
"""
Implements the forward propagation: LINEAR -> RELU -> DROPOUT -> BATCHNORM.
Arguments:
X -- input data of shape (n_x, m)
parameters -- python dictionary containing your parameters:
W -- weight matrix of shape (n_y, n_x)
b -- bias vector of shape (n_y, 1)
keep_prob -- probability of keeping a neuron active during drop-out, scalar
gamma -- shifting parameter of the batch normalization layer, scalar
beta -- scaling parameter of the batch normalization layer, scalar
epsilon -- stability parameter of the batch normalization layer, scalar
Returns:
A2 -- output of the forward propagation
cache -- tuple, information stored for computing the backward propagation
"""
# retrieve parameters
W = parameters["W"]
b = parameters["b"]
keep_prob = parameters["W2"]
gamma = parameters["gamma"]
beta = parameters["beta"]
epsilon = parameters["epsilon"]
### START CODE HERE ###
# LINEAR
Z1 = np.dot(W, X) + b
# RELU
A1 = relu(Z1)
# DROPOUT
# Step 1: initialize matrix D1 = np.random.rand(..., ...)
D1 = np.random.rand(A1.shape[0], A1.shape[1])
# Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
D1 = D1 <= keep_prob
# Step 3: shut down some neurons of A1
A1 = D1 * A1
# Step 4: scale the value of neurons that haven't been shut down
A1 = A1 / keep_prob
# BATCHNORM
# Step 1: compute the vector of means
mu = np.mean(A1, axis=1)
# Step 2: compute the vector of variances
var = np.var(A1, axis=1)
# Step 3: normalize the activations (don't forget about stability)
A1_norm = (A1 - mu) / np.sqrt(var + epsilon)
# Step 4: scale and shift
A2 = gamma * X_norm + beta
### END CODE HERE ###
cache = (Z1, D1, A1, W, b, A1_norm, mu, var, gamma, beta, epsilon)
return A2, cache