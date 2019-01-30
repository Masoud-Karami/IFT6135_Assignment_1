# Authors:
# p1220459 Mathieu Champagne
# pXXXXXXX Masoud Karami
# pXXXXXXX Narges Salehi

# UdeM, IFT6135, Assignment 1, H2019

import numpy as np

class NN(object):
    
    def __init__(self,hidden_dims=(1024,2048),n_hidden=2,mode='train',datapath=None,model_path=None):
	
	
	
	def initialize_weights(self,n_hidden,dims):
	# either ZERO init / NORMAL DISTRIBUTION init / GLOROT init
	
	
	def forward(self,input,labels,..):
	# forward propagation of the NN (use activation functions)
	# need to figure out how to store the weights/bias (vector? list? vector of vectors?)
	
	def activation(self,input):
	# activation function (sigmoid / ReLU / Maxout / linear / or whatever)
	# input : vector with results of pre-activation of one layer
	# output : vector with results of activation of the layer
	
	# We could add a switch case to let us decide what function we use istead of keeping the same for each layer (or having to comment and uncomment parts to test out things)
	#ReLU : 
	output = [max(i,0) for i in input]
	return output
	
	def loss(self,prediction,..):
	
	
	def softmax(self,input):
	# softmax activation function (slide #17 of Artificial Neuron presentation)
	# the sum of the output vector will be = 1.
	total = sum([np.exp(i) for i in input])
	output = [np.exp(i)/total for i in input]
	return output
	
	def backward(self,cache,labels,...):
	# backward propagation
	
	
	def update(self,grads,..):
	# Upgrade the weights after backward propagation
	
	
	def train(self):
	# RUN	- initialization of weights
	# THEN DO
	#		- forward
	#		- loss
	#		- backward
	#		- update
	# UNTIL satisfied with the training / loss
	
	
	def test(self):
	# test the non-training dataset and output the accuracy rate...
	
	
def main():
	classifier = NN()
	
	# Training
	classifier.train(dataset_train)
	
	# Test
	classifier.test(dataset_test)
	
	
if __name__ == '__main__':
    main()
