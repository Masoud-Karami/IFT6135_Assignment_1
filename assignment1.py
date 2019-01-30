# Authors:
# p1220459 Mathieu Champagne
# pXXXXXXX Masoud Karami
# pXXXXXXX Narges Salehi

# UdeM, IFT6135, Assignment 1, H2019

import numpy as np
import random

class NN(object):
    
    def __init__(self,dims=(2,2,2,2),n_hidden=2,mode='train',datapath=None,model_path=None): #dims=(784,1024,2048,10)
        
        self.layers = []
        self.initialize_weights(n_hidden,dims)
        self.cool = []
    
    def initialize_weights(self,n_hidden,dims):
	# either ZERO init / NORMAL DISTRIBUTION init / GLOROT init
        for l in range(n_hidden+1):
            bias = random.random()
            self.layers.append(Layer(bias,dims[l+1]))
            for n in range(len(self.layers[l].neurons)):
                for w in range(dims[l]):
                    self.layers[l].neurons[n].weights.append(random.random()) #here instead of random, put the normal random function or whatever else
                #self.layers[l].neurons[n].display() #to visualize the weight of each neurons
    
    def forward(self,input,labels):#..
	# forward propagation of the NN (use activation functions)
	# need to figure out how to store the weights/bias (vector? list? vector of vectors?)
        print("")
        
    def activation(self,input):
    # activation function (sigmoid / ReLU / Maxout / linear / or whatever)
	# input : vector with results of pre-activation of one layer
	# output : vector with results of activation of the layer
	
	# We could add a switch case to let us decide what function we use istead of keeping the same for each layer (or having to comment and uncomment parts to test out things)
	#ReLU :    
        output = [max(i,0) for i in input]
        return output
    
    def loss(self,prediction): #..
        print("")
    
    def softmax(self,input):
	# softmax activation function (slide #17 of Artificial Neuron presentation)
	# the sum of the output vector will be = 1.
        total = sum([np.exp(i) for i in input])
        output = [np.exp(i)/total for i in input]
        return output
	
    def backward(self,cache,labels): #...
	# backward propagation
        print("")
    
    def update(self,grads): #...
	# Upgrade the weights after backward propagation
        print("")
	
    def train(self):
	# RUN	- initialization of weights
	# THEN DO
	#		- forward
	#		- loss
	#		- backward
	#		- update
	# UNTIL satisfied with the training / loss
    	print("")
    
    def test(self):
	# test the non-training dataset and output the accuracy rate...
    	print("")
        
class Layer:
    def __init__(self, bias, nNeurons):
        self.bias = bias
        self.neurons = []
        for i in range(nNeurons):
            self.neurons.append(Neuron())

class Neuron:
    def __init__(self):
        self.weights = []
        
    def display(self):
        for w in self.weights:
            print(str(w) + "\n")
        
def main():
    classifier = NN()
	
	# Training
	#classifier.train(dataset_train)
	
	# Test
	#classifier.test(dataset_test)
	

if __name__ == '__main__':
    main()
