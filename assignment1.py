# Authors:
# p1220459 Mathieu Champagne
# pXXXXXXX Masoud Karami
# pXXXXXXX Narges Salehi

# UdeM, IFT6135, Assignment 1, H2019

#import os
import datetime
import numpy as np
#import random
import pickle
#import logging

class BadInit(Exception):
    """Something in the initialization is wrong"""
    pass
class BadInput(Exception):
    """Something is wrong with the inputs"""
    pass

class NN(object):
    
    def __init__(self,dims=(784,1024,2048,10),n_hidden=2,init_mode='GLOROT',mode='train',datapath=None,model_path=None):
        
        if model_path is None:
            if n_hidden+2 != len(dims):
                raise BadInit('The dimentions and number of hidden layers do not add up!')
            self.dims = dims
            self.n_hidden = n_hidden
            self.layers = []
            self.initialize_weights(n_hidden,dims,init_mode)
        else:
            self.load(model_path)
    
    def initialize_weights(self,n_hidden,dims,init_mode):
	# either ZERO init / NORMAL DISTRIBUTION init / GLOROT init
        for l in range(n_hidden+1):
            bias = 0.0
            self.layers.append(Layer(bias,dims[l:l+2],init_mode))
            

    
    def forward(self,input,labels):#..
        # forward propagation of the NN (use activation functions)
        
        # Input verifications
        if np.ndim(input) == 2:
            if np.shape(input)[1] != self.dims[0]:
                raise BadInput('The number of inputs do not match the dimentions!')
        else:
            if len(input) != self.dims[0]:
                raise BadInput('The number of inputs do not match the dimentions!')
        
        # propagate forward until output layer
        for l in range(self.n_hidden+1):
            try:
                # manage the case when multiple inputs
                input = np.concatenate((np.ones([np.shape(input)[0],1]),input),axis=1)
            except:
                # manage the case when only one input
                input = np.concatenate((np.ones(1),input))
            output = input.dot(self.layers[l].weights)
            if l < self.n_hidden:
                input = self.activation(output)
            else:
                return (self.softmax(output))
        
    def activation(self,input):
    # activation function (sigmoid / ReLU / Maxout / linear / or whatever)
	# input : vector with results of pre-activation of one layer
	# output : vector with results of activation of the layer
	
	# We could add a switch case to let us decide what function we use istead of keeping the same for each layer (or having to comment and uncomment parts to test out things)
	#ReLU :  
        output = np.maximum(input,np.zeros(np.shape(input)))
        return output
    
    def loss(self,prediction): #..
        print("")
    
    def softmax(self,input):
	# softmax activation function (slide #17 of Artificial Neuron presentation)
	# the sum of the output vector will be = 1.
        if np.ndim(input) == 2:
            output = [a/a.sum() for a in np.exp(input)]
        else:
            a = np.exp(input)
            output = a/a.sum()
        return output
	
    def backward(self,cache,labels): #...
	# backward propagation
        print("")
    
    def update(self,grads): #...
	# Upgrade the weights after backward propagation
        print("")
	
    def train(self,training_set,validation,batch_size,epochs):
        n_batch = int(len(training_set) / batch_size)
        for epoch in range(1,epochs+1):
            y = self.forward(training_set)
    
    def test(self,epoch=10):
	# test the non-training dataset and output the accuracy rate...
    	print("")
        
    def save(self,filename=None):
        # saves the weights and structure of the current NN
        if filename is None:
            now = datetime.datetime.now()
            filename = 'NN_' + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + 'h' + str(now.minute) + 'm' + str(now.second) + 's'
        
        with open(filename, 'wb') as f:
            pickle.dump([self.dims, self.n_hidden, self.layers], f)
            
    def load(self, filename):
        # load the weights and structure of the saved NN
        with open(filename, 'rb') as f:
            self.dims, self.n_hidden, self.layers = pickle.load(f)
    
    def display(self, display_weights=False):
        print("")
        strings = ["#inputs"]
        for i in range(1,self.n_hidden+1):
            strings.append("hid.layer." + str(i))
        strings.append("#outputs")
        strings.append("#param.")
        out_str = []
        for i in self.dims:
            out_str.append(i)
        num_param = 0
        for i in range(len(self.dims)-1):
            num_param += self.dims[i]*self.dims[i+1] + 1
        out_str.append(num_param)
        for i in range(len(out_str)):
            print('%-13s%-12i' % (strings[i], out_str[i]))
        
         #to visualize the weight of each neurons
        if display_weights:
            for l in range(self.n_hidden+1):
                self.layers[l].display(l+1)
        
class Layer:
    def __init__(self, bias, dims, init_mode):
        d = np.sqrt(6/(dims[0]+dims[1])) #uniform limits for GLOROT init
        if init_mode.upper() == 'GLOROT':
            self.weights = np.random.uniform(-d,d,[dims[0],dims[1]])
        elif init_mode.upper() == 'NORMAL':
            self.weights = np.random.randn(dims[0],dims[1]) #here instead of random, put the normal random function or whatever else
        elif init_mode.upper() == 'ZERO':
            self.weights = np.zeros([dims[0],dims[1]])
        else:
            raise BadInit('Initializing function not valid, choose between GLOROT, NORMAL and ZERO')
        self.weights = np.concatenate((np.ones([1,dims[1]])*bias,self.weights))
    
    def display(self, layer_num):
        print('\nLayer %i parameters :' % layer_num)
        print('Bias : %1.2f' % self.weights[0][0])
        print('Weights :')
        print(self.weights[1::][::])

#class Neuron:
#    def __init__(self):
#        self.weights = []
#        
#    def display(self):
#        print(*["{0:0.2f}".format(i) for i in self.weights], sep = ", ")
        
def main():
    dataset = [[0,0],[0,1],[1,0],[1,1]]
    y = [[1,0],[0,1],[0,1],[1,0]]
    
    classifier = NN((2,3,2), 1, 'GLOROT')
    #classifier.save()
    #classifier = NN((1,4,1,1),2,'GLOROT','train',None,'NN_2019_1_31_13h10m16s')
    
    display_weights = False
    classifier.display(display_weights)
	
    out = classifier.forward(dataset,0)
    
    print('\nOutputs :')
    for o in out:
        print(o)
	

if __name__ == '__main__':
    main()
