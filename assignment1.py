# Authors:
# p1220459 Mathieu Champagne
# pXXXXXXX Masoud Karami
# pXXXXXXX Narges Salehi

# UdeM, IFT6135, Assignment 1, H2019

import datetime
import numpy as np
import random
import pickle

class NN(object):
    
    def __init__(self,dims=(784,1024,2048,10),n_hidden=2,mode='train',datapath=None,model_path=None):
        
        if model_path is None:
            self.dims = dims
            self.n_hidden = n_hidden
            self.layers = []
            self.initialize_weights(n_hidden,dims)
        else:
            self.load(model_path)
    
    def initialize_weights(self,n_hidden,dims):
	# either ZERO init / NORMAL DISTRIBUTION init / GLOROT init
        for l in range(n_hidden+1):
            bias = random.random()
            self.layers.append(Layer(bias,dims[l+1]))
            for n in range(len(self.layers[l].neurons)):
                for w in range(dims[l]):
                    self.layers[l].neurons[n].weights.append(random.random()) #here instead of random, put the normal random function or whatever else
    
    def forward(self,input,labels):#..
        # forward propagation of the NN (use activation functions)
        for l in range(self.n_hidden+1):
            output = []
            for n in range(len(self.layers[l].neurons)):
                output.append(sum(np.multiply(input,self.layers[l].neurons[n].weights)))
            if l < self.n_hidden:
                input = self.activation(output)
            else:
                input = self.softmax(output)
        return input
        
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
	# DO
	#		- forward
	#		- loss
	#		- backward
	#		- update
	# UNTIL satisfied with the training / loss
    	print("")
    
    def test(self):
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
    def __init__(self, bias, nNeurons):
        self.bias = bias
        self.neurons = []
        for i in range(nNeurons):
            self.neurons.append(Neuron())
    
    def display(self, layer_num):
        print('\nLayer %i parameters :' % layer_num)
        print('Bias : %1.2f' % self.bias)
        for n in range(len(self.neurons)):
            print('Neuron %i weights :' % (n+1))
            self.neurons[n].display()

class Neuron:
    def __init__(self):
        self.weights = []
        
    def display(self):
        print(*["{0:0.2f}".format(i) for i in self.weights], sep = ", ")
        
def main():
    classifier = NN((3,2,2,2), 2)
    #classifier.save()
    #classifier = NN((1,1,1,1),2,'train',None,'stuff')
    
    classifier.display()
	
    out = classifier.forward([1,-1,5],0)
    print(out)
    # Training
	#classifier.train(dataset_train)
	
	# Test
	#classifier.test(dataset_test)
	

if __name__ == '__main__':
    main()
