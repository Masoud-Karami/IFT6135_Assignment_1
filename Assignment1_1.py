# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:27:11 2019

@author: karm2204
"""
#Assignment1_1

# Authors:
# p1220459 Mathieu Champagne
# pXXXXXXX Masoud Karami
# p1220468 Narges Salehi

# UdeM, IFT6135, Assignment 1, H2019



import numpy as np
import random


class MNIST_NN:
#    L_RATE = 0.1

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, 
                 hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        
        self.num_inputs = num_inputs
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)
        self.init_w_in_to_hid(hidden_layer_weights)
        self.init_w_hid_to_out(output_layer_weights)

    def init_w_in_to_hid(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(np.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_w_hid_to_out(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(np.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1
                
                
class NeuronLayer:
    
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else np.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))
            
            
class Neuron:
    
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        
    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias


nn = MNIST_NN(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, 
              output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
    
    
    
    
    
    
    
    
    
